"""Lifecycle management for a pipeline-spawned `llama-server` process.

`LlamaCppInferenceModel` is a thin OpenAI-SDK client; this module is what
makes the server it talks to actually exist. One `LlamaServerManager`
instance owns one `llama-server` subprocess for the duration of a single
model's task group: it picks a free port, builds the CLI args from
`LlamaCppServerArgs`, waits for `/v1/models` to come up (and to advertise
the configured `model_id`), then tears the process down on exit.

The async context manager is the primary path. The module-level
`_live_managers` set + `atexit` hook is a safety net for the case where
the pipeline crashes before `__aexit__` runs — orphaned `llama-server`
children pin GPU/RAM, so we always want a way to take them down.
"""
import asyncio
import atexit
import logging
import os
import shutil
import socket
from collections import deque
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from src.config import LlamaCppLocalModelParams


_live_managers: "set[LlamaServerManager]" = set()


def _atexit_cleanup() -> None:
    for mgr in list(_live_managers):
        mgr._sync_kill()


atexit.register(_atexit_cleanup)


def _allocate_free_port() -> int:
    """Pick an unused TCP port by letting the kernel assign one.

    There's a tiny TOCTOU window between releasing the socket here and
    `llama-server` re-binding it. Acceptable for a single-process pipeline
    on a single host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServerManager:
    """Async context manager wrapping a `llama-server` subprocess.

    On enter: allocate a port, spawn the binary in its own session, stream
    its stdout/stderr to the logger, poll `/v1/models` until the configured
    `model_id` is registered, and return the resulting `base_url`. On exit:
    terminate the whole process group and wait for it to die.
    """

    # How long to wait for `/v1/models` to start responding. First-time
    # `-hf <repo>:<quant>` launches download multi-GB GGUFs into
    # `~/.cache/llama.cpp/`, so this needs to be generous.
    _READINESS_TIMEOUT_S: float = 30 * 60.0
    _READINESS_INITIAL_BACKOFF_S: float = 0.5
    _READINESS_MAX_BACKOFF_S: float = 5.0

    # Time we give SIGTERM before escalating to SIGKILL on the process group.
    _TERMINATE_TIMEOUT_S: float = 10.0

    # Lines of stderr to retain for inclusion in the "exited before ready"
    # error message. Bounded to keep memory predictable for long-lived runs
    # whose servers print verbose status output.
    _STDERR_TAIL_LINES: int = 60

    def __init__(
        self,
        cfg: LlamaCppLocalModelParams,
        parallel: int,
    ) -> None:
        if cfg.base_url is not None:
            raise ValueError(
                "LlamaServerManager is for managed mode (cfg.base_url is None). "
                f"Got base_url={cfg.base_url!r}."
            )
        self.cfg = cfg
        self.parallel = max(1, parallel)
        self.port: int | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=self._STDERR_TAIL_LINES)

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("base_url accessed before manager started")
        return f"http://{self.cfg.server.host}:{self.port}/v1"

    def _build_argv(self) -> list[str]:
        s = self.cfg.server

        binary = s.binary
        if not os.path.isabs(binary):
            resolved = shutil.which(binary)
            if resolved is None:
                raise RuntimeError(
                    f"`{binary}` not found on PATH. Install llama.cpp or set "
                    "`server.binary` to an absolute path."
                )
            binary = resolved

        argv: list[str] = [
            binary,
            "-hf", self.cfg.model_id,
            "--host", s.host,
            "--port", str(self.port),
            "--parallel", str(self.parallel),
        ]
        if s.context_size is not None:
            argv += ["-c", str(s.context_size)]
        if s.gpu_layers is not None:
            argv += ["-ngl", str(s.gpu_layers)]
        if s.batch_size is not None:
            argv += ["-b", str(s.batch_size)]
        if s.threads is not None:
            argv += ["-t", str(s.threads)]
        if s.flash_attn is not None:
            # Modern `llama-server` builds expect a value here ('on', 'off',
            # 'auto'); the bare-flag form was retired. Map the bool through.
            argv += ["--flash-attn", "on" if s.flash_attn else "off"]
        argv += list(s.extra_args)
        return argv

    async def __aenter__(self) -> str:
        self.port = _allocate_free_port()
        argv = self._build_argv()

        logging.info(
            f"Starting llama-server for {self.cfg.model_id} on "
            f"{self.cfg.server.host}:{self.port} (--parallel {self.parallel})"
        )
        logging.debug(f"llama-server argv: {argv}")

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        _live_managers.add(self)

        self._stdout_task = asyncio.create_task(
            self._forward_stream(self._proc.stdout, logging.INFO),
            name=f"llama-server[{self.cfg.model_id}]:stdout",
        )
        self._stderr_task = asyncio.create_task(
            self._forward_stream(self._proc.stderr, logging.WARNING, capture=True),
            name=f"llama-server[{self.cfg.model_id}]:stderr",
        )

        try:
            await self._wait_until_ready()
        except BaseException:
            # Don't leak the subprocess if readiness fails or is cancelled.
            await self._terminate()
            raise

        logging.info(
            f"llama-server ready at {self.base_url} (model {self.cfg.model_id!r})"
        )
        return self.base_url

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self._terminate()

    async def _forward_stream(
        self,
        stream: asyncio.StreamReader | None,
        level: int,
        *,
        capture: bool = False,
    ) -> None:
        if stream is None:
            return
        prefix = f"[llama-server {self.cfg.model_id}]"
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode(errors="replace").rstrip()
                logging.log(level, f"{prefix} {text}")
                if capture:
                    self._stderr_tail.append(text)
        except asyncio.CancelledError:
            return

    async def _wait_until_ready(self) -> None:
        api_key = (
            os.environ[self.cfg.api_key_env]
            if self.cfg.api_key_env is not None
            else "EMPTY"
        )
        client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._READINESS_TIMEOUT_S
        backoff = self._READINESS_INITIAL_BACKOFF_S

        while True:
            assert self._proc is not None
            if self._proc.returncode is not None:
                # Give the stderr forwarder a brief window to drain any
                # remaining buffered output before we read the tail — when
                # the process dies fast, the OS pipe can still hold lines
                # that haven't been logged yet.
                if self._stderr_task is not None and not self._stderr_task.done():
                    await asyncio.wait(
                        {self._stderr_task}, timeout=0.5
                    )
                tail = (
                    "\n  ".join(self._stderr_tail)
                    if self._stderr_tail
                    else "(no stderr captured)"
                )
                raise RuntimeError(
                    f"llama-server exited (code {self._proc.returncode}) "
                    f"before becoming ready. Last stderr lines:\n  {tail}"
                )

            try:
                served = {m.id for m in (await client.models.list()).data}
            except (openai.OpenAIError, httpx.HTTPError, OSError) as e:
                # `httpx.HTTPError` covers `ConnectError` / `RemoteProtocolError`
                # that the openai SDK occasionally lets through unwrapped while
                # the server is mid-startup (port not yet listening, GGUF still
                # loading); `OSError` catches the raw `ConnectionRefusedError`
                # path. All transient — keep polling until the deadline or the
                # subprocess exits.
                logging.debug(
                    f"llama-server probe at {self.base_url} not ready: "
                    f"{type(e).__name__}: {e}"
                )
                served = None

            if served is not None:
                if self.cfg.model_id in served:
                    return
                # Server up but advertising a different model id — fail fast,
                # because retrying won't change what's loaded.
                raise ValueError(
                    f"Model {self.cfg.model_id!r} not loaded on llama-server "
                    f"at {self.base_url}; server reports: {sorted(served)}."
                )

            if loop.time() >= deadline:
                raise TimeoutError(
                    f"llama-server at {self.base_url} did not become ready "
                    f"within {self._READINESS_TIMEOUT_S:.0f}s."
                )

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._READINESS_MAX_BACKOFF_S)

    async def _terminate(self) -> None:
        if self._proc is None:
            _live_managers.discard(self)
            return

        if self._proc.returncode is None:
            pgid = self._safe_pgid()
            if pgid is not None:
                try:
                    os.killpg(pgid, 15)  # SIGTERM
                except ProcessLookupError:
                    pass

            try:
                await asyncio.wait_for(
                    self._proc.wait(), timeout=self._TERMINATE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                logging.warning(
                    f"llama-server for {self.cfg.model_id} did not exit on "
                    f"SIGTERM within {self._TERMINATE_TIMEOUT_S}s; sending SIGKILL"
                )
                if pgid is not None:
                    try:
                        os.killpg(pgid, 9)  # SIGKILL
                    except ProcessLookupError:
                        pass
                await self._proc.wait()

        for t in (self._stdout_task, self._stderr_task):
            if t is not None and not t.done():
                t.cancel()
        for t in (self._stdout_task, self._stderr_task):
            if t is not None:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

        logging.info(
            f"llama-server for {self.cfg.model_id} stopped "
            f"(exit code {self._proc.returncode})"
        )
        _live_managers.discard(self)

    def _safe_pgid(self) -> int | None:
        if self._proc is None or self._proc.pid is None:
            return None
        try:
            return os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return None

    def _sync_kill(self) -> None:
        """Best-effort teardown from `atexit`, where no event loop is running.

        Sends SIGTERM to the process group; we deliberately don't wait —
        `atexit` is already past the point where blocking matters.
        """
        if self._proc is None or self._proc.returncode is not None:
            _live_managers.discard(self)
            return
        pgid = self._safe_pgid()
        if pgid is None:
            return
        try:
            os.killpg(pgid, 15)
        except (ProcessLookupError, PermissionError):
            pass
        _live_managers.discard(self)
