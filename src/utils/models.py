import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, cast, override

import anthropic
import backoff
import ollama
import openai
from anthropic import AsyncAnthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from google import genai as google_genai
from google.genai import errors as google_errors
from google.genai import types as google_types
from openai import AsyncOpenAI
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)
from pydantic import BaseModel
from tqdm import tqdm

from src.config import (
    CloudModelConfig,
    InferenceConcurrencyConfig,
    InferenceParamsConfig,
    LocalModelConfig,
    ModelConfig,
    OllamaLocalModelParams,
)

_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(key: str, size: int) -> asyncio.Semaphore:
    """Return a process-wide semaphore for the given backend key.

    The first call wins on size: subsequent calls with the same key reuse the
    existing semaphore so all model instances of the same backend share one
    cap, regardless of how many model entries reference it.
    """
    sem = _semaphores.get(key)
    if sem is None:
        sem = asyncio.Semaphore(size)
        _semaphores[key] = sem
    return sem


class InferenceStats(BaseModel):
    prompt_tokens: int = -1
    completion_tokens: int = -1
    total_tokens: int = -1


class InferenceParams(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = None
    text_verbosity: Literal["low", "medium"] | None = None
    reasoning_budget: int | None = None
    # Label tying a row back to the truncation profile that produced it; only
    # populated for Ollama, where (top_p, min_p) move together as a single cell.
    truncation_name: str | None = None


class InferenceOutput(BaseModel):
    stats: InferenceStats
    params: InferenceParams
    text: str
    success: bool = True
    attempts: int = 1
    errors: list[str] = []


class InferenceModel(ABC):
    """
    Base class that defines the minimal API for text generation.

    `generate` is async so all backends share one shape and the pipeline
    can dispatch tasks via `asyncio.gather` without caring whether the
    underlying call is local or cloud. Per-backend concurrency is enforced
    inside `generate` via a shared semaphore — callers don't see it.
    """

    # Subclasses set these. The pipeline reads them to project the YAML
    # `defaults` and `constants` blocks into per-backend matrices, so each
    # backend only sweeps axes its API actually accepts.
    BACKEND_KEY: str = ""
    CONSUMES_DEFAULTS: frozenset[str] = frozenset()
    CONSUMES_CONSTANTS: frozenset[str] = frozenset()

    def __init__(self, name: str, sem: asyncio.Semaphore) -> None:
        self.name = name
        self._sem = sem

    @abstractmethod
    async def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        """
        Return a single generated text for the given prompt.

        Parameters
        ----------
        prompt: str
            Input prompt.
        params: InferenceParams
            Model-specific parameters.

        Returns
        -------
        InferenceOutput
            Generated text and stats.
        """
        ...


_ollama_async_client: ollama.AsyncClient | None = None


class OllamaInferenceModel(InferenceModel):
    """
    Local inference using Ollama.
    """

    BACKEND_KEY = "ollama"
    CONSUMES_DEFAULTS = frozenset({"temperature"})
    CONSUMES_CONSTANTS = frozenset({"seed", "top_k", "repetition_penalty"})

    def __init__(
        self, name: str, cfg: OllamaLocalModelParams, sem: asyncio.Semaphore
    ) -> None:
        super().__init__(name, sem)
        self.cfg = cfg
        self._pull()

        global _ollama_async_client
        if _ollama_async_client is None:
            _ollama_async_client = ollama.AsyncClient()

    def _pull(self) -> None:
        logging.info(f"Pulling Ollama model {self.cfg.model_id}")

        bars: dict[str, tqdm] = {}
        for event in ollama.pull(self.cfg.model_id, stream=True):
            digest = getattr(event, "digest", None) or ""
            total = getattr(event, "total", None) or 0
            completed = getattr(event, "completed", None) or 0
            status = getattr(event, "status", "") or ""

            if digest and total:
                bar = bars.get(digest)
                if bar is None:
                    bar = tqdm(
                        total=total,
                        desc=digest[:19],
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        dynamic_ncols=True,
                        colour="cyan",
                    )
                    bars[digest] = bar
                bar.n = completed
                bar.refresh()
            elif status:
                logging.debug(f"ollama pull: {status}")

        for bar in bars.values():
            bar.close()

    @override
    async def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        assert _ollama_async_client is not None
        # Ollama's option keys differ from ours in one place: `repeat_penalty`
        # is what Ollama calls our `repetition_penalty`.
        raw_options: dict[str, float | int | None] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "seed": params.seed,
            "repeat_penalty": params.repetition_penalty,
        }
        if params.min_p is not None:
            raw_options["min_p"] = params.min_p
        options: dict[str, float | int] = {
            k: v for k, v in raw_options.items() if v is not None
        }

        async with self._sem:
            res = await _ollama_async_client.generate(
                prompt=prompt,
                model=self.cfg.model_id,
                options=options,
            )

        completion_tokens = res.eval_count or -1
        prompt_tokens = res.prompt_eval_count or -1

        return InferenceOutput(
            params=params,
            text=res.response,
            stats=InferenceStats(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=completion_tokens + prompt_tokens,
            ),
        )


openai_client: AsyncOpenAI | None = None


class OpenAIInferenceModel(InferenceModel):
    """
    Cloud inference using the OpenAI API.
    """

    BACKEND_KEY = "openai"
    CONSUMES_DEFAULTS = frozenset({"temperature", "top_p"})
    CONSUMES_CONSTANTS = frozenset({"seed"})

    def __init__(
        self, name: str, cfg: CloudModelConfig, sem: asyncio.Semaphore
    ) -> None:
        super().__init__(name, sem)

        global openai_client
        if not openai_client:
            openai_client = AsyncOpenAI()

        self.cfg = cfg

    @override
    async def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        # Build kwargs incrementally so axes the matrix did not sweep don't
        # show up in the request — the SDK rejects None for some of these.
        kwargs: CompletionCreateParamsNonStreaming = {
            "model": self.cfg.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.reasoning_effort is not None:
            # Our domain literal includes "xhigh" (a GPT-5-tier setting the
            # SDK's literal hasn't caught up to yet); cast through to the
            # SDK's narrower type.
            kwargs["reasoning_effort"] = cast(
                "Literal['minimal', 'low', 'medium', 'high']",
                params.reasoning_effort,
            )
        if params.text_verbosity is not None:
            kwargs["verbosity"] = params.text_verbosity

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        async def create_completion():
            assert openai_client
            return await openai_client.chat.completions.create(**kwargs)

        async with self._sem:
            result = await create_completion()

        return InferenceOutput(
            params=params,
            text=result.choices[0].message.content or "",
            stats=InferenceStats(
                completion_tokens=result.usage.completion_tokens,
                prompt_tokens=result.usage.prompt_tokens,
                total_tokens=result.usage.total_tokens,
            )
            if result.usage
            else InferenceStats(),
        )


anthropic_client: AsyncAnthropic | None = None


class AnthropicInferenceModel(InferenceModel):
    """
    Cloud inference using the Anthropic API.
    """

    BACKEND_KEY = "anthropic"
    CONSUMES_DEFAULTS = frozenset({"temperature", "top_p"})
    # Anthropic's API does not accept a `seed` parameter today — that's why
    # `seed` is absent here even though the constants block declares one.
    CONSUMES_CONSTANTS = frozenset({"top_k"})

    # Output budget held over after the thinking budget. Anthropic requires
    # `max_tokens > thinking.budget_tokens`; this gives the model a fixed
    # answer-budget regardless of how deep it thought.
    _OUTPUT_TOKEN_HEADROOM = 32_000

    def __init__(
        self, name: str, cfg: CloudModelConfig, sem: asyncio.Semaphore
    ) -> None:
        super().__init__(name, sem)

        global anthropic_client
        if not anthropic_client:
            anthropic_client = AsyncAnthropic()

        self.cfg = cfg

    @override
    async def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        thinking_on = params.reasoning_budget is not None
        # Anthropic forces temperature=1.0 when extended thinking is enabled.
        applied_temperature = 1.0 if thinking_on else params.temperature
        max_tokens = (
            params.reasoning_budget + self._OUTPUT_TOKEN_HEADROOM
            if params.reasoning_budget is not None
            else self._OUTPUT_TOKEN_HEADROOM
        )

        kwargs: MessageCreateParamsNonStreaming = {
            "model": self.cfg.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if applied_temperature is not None:
            kwargs["temperature"] = applied_temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.top_k:
            kwargs["top_k"] = params.top_k
        if params.reasoning_budget is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.reasoning_budget,
            }

        @backoff.on_exception(backoff.expo, anthropic.RateLimitError)
        async def create_completion():
            assert anthropic_client
            return await anthropic_client.messages.create(**kwargs)

        async with self._sem:
            result = await create_completion()

        # Surface the *applied* temperature so downstream analysis joins on
        # what the model actually saw, not the matrix cell we asked for.
        out_params = params.model_copy(update={"temperature": applied_temperature})

        # Anthropic prepends a `thinking` block before the text when extended
        # thinking is on; pick the first text block instead of indexing [0].
        text = next(
            (b.text for b in result.content if b.type == "text"),
            "",
        )

        return InferenceOutput(
            params=out_params,
            text=text,
            stats=InferenceStats(
                completion_tokens=result.usage.output_tokens,
                prompt_tokens=result.usage.input_tokens,
                total_tokens=result.usage.input_tokens + result.usage.output_tokens,
            )
            if result.usage
            else InferenceStats(),
        )


google_client: google_genai.Client | None = None


class GoogleInferenceModel(InferenceModel):
    """
    Cloud inference using the Google API.
    """

    BACKEND_KEY = "google"
    CONSUMES_DEFAULTS = frozenset({"temperature", "top_p"})
    CONSUMES_CONSTANTS = frozenset({"seed", "top_k"})

    def __init__(
        self, name: str, cfg: CloudModelConfig, sem: asyncio.Semaphore
    ) -> None:
        super().__init__(name, sem)

        global google_client
        if not google_client:
            google_client = google_genai.Client()

        self.cfg = cfg

    @override
    async def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        config: google_types.GenerateContentConfigDict = {
            "max_output_tokens": 32_000,
        }
        if params.temperature is not None:
            config["temperature"] = params.temperature
        if params.top_p is not None:
            config["top_p"] = params.top_p
        if params.top_k:
            config["top_k"] = params.top_k
        if params.seed is not None:
            config["seed"] = params.seed
        if params.reasoning_budget is not None:
            config["thinking_config"] = {"thinking_budget": params.reasoning_budget}

        @backoff.on_exception(
            backoff.expo,
            google_errors.APIError,
            giveup=lambda e: getattr(e, "code", None) != 429,
        )
        async def create_completion():
            assert google_client
            return await google_client.aio.models.generate_content(
                model=self.cfg.model_id,
                contents=prompt,
                config=config,
            )

        async with self._sem:
            result = await create_completion()

        usage = result.usage_metadata
        return InferenceOutput(
            params=params,
            text=result.text or "",
            stats=InferenceStats(
                completion_tokens=usage.candidates_token_count or -1,
                prompt_tokens=usage.prompt_token_count or -1,
                total_tokens=usage.total_token_count or -1,
            )
            if usage
            else InferenceStats(),
        )


def get_model(
    cfg: ModelConfig, concurrency: InferenceConcurrencyConfig
) -> InferenceModel:
    """Return an inference model instance.

    The semaphore enforcing concurrency for this model is shared across all
    instances of the same backend (driver for local, provider for cloud), so
    multiple model entries pointing at the same backend never exceed that
    backend's cap collectively.
    """
    match cfg.meta.kind:
        case "local":
            local_meta = cast(LocalModelConfig, cfg.meta)
            sem = _get_semaphore(local_meta.params.driver, concurrency.ollama)
            return OllamaInferenceModel(cfg.name, local_meta.params, sem)
        case "cloud":
            cloud_meta = cast(CloudModelConfig, cfg.meta)
            match cloud_meta.provider:
                case "openai":
                    sem = _get_semaphore("openai", concurrency.openai)
                    return OpenAIInferenceModel(cfg.name, cloud_meta, sem)
                case "anthropic":
                    sem = _get_semaphore("anthropic", concurrency.anthropic)
                    return AnthropicInferenceModel(cfg.name, cloud_meta, sem)
                case "google":
                    sem = _get_semaphore("google", concurrency.google)
                    return GoogleInferenceModel(cfg.name, cloud_meta, sem)
                case _:
                    raise ValueError(f"Unknown provider {cloud_meta.provider}")
    raise ValueError()


def expand_for_backend(
    cfg: InferenceParamsConfig,
    model: InferenceModel,
) -> list[InferenceParams]:
    """Materialise the per-model sweep matrix as a list of `InferenceParams`.

    The YAML's `defaults` and `per_backend.<key>` blocks are projected through
    `model.CONSUMES_DEFAULTS` / `model.CONSUMES_CONSTANTS` so each backend only
    sweeps axes its API actually consumes. Constants are stamped onto every
    cell. Ollama's `truncation` profiles are a tagged tuple axis: each entry
    contributes both `top_p` and `min_p` (and a label) as a single cell.
    """
    sweep_axes: list[list[tuple[str, Any]]] = []

    defaults_dump = cfg.defaults.model_dump()
    for axis in model.CONSUMES_DEFAULTS:
        sweep_axes.append([(axis, v) for v in defaults_dump[axis]])

    pb = getattr(cfg.per_backend, model.BACKEND_KEY).model_dump()
    for axis, values in pb.items():
        if axis == "truncation":
            sweep_axes.append([("__truncation__", profile) for profile in values])
        else:
            sweep_axes.append([(axis, v) for v in values])

    profiles: list[InferenceParams] = []
    for combo in itertools.product(*sweep_axes):
        fields: dict[str, Any] = {}
        for axis, value in combo:
            if axis == "__truncation__":
                fields["top_p"] = value["top_p"]
                fields["min_p"] = value["min_p"]
                fields["truncation_name"] = value["name"]
            else:
                fields[axis] = value
        for c in model.CONSUMES_CONSTANTS:
            fields[c] = getattr(cfg.constants, c)
        profiles.append(InferenceParams(**fields))
    return profiles


def extract_final_answer(model: str, text: str) -> str:
    if model.startswith("gpt-oss"):
        return extract_gpt_oss(text)
    return text


def extract_gpt_oss(text: str) -> str:
    marker = "<|start|>assistant<|channel|>final<|message|>"
    index = text.find(marker)
    if index == -1:
        return ""
    return text[index + len(marker) :]
