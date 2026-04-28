import asyncio
import logging
from abc import ABC, abstractmethod
from typing import cast, override

import anthropic
import backoff
import ollama
import openai
from anthropic import AsyncAnthropic
from google import genai as google_genai
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.config import (
    CloudModelConfig,
    InferenceConcurrencyConfig,
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
    temperature: float
    top_p: float
    top_k: int


class InferenceOutput(BaseModel):
    stats: InferenceStats
    params: InferenceParams
    text: str
    success: bool = True
    attempts: int = 1


class InferenceModel(ABC):
    """
    Base class that defines the minimal API for text generation.

    `generate` is async so all backends share one shape and the pipeline
    can dispatch tasks via `asyncio.gather` without caring whether the
    underlying call is local or cloud. Per-backend concurrency is enforced
    inside `generate` via a shared semaphore — callers don't see it.
    """

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
        async with self._sem:
            res = await _ollama_async_client.generate(
                prompt=prompt,
                model=self.cfg.model_id,
                options={
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "top_k": params.top_k,
                },
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

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        async def create_completion():
            assert openai_client
            return await openai_client.chat.completions.create(
                model=self.cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
            )

        async with self._sem:
            result = await create_completion()

        return InferenceOutput(
            params=InferenceParams(
                temperature=params.temperature, top_p=1.0, top_k=0
            ),  # top_k/top_n not supported
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

        @backoff.on_exception(backoff.expo, anthropic.RateLimitError)
        async def create_completion():
            assert anthropic_client
            return await anthropic_client.messages.create(
                max_tokens=32_000,  # might need to be changed
                model=self.cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k if params.top_k != 0 else anthropic.omit,
            )

        async with self._sem:
            result = await create_completion()

        return InferenceOutput(
            params=params,
            text=result.content[0].text if result.content[0].type == "text" else "",
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
        raise NotImplementedError()


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

