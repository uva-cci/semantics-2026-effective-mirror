import logging
from abc import ABC, abstractmethod
from typing import cast, override

import anthropic
import backoff
import ollama
import openai
from anthropic import Anthropic
from google import genai as google_genai
from openai import OpenAI
from pydantic import BaseModel

from src.config import (
    CloudModelConfig,
    Config,
    LocalModelConfig,
    ModelConfig,
    OllamaLocalModelParams,
)


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
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
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


class OllamaInferenceModel(InferenceModel):
    """
    Local inference using Ollama.
    """

    def __init__(self, name: str, cfg: OllamaLocalModelParams) -> None:
        super().__init__(name)
        self.cfg = cfg

    @override
    def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        res = ollama.generate(
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


openai_client: OpenAI | None = None


class OpenAIInferenceModel(InferenceModel):
    """
    Cloud inference using the OpenAI API.
    """

    def __init__(self, name: str, cfg: CloudModelConfig) -> None:
        super().__init__(name)

        global openai_client
        if not openai_client:
            openai_client = OpenAI()

        self.cfg = cfg

    def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def create_completion():
            assert openai_client
            return openai_client.chat.completions.create(
                model=self.cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
            )

        result = create_completion()

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


anthropic_client: Anthropic | None = None


class AnthropicInferenceModel(InferenceModel):
    """
    Cloud inference using the Anthropic API.
    """

    def __init__(self, name: str, cfg: CloudModelConfig) -> None:
        super().__init__(name)

        global anthropic_client
        if not anthropic_client:
            anthropic_client = Anthropic()

        self.cfg = cfg

    def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:

        @backoff.on_exception(backoff.expo, anthropic.RateLimitError)
        def create_completion():
            assert anthropic_client
            return anthropic_client.messages.create(
                max_tokens=32_000,  # might need to be changed
                model=self.cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k if params.top_k != 0 else anthropic.omit,
            )

        result = create_completion()

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

    def __init__(self, name: str, cfg: CloudModelConfig) -> None:
        super().__init__(name)

        global google_client
        if not google_client:
            google_client = google_genai.Client()

        self.cfg = cfg

    def generate(self, prompt: str, params: InferenceParams) -> InferenceOutput:
        raise NotImplementedError()


def get_model(cfg: ModelConfig) -> InferenceModel:
    """
    Return an inference model instance.

    Parameters
    ----------
    backend : str
        "local" or "openai".
    config : dict
        Arguments forwarded to the concrete constructor.
    """
    match cfg.meta.kind:
        case "local":
            local_meta = cast(LocalModelConfig, cfg.meta)
            return OllamaInferenceModel(cfg.name, local_meta.params)
        case "cloud":
            cloud_meta = cast(CloudModelConfig, cfg.meta)
            match cloud_meta.provider:
                case "openai":
                    return OpenAIInferenceModel(cfg.name, cloud_meta)
                case "anthropic":
                    return AnthropicInferenceModel(cfg.name, cloud_meta)
                case _:
                    raise ValueError(f"Unknown provider {cloud_meta.provider}")
    raise ValueError()


async def download_models(cfg: Config) -> None:
    logging.info("Pulling Ollama models.")

    for model in cfg.models:
        logging.info(f"Pulling model: {model.name}")

        if model.meta.kind != "local":
            logging.warning(f"Skipping cloud model entry: {model.name}")
            continue

        assert model.meta.params.model_id, "ollama driver requires model_id"
        logging.info(f"Pulling ollama: {model.meta.params.model_id}")
        ollama.pull(model.meta.params.model_id)
        logging.info(f"✓ Done: {model.meta.params.model_id}")

    logging.info("All model pulls finished.")


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

