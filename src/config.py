import logging
import pathlib
from enum import StrEnum
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator


class DSLConfig(BaseModel):
    name: str
    examples: FilePath
    schema_path: FilePath = Field(alias="schema")


class OllamaLocalModelParams(BaseModel):
    driver: Literal['ollama']
    model_id: str


class LlamaCppLocalModelParams(BaseModel):
    driver: Literal['llamacpp']
    model_id: str
    # llama-server URL, e.g. `http://localhost:8080/v1`. Required: the OpenAI
    # SDK has no localhost convention to fall back on, unlike the Ollama SDK
    # which honours `OLLAMA_HOST`.
    base_url: str
    # Env var holding the bearer token. When unset the SDK receives "EMPTY",
    # which is fine for the default llama-server config that doesn't validate auth.
    api_key_env: str | None = None


class LocalModelConfig(BaseModel):
    kind: Literal['local']
    params: OllamaLocalModelParams | LlamaCppLocalModelParams = Field(
        discriminator="driver",
    )


class CloudProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class CloudModelConfig(BaseModel):
    kind: Literal['cloud']
    provider: CloudProvider
    model_id: str
    # OpenAI-compatible endpoint override. Lets a `provider: openai` entry hit
    # a self-hosted server (vLLM, llama.cpp) or a router (LiteLLM) instead of
    # api.openai.com. Anthropic/Google SDKs have their own endpoint
    # conventions and aren't covered by this knob.
    base_url: str | None = None
    # Name of the env var holding the key for `base_url`. When unset, the key
    # defaults to "EMPTY" so servers that don't validate auth still get a
    # non-empty string and the SDK doesn't raise on missing OPENAI_API_KEY.
    api_key_env: str | None = None

    @model_validator(mode="after")
    def _overrides_only_for_openai(self) -> Self:
        if self.provider is not CloudProvider.OPENAI and (
            self.base_url is not None or self.api_key_env is not None
        ):
            raise ValueError(
                "base_url / api_key_env are only supported on provider=openai entries"
            )
        return self


class EmbeddingModelConfig(BaseModel):
    """Represents a `sentence_transformer` transformer id."""
    name: str
    org: str


class InferenceConcurrencyConfig(BaseModel):
    ollama: int = 1
    llamacpp: int = 1
    openai: int = 8
    anthropic: int = 4
    google: int = 4


class InferenceConstants(BaseModel):
    """Fixed values applied to every cell of the matrix for backends that consume them."""
    model_config = ConfigDict(extra="forbid")
    seed: int
    top_k: int
    repetition_penalty: float


class InferenceDefaults(BaseModel):
    """Sweep axes shared across backends. Each backend pulls only the axes its API supports."""
    model_config = ConfigDict(extra="forbid")
    temperature: list[float]
    top_p: list[float]


class TruncationProfile(BaseModel):
    """A tagged tuple binding `top_p` and `min_p` together as a single sampling-strategy cell."""
    model_config = ConfigDict(extra="forbid")
    name: str
    top_p: float
    min_p: float | None


class OllamaInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    truncation: list[TruncationProfile]


class LlamaCppInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    truncation: list[TruncationProfile]


class OpenAIInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning_effort: list[Literal["low", "medium", "high", "xhigh"]]
    text_verbosity: list[Literal["low", "medium"]]


class AnthropicInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning_budget: list[int]


class GoogleInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning_budget: list[int]


class InferencePerBackend(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ollama: OllamaInferenceConfig
    llamacpp: LlamaCppInferenceConfig
    openai: OpenAIInferenceConfig
    anthropic: AnthropicInferenceConfig
    google: GoogleInferenceConfig


class InferenceParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concurrency: InferenceConcurrencyConfig = Field(
        default_factory=InferenceConcurrencyConfig)
    constants: InferenceConstants
    defaults: InferenceDefaults
    per_backend: InferencePerBackend


class OverrideableConstants(BaseModel):
    """Subset of `InferenceConstants` that may be overridden per-model.

    `seed` is deliberately absent: it is the cross-model reproducibility
    baseline and `extra="forbid"` rejects any attempt to set it here at
    config-load time, keeping every cell of every model anchored to the same
    global `inference.constants.seed`.
    """
    model_config = ConfigDict(extra="forbid")
    top_k: int | None = None
    repetition_penalty: float | None = None


class ModelInferenceOverride(BaseModel):
    """Per-model replacement for the global inference matrix axes.

    Each section is optional; missing sections inherit from the corresponding
    global `inference` block. `per_backend` is stored as a raw dict and
    re-validated against the backend's per-backend class by the `ModelConfig`
    validator — which is where we have access to `meta` to pick the right
    schema.
    """
    model_config = ConfigDict(extra="forbid")
    defaults: InferenceDefaults | None = None
    per_backend: dict[str, Any] | None = None
    constants: OverrideableConstants | None = None


def _per_backend_class_for(
    meta: LocalModelConfig | CloudModelConfig,
) -> tuple[type[BaseModel], frozenset[str]]:
    """Map a model's `meta` to (per_backend_class, allowed_constants).

    `allowed_constants` is the model backend's `CONSUMES_CONSTANTS` minus
    `seed` — i.e. the constants a per-model override is permitted to touch.
    """
    if isinstance(meta, LocalModelConfig):
        match meta.params.driver:
            case "ollama":
                return OllamaInferenceConfig, frozenset({"top_k", "repetition_penalty"})
            case "llamacpp":
                return LlamaCppInferenceConfig, frozenset({"top_k", "repetition_penalty"})
    if isinstance(meta, CloudModelConfig):
        match meta.provider:
            case CloudProvider.OPENAI:
                return OpenAIInferenceConfig, frozenset()
            case CloudProvider.ANTHROPIC:
                return AnthropicInferenceConfig, frozenset({"top_k"})
            case CloudProvider.GOOGLE:
                return GoogleInferenceConfig, frozenset({"top_k"})
    raise ValueError(f"Cannot map meta to backend class: {meta!r}")


class ModelConfig(BaseModel):
    name: str
    meta: LocalModelConfig | CloudModelConfig = Field(discriminator="kind")
    inference_override: ModelInferenceOverride | None = None

    @model_validator(mode="after")
    def _validate_inference_override(self) -> Self:
        if self.inference_override is None:
            return self
        backend_cls, allowed_constants = _per_backend_class_for(self.meta)

        # Re-parse `per_backend` against the model's backend-specific schema
        # so typos and cross-backend axes (e.g. `reasoning_effort` under an
        # Ollama override) raise at load with a precise Pydantic error
        # instead of silently projecting away at matrix-expansion time.
        # Store the dump back so downstream consumers see coerced values.
        if self.inference_override.per_backend is not None:
            typed = backend_cls.model_validate(self.inference_override.per_backend)
            self.inference_override.per_backend = typed.model_dump()

        if self.inference_override.constants is not None:
            c = self.inference_override.constants
            for field_name in ("top_k", "repetition_penalty"):
                if (
                    getattr(c, field_name) is not None
                    and field_name not in allowed_constants
                ):
                    raise ValueError(
                        f"Constant {field_name!r} is not consumed by the "
                        f"backend for model {self.name!r}; remove it from "
                        f"`inference_override.constants`."
                    )
        return self


class Config(BaseModel):
    scenarios: pathlib.Path
    legenda_schema: FilePath
    dsl: list[DSLConfig]
    models: list[ModelConfig]
    encoding: EmbeddingModelConfig
    inference: InferenceParamsConfig
    max_syntax_retries: int
    output: pathlib.Path | None = None


def load_config(path: pathlib.Path = pathlib.Path("config.yaml")) -> Config:
    """Read a YAML file and return the parsed dictionary."""
    logging.info(f"Loading config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        return Config.model_validate(raw)
