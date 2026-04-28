import logging
import pathlib
from enum import StrEnum
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, FilePath


class DSLConfig(BaseModel):
    name: str
    examples: FilePath
    schema_path: FilePath = Field(alias="schema")


class OllamaLocalModelParams(BaseModel):
    driver: Literal['ollama']
    model_id: str


class LocalModelConfig(BaseModel):
    kind: Literal['local']
    params: OllamaLocalModelParams


class CloudProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class CloudModelConfig(BaseModel):
    kind: Literal['cloud']
    provider: CloudProvider
    model_id: str


class ModelConfig(BaseModel):
    name: str
    meta: LocalModelConfig | CloudModelConfig = Field(discriminator="kind")


class EmbeddingModelConfig(BaseModel):
    """Represents a `sentence_transformer` transformer id."""
    name: str
    org: str


class InferenceConcurrencyConfig(BaseModel):
    ollama: int = 1
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


class Config(BaseModel):
    scenarios: pathlib.Path
    legenda_schema: FilePath
    dsl: list[DSLConfig]
    models: list[ModelConfig]
    encodings: list[EmbeddingModelConfig]
    inference: InferenceParamsConfig
    max_syntax_retries: int
    output: pathlib.Path | None = None


def load_config(path: pathlib.Path = pathlib.Path("config.yaml")) -> Config:
    """Read a YAML file and return the parsed dictionary."""
    logging.info(f"Loading config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        return Config.model_validate(raw)
