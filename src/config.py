import logging
import pathlib
from enum import StrEnum
from typing import Literal

import yaml
from pydantic import BaseModel, Field, FilePath


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


class InferenceParamsConfig(BaseModel):
    temperature: list[float]
    top_p: list[float]
    top_k: list[int]
    concurrency: InferenceConcurrencyConfig = Field(
        default_factory=InferenceConcurrencyConfig)


class Config(BaseModel):
    scenarios: pathlib.Path
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
