import pathlib
from enum import StrEnum
from typing import Literal

import yaml
from pydantic import BaseModel, Field, FilePath, HttpUrl


class PipelineName(StrEnum):
    MIRRORING = "mirroring"
    REASONING = "reasoning"


class ValidationFormat(StrEnum):
    JSON_SCHEMA = "json-schema"
    JSON_LD = "json-ld"
    BNF = "bnf"


class DSLValidationConfig(BaseModel):
    kind: ValidationFormat
    path: FilePath


class DSLConfig(BaseModel):
    name: str
    validation: list[DSLValidationConfig]


class LocalModelConfig(BaseModel):
    kind: Literal['local']
    context_length: int
    url: HttpUrl


class CloudProvider(StrEnum):
    OPENAI = "openai"


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


class InferenceParamsConfig(BaseModel):
    temperature: list[float]
    top_p: list[float]
    top_k: list[int]


class Config(BaseModel):
    pipelines: list[PipelineName]
    scenarios: pathlib.Path
    dsl: list[DSLConfig]
    models: list[ModelConfig]
    encodings: list[EmbeddingModelConfig]
    inference: InferenceParamsConfig


def load_config(path: pathlib.Path = pathlib.Path("config.yaml")) -> Config:
    """Read a YAML file and return the parsed dictionary."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        return Config.model_validate(raw)
