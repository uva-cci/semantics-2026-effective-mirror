import logging
from pathlib import Path

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from sentence_transformers import SentenceTransformer

from src.config import Config, DSLConfig, DSLValidationConfig, PipelineName
from src.embeddings import get_encoder, score_vectors
from src.models import InferenceModel, InferenceOutput, InferenceParams
from src.pipeline import (
    AblationFlags,
    FewShotExample,
    Pipeline,
    PipelineOutput,
    Scenario,
)

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts"


class MirroringPipelineOutput(PipelineOutput):
    symbolic_output1: InferenceOutput
    symbolic_output2: InferenceOutput
    natural_language: InferenceOutput
    # encoder -> score
    semantic_scores: dict[str, float]
    symbolic_equivalence: bool


class MirroringPipeline(Pipeline[MirroringPipelineOutput]):

    def __init__(self, cfg: Config, scenarios: list[Scenario]) -> None:
        super().__init__(cfg, scenarios)

        self.tmpl_env = Environment(
            loader=FileSystemLoader(PROMPTS_PATH),
            undefined=StrictUndefined,
        )

        self.tmpl_encode = self.tmpl_env.get_template("encode.jinja")
        self.tmpl_decode = self.tmpl_env.get_template("decode.jinja")

        self.dsl_definitions: dict[str, dict[str, str]] = {}
        for dsl in cfg.dsl:
            for validation in dsl.validation:
                with open(validation.path, "r") as f:
                    self.dsl_definitions.setdefault(
                        dsl.name, {})[validation.kind] = f.read()

        self.encoders: dict[str, SentenceTransformer] = {}
        for encoding in self.cfg.encodings:
            self.encoders[encoding.name] = get_encoder(encoding)

    def produce_datapoint(
        self,
        scenario: Scenario,
        dsl: DSLConfig,
        validation: DSLValidationConfig,
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
    ) -> MirroringPipelineOutput:
        dsl_definition = self.dsl_definitions[dsl.name][validation.kind]

        encode_prompt = self.tmpl_encode.render({
            "scenario": scenario.description,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "validation": {"definition": dsl_definition, **validation.model_dump()},
            "examples": self.examples[dsl.name]
        })

        logging.debug(encode_prompt)

        symbolic_output1 = model.generate(encode_prompt, params)

        logging.debug(f"output: {symbolic_output1}")

        decode_prompt = self.tmpl_decode.render({
            "dsl_input": symbolic_output1.text,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "validation": {"definition": dsl_definition, **validation.model_dump()},
            "examples": [
                # reverse input/output for decoding
                FewShotExample(
                    validation_kind=validation.kind,
                    input=ex.output,
                    output=ex.input)
                for ex in self.examples[dsl.name]
            ]
        })

        logging.debug(decode_prompt)

        natural_language = model.generate(decode_prompt, params)

        logging.debug(f"output: {natural_language}")

        encode_prompt = self.tmpl_encode.render({
            "scenario": natural_language.text,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "validation": {"definition": dsl_definition, **validation.model_dump()},
            "examples": self.examples[dsl.name]
        })

        logging.debug(encode_prompt)

        symbolic_output2 = model.generate(encode_prompt, params)

        logging.debug(f"output: {symbolic_output2}")

        semantic_scores: dict[str, float] = {}
        for encoding, encoder in self.encoders.items():
            a: torch.Tensor = encoder.encode(
                scenario.description, convert_to_tensor=True, show_progress_bar=False)
            b: torch.Tensor = encoder.encode(
                natural_language.text, convert_to_tensor=True, show_progress_bar=False)
            semantic_scores[encoding] = score_vectors(a, b)

        return MirroringPipelineOutput(
            pipeline=PipelineName.MIRRORING,
            scenario_id=scenario.id,
            model=model.name,
            dsl=dsl.name,
            ablation=ablation,
            symbolic_output1=symbolic_output1,
            symbolic_output2=symbolic_output2,
            natural_language=natural_language,
            semantic_scores=semantic_scores,
            symbolic_equivalence=False  # TODO: run symbolic static analysis
        )
