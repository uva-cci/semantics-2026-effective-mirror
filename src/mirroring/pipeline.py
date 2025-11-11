import json
import logging
from pathlib import Path

import jsonschema
import lark
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from sentence_transformers import SentenceTransformer

from src.config import Config, DSLConfig, DSLValidationConfig, PipelineName
from src.embeddings import get_encoder, score_vectors
from src.models import InferenceModel, InferenceOutput, InferenceParams
from src.pipeline import (
    AblationFlags,
    DSLSetup,
    FewShotExample,
    Pipeline,
    PipelineOutput,
    Scenario,
    ValidationFormat,
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

        self.dsl_definitions: dict[str, dict[ValidationFormat, str]] = {}
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

        symbolic_output1 = self.generate_symbolic(
            scenario.description, dsl, validation, model, ablation, params)

        decode_prompt = self.tmpl_decode.render({
            "dsl_input": symbolic_output1.text,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "validation": {
                "definition": self.dsl_definitions[dsl.name][validation.kind],
                **validation.model_dump()
            },
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

        symbolic_output2 = self.generate_symbolic(
            natural_language.text, dsl, validation, model, ablation, params)

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
            dsl=DSLSetup(name=dsl.name, validation=validation.kind),
            ablation=ablation,
            symbolic_output1=symbolic_output1,
            symbolic_output2=symbolic_output2,
            natural_language=natural_language,
            semantic_scores=semantic_scores,
            symbolic_equivalence=False  # TODO: run symbolic static analysis
        )

    def generate_symbolic(
        self,
        scenario: str,
        dsl: DSLConfig,
        validation: DSLValidationConfig,
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
    ) -> InferenceOutput:
        assert self.cfg.max_syntax_retries > 0

        dsl_definition = self.dsl_definitions[dsl.name][validation.kind]

        encode_prompt = self.tmpl_encode.render({
            "scenario": scenario,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "validation": {"definition": dsl_definition, **validation.model_dump()},
            "examples": self.examples[dsl.name],
            "attempt": None
        })

        logging.debug(encode_prompt)

        ok = False
        err = ""
        output = model.generate(encode_prompt, params)
        for i in range(self.cfg.max_syntax_retries):
            match validation.kind:
                case ValidationFormat.JSON_SCHEMA:
                    ok, err = self.validate_json(output.text, dsl.name)
                case ValidationFormat.BNF:
                    ok, err = self.validate_bnf(output.text, dsl.name)

            if ok:
                break

            encode_prompt = self.tmpl_encode.render({
                "scenario": scenario,
                "ablation": ablation.model_dump(),
                "dsl": dsl,
                "validation": {"definition": dsl_definition, **validation.model_dump()},
                "examples": self.examples[dsl.name],
                "attempt": {
                    "previous": output.text,
                    "error": err
                }
            })

            logging.debug(encode_prompt)

            output = model.generate(encode_prompt, params)
            output.attempts = i + 1

            logging.debug(f"output: {output}")

        output.success = ok
        return output

    def validate_json(self, s: str, dsl: str) -> tuple[bool, str]:
        schema = json.loads(
            self.dsl_definitions[dsl][ValidationFormat.JSON_SCHEMA])
        try:
            raw = json.loads(s)
            jsonschema.validate(instance=raw, schema=schema)
            return (True, "")
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logging.debug(e)
            return (False, str(e))

    def validate_bnf(self, s: str, dsl: str) -> tuple[bool, str]:
        parser = lark.Lark(self.dsl_definitions[dsl][ValidationFormat.BNF])
        try:
            parser.parse(s)  # type: ignore
            return (True, "")
        except lark.UnexpectedInput as e:
            return (False, str(e))
