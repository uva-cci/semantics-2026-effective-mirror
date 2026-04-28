import itertools
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

import jsonschema
import lark
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.config import (
    Config,
    DSLConfig,
    DSLValidationConfig,
    ValidationFormat,
)
from src.utils.embeddings import get_encoder, score_vectors
from src.utils.models import InferenceModel, InferenceOutput, InferenceParams, get_model

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts"


class Scenario(BaseModel):
    id: str
    source: Any
    description: str


class FewShotExample(BaseModel):
    validation_kind: ValidationFormat
    input: str
    output: str


class AblationFlags(BaseModel):
    syntax: bool
    few_shot: bool


class DSLSetup(BaseModel):
    name: str
    validation: ValidationFormat


class MirroringPipelineOutput(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    scenario_id: str
    ablation: AblationFlags
    dsl: DSLSetup
    model: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc))
    symbolic_output1: InferenceOutput
    symbolic_output2: InferenceOutput
    natural_language: InferenceOutput
    # encoder -> score
    semantic_scores: dict[str, float]
    symbolic_equivalence: bool


class MirroringPipeline:

    def __init__(self, cfg: Config, scenarios: list[Scenario]) -> None:
        self.cfg = cfg
        self.scenarios = scenarios
        self.output: Queue[MirroringPipelineOutput | None] = Queue()

        # DSL name -> examples
        self.examples: dict[str, list[FewShotExample]] = {}
        for dsl in cfg.dsl:
            self.examples[dsl.name] = []
            with open(dsl.examples, "r") as f:
                raw = json.load(f)
                for ex in raw:
                    self.examples[dsl.name].append(FewShotExample(**ex))

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

    def run(self) -> None:
        logging.info(f"Pipeline {self.__class__.__name__} started")

        if self.cfg.output is not None:
            output_path = self.cfg.output
        else:
            stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = Path(f"output-{stamp}.ndjson")

        t = threading.Thread(
            target=self.writer_worker,
            args=(output_path,),
            daemon=True
        )
        t.start()

        # create permutations of inference parameters based on the config
        inference_profiles = list(itertools.product(
            self.cfg.inference.temperature,
            self.cfg.inference.top_p,
            self.cfg.inference.top_k,
        ))

        for model_cfg in self.cfg.models:
            logging.info(f"- Model: {model_cfg.name}")
            model = get_model(model_cfg)
            for scenario in self.scenarios:
                logging.info(f"Running scenario: {scenario.id}")
                for dsl in self.cfg.dsl:
                    logging.info(f"- DSL: {dsl.name}")
                    for validation in dsl.validation:
                        logging.info(f"- Validation: {validation.kind}")
                        for abl in [
                            AblationFlags(syntax=False, few_shot=False),
                            AblationFlags(syntax=True, few_shot=False),
                            AblationFlags(syntax=False, few_shot=True),
                            AblationFlags(syntax=True, few_shot=True),
                        ]:
                            logging.info(f"- Syntax: {abl.syntax}")
                            logging.info(f"- Few-shot: {abl.few_shot}")

                            for temp, top_p, top_k in inference_profiles:
                                params = InferenceParams(
                                    temperature=temp,
                                    top_p=top_p,
                                    top_k=top_k
                                )
                                logging.info(f"- Params: {params}")

                                self.output.put(
                                    self.produce_datapoint(
                                        scenario, dsl, validation, model, abl, params)
                                )

                            logging.info("✓ Done")

        self.output.put(None)   # tell the writer to finish
        t.join()

        logging.info(f"Pipeline {self.__class__.__name__} completed")

    def writer_worker(self, fp: Path):
        logging.info(f"Writer worker for {self.__class__.__name__} started")

        with fp.open("a", encoding="utf-8", buffering=1) as f:
            while True:
                dp = self.output.get()
                if dp is None:
                    break
                f.write(dp.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

        logging.info(f"Writer worker for {self.__class__.__name__} closed")

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
