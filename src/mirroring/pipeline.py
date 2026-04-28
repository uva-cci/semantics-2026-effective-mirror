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
import jsonschema.exceptions
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.config import Config, DSLConfig
from src.utils.embeddings import get_encoder, score_vectors
from src.utils.models import InferenceModel, InferenceOutput, InferenceParams, get_model

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts"


class Scenario(BaseModel):
    id: str
    source: Any
    description: str


class FewShotExample(BaseModel):
    input: str
    output: str


class AblationFlags(BaseModel):
    syntax: bool
    few_shot: bool


class DSLSetup(BaseModel):
    name: str


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

        # DSL name -> raw JSON-schema text
        self.dsl_schemas: dict[str, str] = {}
        for dsl in cfg.dsl:
            with open(dsl.schema_path, "r") as f:
                self.dsl_schemas[dsl.name] = f.read()

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
                                    scenario, dsl, model, abl, params)
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
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
    ) -> MirroringPipelineOutput:

        symbolic_output1 = self.generate_symbolic(
            scenario.description, dsl, model, ablation, params)

        decode_prompt = self.tmpl_decode.render({
            "dsl_input": symbolic_output1.text,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "schema": self.dsl_schemas[dsl.name],
            "examples": [
                # reverse input/output for decoding
                FewShotExample(input=ex.output, output=ex.input)
                for ex in self.examples[dsl.name]
            ]
        })

        logging.debug(decode_prompt)
        natural_language = model.generate(decode_prompt, params)
        logging.debug(f"output: {natural_language}")

        symbolic_output2 = self.generate_symbolic(
            natural_language.text, dsl, model, ablation, params)

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
            dsl=DSLSetup(name=dsl.name),
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
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
    ) -> InferenceOutput:
        assert self.cfg.max_syntax_retries > 0

        schema_text = self.dsl_schemas[dsl.name]

        encode_prompt = self.tmpl_encode.render({
            "scenario": scenario,
            "ablation": ablation.model_dump(),
            "dsl": dsl,
            "schema": schema_text,
            "examples": self.examples[dsl.name],
            "attempt": None
        })

        logging.debug(encode_prompt)

        ok = False
        err = ""
        output = model.generate(encode_prompt, params)
        for i in range(self.cfg.max_syntax_retries):
            ok, err = self.validate_json(output.text, dsl.name)
            if ok:
                break

            encode_prompt = self.tmpl_encode.render({
                "scenario": scenario,
                "ablation": ablation.model_dump(),
                "dsl": dsl,
                "schema": schema_text,
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
        """Validate `s` against the DSL's JSON schema.

        Returns `(ok, err)`. The error message is engineered for the LLM
        self-refinement loop: it (1) names the failure mode (parse vs schema),
        (2) quotes the offending region of the model's own output, (3) surfaces
        the failing keyword, JSON path, and required value, and (4) ends with
        a concrete "Fix:" line. Without this shape small models tend to either
        re-emit the same mistake or "fix" the wrong layer (e.g. retype a
        well-formed value when the actual problem was a missing required key).
        """
        schema = json.loads(self.dsl_schemas[dsl])
        try:
            instance = json.loads(s)
        except json.JSONDecodeError as e:
            # Quote a small window around the offending byte so the model can
            # see *its own* characters that broke the parse, not just an offset.
            start = max(0, e.pos - 40)
            end = min(len(s), e.pos + 40)
            snippet = s[start:end].replace("\n", "\\n")
            err = (
                f"The output is not valid JSON.\n"
                f"Parser error: {e.msg} at line {e.lineno}, column {e.colno}.\n"
                f"Context (around the failure): ...{snippet}...\n"
                f"Fix: produce a single well-formed JSON document with no "
                f"surrounding prose, markdown, or code fences."
            )
            logging.debug(err)
            return (False, err)

        try:
            jsonschema.validate(instance=instance, schema=schema)
            return (True, "")
        except jsonschema.ValidationError as e:
            path = "/" + \
                "/".join(str(p) for p in e.absolute_path) if e.absolute_path else "<root>"
            err = (
                f"The output is valid JSON but does not conform to the schema.\n"
                f"Failing keyword: {e.validator}\n"
                f"Schema requirement: {json.dumps(e.validator_value)[:200]}\n"
                f"JSON path of the failure: {path}\n"
                f"Offending value: {json.dumps(e.instance)[:200]}\n"
                f"Reason: {e.message}\n"
                f"Fix: change the value at {path} so that the "
                f"`{e.validator}` constraint above is satisfied."
            )
            # For oneOf/anyOf failures, the top-level message is generic
            # ("is not valid under any of the given schemas"); the actionable
            # detail lives in `e.context`. `best_match` picks the most specific
            # branch error so the model sees the precise field that failed.
            if e.context:
                best = jsonschema.exceptions.best_match(e.context)
                best_path = "/" + "/".join(str(p) for p in best.absolute_path) \
                    if best.absolute_path else "<root>"
                err += (
                    f"\nMost specific sub-error: {best.message} at {best_path}"
                )
            logging.debug(err)
            return (False, err)
