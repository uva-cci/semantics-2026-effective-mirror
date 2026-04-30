import asyncio
import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Callable

import jsonschema
import jsonschema.exceptions
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.config import Config, DSLConfig
from src.utils.embeddings import get_encoder, score_vectors
from src.utils.models import (
    InferenceModel,
    InferenceOutput,
    InferenceParams,
    expand_for_backend,
    get_model,
)

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts"


def compute_cell_key(
    model_name: str,
    scenario_id: str,
    dsl_name: str,
    ablation: "AblationFlags",
    params: InferenceParams,
) -> str:
    payload = {
        "model": model_name,
        "scenario_id": scenario_id,
        "dsl": dsl_name,
        "ablation": ablation.model_dump(),
        "params": params.model_dump(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


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
    # Deterministic fingerprint of the dispatch tuple (model, scenario_id, dsl,
    # ablation, requested params). Used by the startup resume scan to skip
    # cells that already landed on disk; computed before dispatch so it sees
    # the requested params, not any post-call rewrites (e.g. Anthropic's
    # forced temperature=1.0 under extended thinking).
    cell_key: str
    scenario_id: str
    ablation: AblationFlags
    dsl: DSLSetup
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    symbolic_output1: InferenceOutput
    symbolic_output2: InferenceOutput | None = None
    natural_language: InferenceOutput | None = None
    legenda: InferenceOutput
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
        self.tmpl_legenda = self.tmpl_env.get_template("legenda.jinja")
        self.tmpl_refine = self.tmpl_env.get_template("refine.jinja")
        self.tmpl_error_decode = self.tmpl_env.get_template("error_decode.jinja")
        self.tmpl_error_validation = self.tmpl_env.get_template("error_validation.jinja")

        # DSL name -> raw JSON-schema text
        self.dsl_schemas: dict[str, str] = {}
        for dsl in cfg.dsl:
            with open(dsl.schema_path, "r") as f:
                self.dsl_schemas[dsl.name] = f.read()

        with open(cfg.legenda_schema, "r") as f:
            self.legenda_schema_text: str = f.read()

        # (scenario.id, model.name) -> cached legenda. Shared across all DSLs,
        # ablations, and inference profiles for the pair: legenda content is
        # DSL-agnostic, so regenerating per cell would only inject sampling
        # noise without signal.
        self.legendas: dict[tuple[str, str], InferenceOutput] = {}
        # Per-key locks created lazily as keys appear at runtime; the guard
        # serialises only the lock-creation step, not the LLM call itself, so
        # different keys never block each other.
        self.legenda_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.legenda_locks_guard: asyncio.Lock = asyncio.Lock()

        self.encoders: dict[str, SentenceTransformer] = {}
        self.encoder_locks: dict[str, asyncio.Lock] = {}
        for encoding in self.cfg.encodings:
            self.encoders[encoding.name] = get_encoder(encoding)
            # Per-encoder lock prevents concurrent .encode() calls on the same
            # SentenceTransformer instance — not safe under GPU contention.
            self.encoder_locks[encoding.name] = asyncio.Lock()

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        logging.info(f"Pipeline {self.__class__.__name__} started")

        if self.cfg.output is not None:
            output_path = self.cfg.output
        else:
            stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = Path(f"output-{stamp}.ndjson")

        # Resume: any cell whose key already appears on disk is skipped.
        # Failed rows (symbolic_output1.success=False) count as done — the
        # failure is a reproducible property of the dispatch tuple.
        completed = self._load_completed_cell_keys(output_path)

        t = threading.Thread(
            target=self.writer_worker, args=(output_path,), daemon=True
        )
        t.start()

        ablations = [
            AblationFlags(syntax=False, few_shot=False),
            AblationFlags(syntax=True, few_shot=False),
            AblationFlags(syntax=False, few_shot=True),
            AblationFlags(syntax=True, few_shot=True),
        ]

        for model_cfg in self.cfg.models:
            logging.info(f"- Model: {model_cfg.name}")
            model = get_model(model_cfg, self.cfg.inference.concurrency)

            # Per-model matrix: each backend only sweeps the axes its API
            # consumes (declared on the InferenceModel subclass).
            profiles = expand_for_backend(self.cfg.inference, model)

            cells = [
                (
                    scenario,
                    dsl,
                    abl,
                    params,
                    compute_cell_key(model.name, scenario.id, dsl.name, abl, params),
                )
                for scenario in self.scenarios
                for dsl in self.cfg.dsl
                for abl in ablations
                for params in profiles
            ]

            skipped = sum(1 for *_, key in cells if key in completed)
            tasks = [
                self._run_task(scenario, dsl, model, abl, params, cell_key)
                for scenario, dsl, abl, params, cell_key in cells
                if cell_key not in completed
            ]

            logging.info(
                f"Dispatching {len(tasks)} datapoint(s) for {model.name} "
                f"(skipped {skipped} already complete)"
            )
            await asyncio.gather(*tasks)
            logging.info(f"✓ Done with {model.name}")

        self.output.put(None)  # tell the writer to finish
        t.join()

        logging.info(f"Pipeline {self.__class__.__name__} completed")

    async def _run_task(
        self,
        scenario: "Scenario",
        dsl: DSLConfig,
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
        cell_key: str,
    ) -> None:
        dp = await self.produce_datapoint(
            scenario, dsl, model, ablation, params, cell_key
        )
        # Queue.put is thread-safe and non-blocking on an unbounded queue, so it
        # is safe to call directly from the event loop. The writer thread is the
        # sole consumer, which serialises file writes.
        self.output.put(dp)
        logging.info(
            f"  ✓ {model.name} | scenario={scenario.id} dsl={dsl.name} "
            f"syntax={ablation.syntax} few_shot={ablation.few_shot}"
        )

    def _load_completed_cell_keys(self, fp: Path) -> set[str]:
        """Scan an existing output file and collect cell_keys for resume.

        A torn final line (a partial fsync from a prior crash) is tolerated
        with a warning. Any earlier malformed line is real corruption and
        raises. Rows without a `cell_key` field are legacy (predate this
        column) and are counted but ignored — those runs simply re-dispatch.
        """
        if not fp.exists():
            return set()

        with fp.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        completed: set[str] = set()
        legacy = 0
        last_idx = len(lines) - 1
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == last_idx:
                    logging.warning(
                        f"Ignoring torn final line in {fp} (likely from a "
                        f"prior crash mid-fsync)"
                    )
                    continue
                raise
            key = row.get("cell_key")
            if key is None:
                legacy += 1
                continue
            completed.add(key)

        if legacy:
            logging.info(
                f"{legacy} legacy row(s) in {fp} ignored for resume (no cell_key)"
            )
        if completed:
            logging.info(
                f"Resume: found {len(completed)} completed cell(s) in {fp}"
            )
        return completed

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

    async def produce_datapoint(
        self,
        scenario: Scenario,
        dsl: DSLConfig,
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
        cell_key: str,
    ) -> MirroringPipelineOutput:

        legenda = await self.generate_legenda(scenario, model, params)

        symbolic_output1 = await self.generate_symbolic(
            scenario.description, dsl, model, ablation, params, legenda.text
        )

        natural_language: InferenceOutput | None = None
        symbolic_output2: InferenceOutput | None = None
        semantic_scores: dict[str, float] = {}

        if symbolic_output1.success:
            decode_prompt = self.tmpl_decode.render(
                {
                    "dsl_input": symbolic_output1.text,
                    "ablation": ablation.model_dump(),
                    "dsl": dsl,
                    "schema": self.dsl_schemas[dsl.name],
                    "legenda": legenda.text,
                    "examples": [
                        # reverse input/output for decoding
                        FewShotExample(input=ex.output, output=ex.input)
                        for ex in self.examples[dsl.name]
                    ],
                }
            )

            logging.debug(decode_prompt)
            natural_language = await model.generate(decode_prompt, params)
            logging.debug(f"output: {natural_language}")

            symbolic_output2 = await self.generate_symbolic(
                natural_language.text, dsl, model, ablation, params, legenda.text
            )

            for encoding, encoder in self.encoders.items():
                async with self.encoder_locks[encoding]:
                    a, b = await asyncio.to_thread(
                        self._encode_pair,
                        encoder,
                        scenario.description,
                        natural_language.text,
                    )
                semantic_scores[encoding] = score_vectors(a, b)
        else:
            logging.info(
                f"  ⤼ skipping decode/re-encode for {model.name} | "
                f"scenario={scenario.id} dsl={dsl.name} "
                f"syntax={ablation.syntax} few_shot={ablation.few_shot}: "
                f"symbolic_output1 failed validation after "
                f"{symbolic_output1.attempts} attempt(s)"
            )

        return MirroringPipelineOutput(
            cell_key=cell_key,
            scenario_id=scenario.id,
            model=model.name,
            dsl=DSLSetup(name=dsl.name),
            ablation=ablation,
            symbolic_output1=symbolic_output1,
            symbolic_output2=symbolic_output2,
            natural_language=natural_language,
            legenda=legenda,
            semantic_scores=semantic_scores,
            symbolic_equivalence=False,  # TODO: run symbolic static analysis
        )

    @staticmethod
    def _encode_pair(
        encoder: SentenceTransformer,
        a_text: str,
        b_text: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = encoder.encode(a_text, convert_to_tensor=True, show_progress_bar=False)
        b = encoder.encode(b_text, convert_to_tensor=True, show_progress_bar=False)
        return a, b

    async def _get_legenda_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        async with self.legenda_locks_guard:
            if key not in self.legenda_locks:
                self.legenda_locks[key] = asyncio.Lock()
            return self.legenda_locks[key]

    async def generate_legenda(
        self,
        scenario: Scenario,
        model: InferenceModel,
        params: InferenceParams,
    ) -> InferenceOutput:
        """Extract a DSL-agnostic vocabulary glossary for the scenario.

        Always-on consistency baseline: every datapoint inherits the same
        legenda for its (scenario, model) pair, so symbolic outputs share a
        canonical naming for entities, actions, and durations rather than
        re-inventing labels per cell. Cache key is `(scenario.id, model.name)`;
        on miss, generates and validates against the legenda schema using the
        same retry machinery as `generate_symbolic`.
        """
        key = (scenario.id, model.name)
        lock = await self._get_legenda_lock(key)
        async with lock:
            if key in self.legendas:
                return self.legendas[key]

            def render() -> str:
                return self.tmpl_legenda.render(
                    {
                        "scenario": scenario.description,
                    }
                )

            output = await self._generate_validated(
                render,
                self.legenda_schema_text,
                model,
                params,
            )
            self.legendas[key] = output
            return output

    async def generate_symbolic(
        self,
        scenario: str,
        dsl: DSLConfig,
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
        legenda: str | None = None,
    ) -> InferenceOutput:
        schema_text = self.dsl_schemas[dsl.name]
        examples = self.examples[dsl.name]

        def render() -> str:
            return self.tmpl_encode.render(
                {
                    "scenario": scenario,
                    "ablation": ablation.model_dump(),
                    "dsl": dsl,
                    "schema": schema_text,
                    "examples": examples,
                    "legenda": legenda,
                }
            )

        return await self._generate_validated(render, schema_text, model, params)

    async def _generate_validated(
        self,
        render_base: Callable[[], str],
        schema_text: str,
        model: InferenceModel,
        params: InferenceParams,
    ) -> InferenceOutput:
        """Render → generate → validate-against-schema → retry on failure.

        `render_base()` is the task prompt; on retries the harness appends
        `refine.jinja` (the self-refinement addendum carrying the prior output
        and validator feedback) so callers stay refinement-agnostic. The same
        retry budget (`cfg.max_syntax_retries`) and rich validator-error format
        applied to the original encode loop is shared by every consumer.
        """
        assert self.cfg.max_syntax_retries > 0

        base = render_base()
        prompt = base
        logging.debug(prompt)

        ok = False
        err = ""
        errors: list[str] = []
        output = await model.generate(prompt, params)
        for i in range(self.cfg.max_syntax_retries):
            ok, err = self.validate_json(output.text, schema_text)
            if ok:
                break
            errors.append(err)

            refine = self.tmpl_refine.render(
                {"previous": output.text, "error": err}
            )
            prompt = base + "\n\n" + refine
            logging.debug(prompt)

            output = await model.generate(prompt, params)
            output.attempts = i + 1

            logging.debug(f"output: {output}")

        # Validate the final retry too, otherwise the trail stops one error
        # short of the text that's actually being emitted on the row.
        if not ok:
            ok, err = self.validate_json(output.text, schema_text)
            if not ok:
                errors.append(err)

        output.success = ok
        output.errors = errors
        return output

    def validate_json(self, s: str, schema_text: str) -> tuple[bool, str]:
        """Validate `s` against `schema_text` (a raw JSON-schema document).

        Returns `(ok, err)`. The error message is engineered for the LLM
        self-refinement loop: it (1) names the failure mode (parse vs schema),
        (2) quotes the offending region of the model's own output, (3) surfaces
        the failing keyword, JSON path, and required value, and (4) ends with
        a concrete "Fix:" line. Without this shape small models tend to either
        re-emit the same mistake or "fix" the wrong layer (e.g. retype a
        well-formed value when the actual problem was a missing required key).

        This function only does the *structural* extraction — path joining,
        branch deduping, JSON truncation. The user-facing prose lives in
        `error_decode.jinja` (JSONDecodeError) and `error_validation.jinja`
        (jsonschema.ValidationError).
        """
        schema = json.loads(schema_text)
        try:
            instance = json.loads(s)
        except json.JSONDecodeError as e:
            # Quote a small window around the offending byte so the model can
            # see *its own* characters that broke the parse, not just an offset.
            start = max(0, e.pos - 40)
            end = min(len(s), e.pos + 40)
            snippet = s[start:end].replace("\n", "\\n")
            err = self.tmpl_error_decode.render(
                {
                    "msg": e.msg,
                    "lineno": e.lineno,
                    "colno": e.colno,
                    "snippet": snippet,
                }
            )
            logging.debug(err)
            return (False, err)

        try:
            jsonschema.validate(instance=instance, schema=schema)
            return (True, "")
        except jsonschema.ValidationError as e:
            path = (
                "/" + "/".join(str(p) for p in e.absolute_path)
                if e.absolute_path
                else "<root>"
            )
            # For oneOf/anyOf failures, the top-level message is generic
            # ("is not valid under any of the given schemas") and the
            # actionable detail lives in `e.context` (one sub-error per
            # branch). We list every branch verbatim instead of collapsing to
            # `best_match`: at the directive-level oneOf, all variants share
            # key shapes, so `best_match`'s depth-then-leaf scoring
            # systematically promotes a shallow `required` failure from the
            # wrong variant (e.g. "condition is required" against
            # transformational_rule) over the deep regex failure of the
            # variant the model is actually targeting (e.g. /0/action does
            # not match the event regex against deontic_frame). Drilling each
            # branch to its leaf is still done with `best_match` — within a
            # single branch the heuristic is fine — so the model sees a
            # pattern/regex-level message per variant rather than the generic
            # nested-oneOf wrapper.
            branches: list[dict[str, str]] = []
            if e.context:
                def _leaf(s: jsonschema.ValidationError) -> jsonschema.ValidationError:
                    while s.context:
                        s = jsonschema.exceptions.best_match(s.context)
                    return s
                # `e.context` flattens to leaves across all branches and may
                # repeat the same message verbatim (typically the
                # `additionalProperties: false` failure, which fires on every
                # variant whenever the model emits keys none of them want).
                # Dedupe on (message, path) to keep the prompt compact while
                # preserving each distinct failure mode the model can use to
                # pick a target variant.
                seen: set[tuple[str, str]] = set()
                for sub in e.context:
                    leaf = _leaf(sub)
                    leaf_path = (
                        "/" + "/".join(str(p) for p in leaf.absolute_path)
                        if leaf.absolute_path
                        else "<root>"
                    )
                    key = (leaf.message, leaf_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    branches.append({"message": leaf.message, "path": leaf_path})

            err = self.tmpl_error_validation.render(
                {
                    "validator": e.validator,
                    "validator_value": json.dumps(e.validator_value)[:200],
                    "path": path,
                    "instance": json.dumps(e.instance)[:200],
                    "message": e.message,
                    "branches": branches,
                }
            )
            logging.debug(err)
            return (False, err)
