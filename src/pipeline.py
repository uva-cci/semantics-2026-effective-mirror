import itertools
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

from pydantic import BaseModel, Field

from .config import Config, DSLConfig, DSLValidationConfig, PipelineName
from .models import InferenceModel, InferenceParams, get_model


class Scenario(BaseModel):
    id: str
    source: Any
    description: str


class PipelineOutput(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    scenario_id: str
    pipeline: PipelineName
    model: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc))


class AblationFlags(BaseModel):
    syntax: bool
    few_shot: bool


class Pipeline[T: PipelineOutput](ABC):

    def __init__(self, cfg: Config, scenarios: list[Scenario]) -> None:
        self.cfg = cfg
        self.scenarios = scenarios
        self.output: Queue[T | None] = Queue()

    def run(self) -> None:
        logging.info(f"Pipeline {self.__class__.__name__} started")

        t = threading.Thread(
            target=self.writer_worker,
            args=(Path(f"{self.__class__.__name__}.ndjson"),),
            daemon=True
        )
        t.start()

        # create permutations of inference parameters based on the config
        inference_profiles = itertools.product(
            self.cfg.inference.temperature,
            self.cfg.inference.top_p,
            self.cfg.inference.top_k,
        )

        for scenario in self.scenarios:
            logging.info(f"Running scenario: {scenario.id}")
            for dsl in self.cfg.dsl:
                logging.info(f"- DSL: {dsl.name}")
                for validation in dsl.validation:
                    logging.info(f"- Validation: {validation.kind}")
                    for model_cfg in self.cfg.models:
                        logging.info(f"- Model: {model_cfg.name}")
                        model = get_model(model_cfg)
                        for abl in [
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

    @abstractmethod
    def produce_datapoint(
        self,
        scenario: Scenario,
        dsl: DSLConfig,
        validation: DSLValidationConfig,
        model: InferenceModel,
        ablation: AblationFlags,
        params: InferenceParams,
    ) -> T:
        ...

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
