import asyncio
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.pipeline import MirroringPipelineOutput
from src.utils.embeddings import get_encoder, score_vectors
from src.utils.structural import StructuralScores
from src.utils.structural import structural_scores as compute_structural_scores

# Stable structural columns (the four ratios returned by `structural_scores`).
# `matching` is only populated for list-vs-list inputs; the others always are
# when both symbolic outputs validated.
STRUCTURAL_FIELDS = ("matching", "alignment", "type_consistency", "content_fidelity")

CSV_COLUMNS = (
    "cell_key",
    "scenario_id",
    "model",
    "dsl",
    "ablation_syntax",
    "ablation_few_shot",
    "sym1_success",
    "sym2_success",
    *(f"structural_{f}" for f in STRUCTURAL_FIELDS),
    "semantic",
)


def _encode_pair(
    encoder: SentenceTransformer,
    a_text: str,
    b_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embs = encoder.encode(
        [a_text, b_text],
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    return embs[0], embs[1]


def _iter_rows(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _structural_for(row: MirroringPipelineOutput) -> StructuralScores | None:
    if not row.symbolic_output1.success:
        return None
    if row.symbolic_output2 is None or not row.symbolic_output2.success:
        return None
    return compute_structural_scores(
        json.loads(row.symbolic_output1.text),
        json.loads(row.symbolic_output2.text),
    )


async def _semantic_for(
    row: MirroringPipelineOutput,
    scenario_text: str,
    encoder: SentenceTransformer,
    encoder_lock: asyncio.Lock,
) -> float | None:
    if not row.symbolic_output1.success or row.natural_language is None:
        return None
    async with encoder_lock:
        a, b = await asyncio.to_thread(
            _encode_pair, encoder, scenario_text, row.natural_language.text,
        )
    return score_vectors(a, b)


def _row_to_csv(
    row: MirroringPipelineOutput,
    structural: StructuralScores | None,
    semantic: float | None,
) -> list[Any]:
    sym2_success = (
        row.symbolic_output2.success if row.symbolic_output2 is not None else ""
    )
    structural_cells: list[Any] = []
    for f in STRUCTURAL_FIELDS:
        v = None if structural is None else getattr(structural, f)
        structural_cells.append("" if v is None else v)

    return [
        row.cell_key,
        row.scenario_id,
        row.model,
        row.dsl.name,
        row.ablation.syntax,
        row.ablation.few_shot,
        row.symbolic_output1.success,
        sym2_success,
        *structural_cells,
        "" if semantic is None else semantic,
    ]


async def analyze_ndjson(
    input_path: Path,
    output_path: Path,
    scenarios: dict[str, str],
    encoder: SentenceTransformer,
    encoder_lock: asyncio.Lock,
) -> None:
    """Score every row of `input_path` and emit a CSV at `output_path`.

    The CSV is written atomically: rows go to a sibling `.tmp` file which is
    renamed at the end, so a partial run never overwrites a previous CSV.
    Re-running always rescores from scratch — analysis is cheap relative to
    the inference matrix it consumes.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            f"In-place analysis is not supported: input and output point to "
            f"the same file ({input_path})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    scored = 0
    missing_scenario = 0

    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for raw_row in _iter_rows(input_path):
            row = MirroringPipelineOutput.model_validate(raw_row)
            scenario_text = scenarios.get(row.scenario_id)
            if scenario_text is None:
                missing_scenario += 1
                logging.warning(
                    f"scenario_id={row.scenario_id} not found in scenarios; "
                    f"semantic score will be empty for cell_key={row.cell_key}"
                )
                scenario_text = ""

            structural = _structural_for(row)
            semantic = await _semantic_for(row, scenario_text, encoder, encoder_lock)
            writer.writerow(_row_to_csv(row, structural, semantic))
            scored += 1

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, output_path)
    logging.info(
        f"Analyze done: scored={scored}, missing_scenario={missing_scenario}"
    )


def load_scenarios_map(cfg: Config) -> dict[str, str]:
    """Return a `scenario_id -> description` mapping from the configured corpus."""
    with cfg.scenarios.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {s["id"]: s["description"] for s in raw}


def run_analyze(cfg: Config, input_path: Path, output_path: Path) -> None:
    scenarios = load_scenarios_map(cfg)
    encoder = get_encoder(cfg.encoding)
    lock = asyncio.Lock()
    asyncio.run(analyze_ndjson(input_path, output_path, scenarios, encoder, lock))
