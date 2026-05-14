# AGENTS.md

Contributor reference for the `semantics-2026-effective-mirror` repository.

## Scope

Two purposes in one repo:

1. **Reproduction package** for the SEMANTICS 2026 submission on neurosymbolic
   mirroring of normative specifications.
2. **Reusable pipeline** for measuring modelling fitness of normative DSLs
   that use JSON as transport with a JSON Schema information model.

Each scenario is run through encode → decode → re-encode and scored against
the original. DSLs currently wired up: `dcpl` and `odrl` (under `inputs/dsl/`).

Production (the LLM matrix) and analysis (structural + semantic scoring) are
two independent subcommands. `run` writes raw cells as NDJSON; `analyze`
reads that NDJSON and writes a scores CSV. They never run together —
production cannot be slowed down by scoring, and scoring can be re-done
without touching the LLM matrix.

## Layout

```
main.py                         CLI entrypoint (argparse subcommands: run, analyze, visualize)
src/config.py                   Pydantic config models + YAML loader
src/pipeline.py                 Mirroring pipeline (encode/decode, resume) — production only
src/analyze.py                  Standalone scoring: NDJSON → scores CSV
src/visualize.py                Paper-figure rendering: scores CSV → PDF figures + summary CSV
src/utils/models.py             Backend dispatch (Ollama, OpenAI, Anthropic, Google)
src/utils/embeddings.py         Sentence-transformer encoders
src/utils/structural.py         Structural similarity scoring for nested JSON
src/prompts/*.jinja             Encode/decode/legenda/refine/error templates
inputs/scenarios{,.smoke}.json  Scenario corpora
inputs/dsl/<name>/{schema,examples}.json
outputs/                        Iterative run NDJSON + analyze CSV (gitignored)
outputs/published/              Curated, paper-ready artefacts (tracked)
config.yaml                     Full experimental setup
config.smoke.yaml               Minimal end-to-end check
Dockerfile, .env.example
```

## Development cycle

1. `uv sync` — installs runtime + dev deps and registers the `mirror` script.
2. Edit code under `src/` (or prompts under `src/prompts/`).
3. Smoke-run the full loop: `uv run mirror --config config.smoke.yaml run`.
   Writes `outputs/output-smoke.ndjson`; resume works if the file is kept.
4. **Before finalising any change, run both type checkers and treat new
   diagnostics as a blocker:**

   ```sh
   uv run pyright
   uv run ty check
   ```

5. Commit. There is no test suite, linter, or pre-commit hook — type checking
   plus a smoke run is the de-facto gate.

## CLI

`uv run mirror` is a subcommand parser. Top-level flags:

| Flag           | Default       | Purpose                              |
| -------------- | ------------- | ------------------------------------ |
| `-c, --config` | `config.yaml` | Path to the YAML config.             |
| `-d, --debug`  | off           | DEBUG-level logging.                 |

### `run` — produce datapoints

| Flag           | Default                              | Purpose                                                                                                |
| -------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `-o, --output` | `outputs/output-<timestamp>.ndjson`  | NDJSON output path; overrides `output:` in the config. Required as a stable path for resume to engage. |

`run` writes raw cells only — no structural or semantic scoring.

### `analyze INPUT_NDJSON` — score an existing run

| Flag           | Default                            | Purpose                                                       |
| -------------- | ---------------------------------- | ------------------------------------------------------------- |
| `-o, --output` | `outputs/<input-stem>.scores.csv`  | CSV output path. Always rewritten in full (atomic rename).    |

Reads an NDJSON produced by `run`, computes structural ratios (`alignment`,
`type_consistency`, `content_fidelity`, list-only `matching`) and per-encoder
semantic cosine similarities, and emits a CSV with one row per cell.
Useful for swapping the encoder set or adding new metrics without re-running
inference. In-place rewrite is rejected — the original NDJSON stays immutable.

### `visualize INPUT_CSV` — render paper figures from a scores CSV

| Flag           | Default                              | Purpose                                                                  |
| -------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `-o, --output` | `outputs/figures/<input-stem>/`      | Output directory for the three PDF figures and `summary_stats.csv`.       |

Reads a scores CSV produced by `analyze` and emits three publication-quality
PDFs plus a `summary_stats.csv` that lists every numeric value plotted
(means, CIs, medians, correlation coefficients). Every figure facets by
model — there is no across-models aggregation. Figures:

- `fig1_ablation.pdf` — 2×2 factorial effect of `syntax` and `few_shot`
  per metric, one row per model. A top strip (one panel per model) shows
  the `sym1 ∧ sym2` success rate behind the means.
- `fig2_dsl.pdf` — DSL comparison faceted by model (rows) × metric (cols),
  restricted to the best-supported ablation (`syntax=1, few_shot=1`) and
  to successful round-trips. `matching` is dcpl-only by construction
  (odrl is dict-vs-dict at the top level, so there is nothing to pair) —
  the odrl side renders an `n/a` annotation rather than fabricated data.
- `fig3_correlation.pdf` — scatter of the structural composite (mean of
  populated ratios) against semantic similarity, faceted by DSL (rows) ×
  model (cols), with Spearman ρ and Pearson r per panel.

Does not require a config file; it consumes the CSV directly. Each figure
is written via an atomic rename, so a failed run never overwrites a
previous PDF.

## Configuration

YAML, validated by Pydantic models in `src/config.py`. A config selects the
scenario set, DSL list (each with `schema.json` + `examples.json`), legenda
schema, retry budget, model registry (local Ollama + cloud Anthropic /
OpenAI / Google), semantic-scoring encoders, and the inference matrix
(`concurrency` / `constants` / `defaults` / `per_backend`). Treat `config.yaml`
as the canonical reference rather than re-documenting keys here.

## Smoke testing

`config.smoke.yaml` shrinks the matrix to one local model (`gemma4-e4b`), one
DSL, one scenario file (`inputs/scenarios.smoke.json`), and a fixed output
(`outputs/output-smoke.ndjson`). Use it before any commit that touches the pipeline,
prompts, or model dispatch — it exercises the full encode → decode →
re-encode → score loop in minutes without burning cloud credits. Requires
`ollama serve` with `gemma4:e4b` pulled.

## API keys and Ollama

Cloud providers read keys from the environment:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`. Copy `.env.example`
to `.env`. Local models require `ollama serve` on the host; override
`OLLAMA_HOST` if it isn't `http://localhost:11434`.

Any `provider: openai` entry can carry `base_url` (e.g. `http://0.0.0.0:8000/v1`)
and `api_key_env` (e.g. `LOCAL_OPENAI_API_KEY`) to redirect through an
OpenAI-compatible server or router. When `api_key_env` is unset the SDK
receives the placeholder `"EMPTY"`. The override is only honoured for
`provider: openai` — Anthropic / Google entries reject it.

For local llama.cpp inference there is a dedicated `kind: local` driver
(`driver: llamacpp`) with its own `model_id` / optional `base_url` /
`api_key_env` / `server` fields and its own `per_backend.llamacpp` sweep
matrix. The wrapper speaks the OpenAI-compatible API exposed by
`llama-server`.

Two modes, picked by whether `base_url` is set:

- **Managed (default; `base_url` unset).** The pipeline spawns one
  `llama-server` subprocess per entry on an auto-allocated port for the
  lifetime of that model's task group, and tears it down when the group
  finishes. The `server` block on the entry maps to real CLI flags at spawn
  time (`context_size` → `-c`, `gpu_layers` → `-ngl`, `batch_size` → `-b`,
  `threads` → `-t`, `flash_attn` → `--flash-attn`, plus `extra_args` as a
  pass-through list). Each spawned server gets `--parallel
  concurrency.llamacpp` so internal slots match the worst-case routing.
  `llama-server` must be on `PATH` (or pin `server.binary` to an absolute
  path); the binary builds itself, this repo carries no Metal/CUDA/build
  glue. First-run `-hf <repo>:<quant>` downloads cache to
  `~/.cache/llama.cpp/`.
- **External (`base_url` set).** The pipeline does *not* spawn anything;
  it pre-flights `<base_url>/models` to confirm the configured `model_id`
  is loaded and then speaks HTTP exactly as today's flow.

`concurrency.llamacpp` is a global in-flight-request cap (one process-wide
semaphore shared by every llama.cpp entry), the same shape as
`concurrency.ollama`. Multiple managed entries spawn multiple processes
simultaneously — size the matrix to your hardware.

## Resume semantics

Each output row carries a deterministic `cell_key` (hash of model × scenario
× DSL × ablation × params). On startup the pipeline scans the NDJSON output
and skips keys already present, so reruns against a stable `--output` path
are idempotent. Validation failures are persisted and treated as done; they
will not be retried on resume.

## Docker (optional)

The image's entrypoint is `mirror`. Mount `/app/inputs` (project assets),
`/app/outputs` (NDJSON artefacts), and `/root/.cache/huggingface` (encoder
cache, downloaded once). Ollama is reached via `OLLAMA_HOST` —
`host.docker.internal` on macOS/Windows Docker Desktop, `--network=host` on
Linux. See `README.md` for full recipes.

## Adding a new DSL

1. Drop `inputs/dsl/<name>/schema.json` and `inputs/dsl/<name>/examples.json`.
2. Add a `dsl:` entry in your config pointing at both files.
3. Keep identifiers in shared artefacts portable: `[A-Za-z][A-Za-z0-9]*`.
