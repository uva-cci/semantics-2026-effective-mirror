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
main.py                         CLI entrypoint (argparse subcommands: run, analyze)
src/config.py                   Pydantic config models + YAML loader
src/pipeline.py                 Mirroring pipeline (encode/decode, resume) — production only
src/analyze.py                  Standalone scoring: NDJSON → scores CSV
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
