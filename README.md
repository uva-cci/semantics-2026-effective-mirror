# Reproduction Package for SEMANTICS 2026

This repository contains the code, input data, and generated outputs for the pipelines proposed in our SEMANTICS 2026 submission.

## Usage

### Configuration

The experimental setup is configured via YAML. Two ready-made configs are provided:

- [`config.yaml`](./config.yaml) — full setup (cloud + local Ollama models)
- [`config.smoke.yaml`](./config.smoke.yaml) — minimal end-to-end smoke configuration

Pass either with `--config FILE`. Cloud-provider entries are activated only when their respective API keys are present in the environment; entries without keys are skipped.

### Local Execution

With the [uv](https://docs.astral.sh/uv/) package manager:

```sh
uv sync                                                        # install dependencies + the `mirror` CLI
uv run mirror --config config.yaml run                         # produce datapoints (encode → decode → re-encode)
uv run mirror --config config.smoke.yaml run                   # quick end-to-end smoke check
uv run mirror --config config.yaml analyze outputs/run.ndjson  # score a previously-produced NDJSON → CSV
uv run mirror visualize outputs/run.scores.csv                 # render paper-ready PDF figures from a scores CSV
uv run mirror --help                                           # see all options
```

The CLI has two independent subcommands:

- **`run`** — drives the LLM matrix and writes one NDJSON row per cell (`outputs/output-<timestamp>.ndjson` by default). Production only; no scoring.
- **`analyze INPUT_NDJSON`** — reads an NDJSON produced by `run` and emits a scores CSV (default `outputs/<input-stem>.scores.csv`). One row per cell, with structural ratios and per-encoder semantic similarities as columns. Always rewritten in full (analysis is cheap; rerun freely after swapping encoders or adding metrics).
- **`visualize INPUT_CSV`** — reads a scores CSV and renders three paper-ready PDF figures (ablation effects, DSL comparison, structural-vs-semantic agreement) plus a `summary_stats.csv` with the underlying numbers. Defaults to `outputs/figures/<input-stem>/`; override with `-o`.

Outputs land under `outputs/` (gitignored); inputs (scenarios + DSL schemas/examples) live under `inputs/`.

Cloud providers read their API tokens from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). Copy [`.env.example`](./.env.example) to `.env` and source it, or export the variables in your shell.

Any `provider: openai` entry can additionally set `base_url` and (optionally) `api_key_env` to redirect through an OpenAI-compatible server or router (vLLM, llama.cpp, LiteLLM). When `api_key_env` is omitted the key is sent as `"EMPTY"`. Anthropic / Google entries have provider-specific endpoint conventions and are not covered by this override.

Local Ollama models require `ollama serve` running on the host. By default the Ollama SDK connects to `http://localhost:11434`; override with `OLLAMA_HOST` if needed.

For local `driver: llamacpp` entries the pipeline spawns and owns one `llama-server` subprocess per entry by default — the `server:` block on the entry (`context_size`, `gpu_layers`, `batch_size`, `threads`, `flash_attn`, `extra_args`) maps directly to CLI flags at spawn time. `llama-server` must be on `PATH` (or pin `server.binary` to an absolute path); first-run `-hf <repo>:<quant>` downloads land in `~/.cache/llama.cpp/`. Set `local_path:` on the entry to load a `.gguf` already on disk (via `-m`) instead of downloading from HuggingFace; `model_id` then becomes the alias served at `/v1/models` (via `-a`). To opt out and point the pipeline at an externally-launched server instead, set `base_url:` on the entry — the pipeline will skip the spawn and pre-flight `<base_url>/models` to verify the configured `model_id` is loaded.

### Docker Execution

The [`Dockerfile`](./Dockerfile) builds an image whose entrypoint is the `mirror` CLI. The container expects three mounts:

- `/app/inputs` — project assets (scenarios + DSL grammars/schemas/examples), bind-mounted from the host so configs can reference them.
- `/app/outputs` — bind for the NDJSON output files produced by `run` and `analyze`. Bind to a host directory so artefacts survive container restarts.
- `/root/.cache/huggingface` — sentence-transformer encoder cache, persisted across container restarts so encoders are downloaded only once. A named Docker volume (e.g. `hf-cache`) or a host bind to your existing `~/.cache/huggingface` both work.

Ollama models live in the daemon's own cache on the host, so no volume is needed for them inside the container.

```sh
docker build -t semantics-2026-mirror .
```

#### Recipe A — cloud providers (Anthropic / OpenAI / Gemini)

API tokens are injected via `--env-file`:

```sh
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v hf-cache:/root/.cache/huggingface \
  semantics-2026-mirror --config /app/config.yaml run
```

#### Recipe B — Ollama running on the host

Ollama must already be running on the host (`ollama serve`). The container reaches the host daemon via the `OLLAMA_HOST` env var; the exact value depends on your platform.

**macOS / Windows (Docker Desktop):**

```sh
docker run --rm -it \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v hf-cache:/root/.cache/huggingface \
  semantics-2026-mirror --config /app/config.yaml run
```

**Linux:**

```sh
docker run --rm -it \
  --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v hf-cache:/root/.cache/huggingface \
  semantics-2026-mirror --config /app/config.yaml run
```

Alternatively on Linux, instead of `--network=host`:

```sh
--add-host=host.docker.internal:host-gateway \
-e OLLAMA_HOST=http://host.docker.internal:11434
```
