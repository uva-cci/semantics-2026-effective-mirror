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
uv sync                                       # install dependencies + the `mirror` CLI
uv run mirror --config config.yaml            # run the pipelines
uv run mirror --config config.smoke.yaml      # quick end-to-end smoke check
uv run mirror --help                          # see all options
```

Cloud providers read their API tokens from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). Copy [`.env.example`](./.env.example) to `.env` and source it, or export the variables in your shell.

Local Ollama models require `ollama serve` running on the host. By default the Ollama SDK connects to `http://localhost:11434`; override with `OLLAMA_HOST` if needed.

### Docker Execution

The [`Dockerfile`](./Dockerfile) builds an image whose entrypoint is the `mirror` CLI. The container expects two mounts:

- `/app/data` — project assets (scenarios + DSL grammars/schemas/examples), bind-mounted from the host so configs can reference them.
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
  -v "$(pwd)/data:/app/data" \
  -v hf-cache:/root/.cache/huggingface \
  semantics-2026-mirror --config /app/config.yaml
```

#### Recipe B — Ollama running on the host

Ollama must already be running on the host (`ollama serve`). The container reaches the host daemon via the `OLLAMA_HOST` env var; the exact value depends on your platform.

**macOS / Windows (Docker Desktop):**

```sh
docker run --rm -it \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  -v hf-cache:/root/.cache/huggingface \
  semantics-2026-mirror --config /app/config.yaml
```

**Linux:**

```sh
docker run --rm -it \
  --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  -v hf-cache:/root/.cache/huggingface \
  semantics-2026-mirror --config /app/config.yaml
```

Alternatively on Linux, instead of `--network=host`:

```sh
--add-host=host.docker.internal:host-gateway \
-e OLLAMA_HOST=http://host.docker.internal:11434
```
