# Reproduction Package for SEMANTICS 2026

This repository contains the code, input data, and generated outputs for the pipelines proposed in our SEMANTICS 2026 submission.

## Usage

### Configuration

The experimental setup is configured via YAML. Three ready-made configs are provided:

- [`config.yaml`](./config.yaml) — full setup (cloud + local Ollama models)
- [`config.cloud.yaml`](./config.cloud.yaml) — cloud providers only (Anthropic, OpenAI, Gemini)
- [`config.local.yaml`](./config.local.yaml) — local Ollama models only

Pass any config file with `--config FILE`.

### Local Execution

With the [uv](https://docs.astral.sh/uv/) package manager:

```sh
uv sync                                       # install dependencies + the `mirror` CLI
uv run mirror --config config.yaml            # run the pipelines
uv run mirror --help                          # see all options
```

Cloud providers read their API tokens from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). Copy [`.env.example`](./.env.example) to `.env` and source it, or export the variables in your shell.

Local Ollama models require `ollama serve` running on the host. By default the Ollama SDK connects to `http://localhost:11434`; override with `OLLAMA_HOST` if needed.

### Docker Execution

The [`Dockerfile`](./Dockerfile) builds an image whose entrypoint is the `mirror` CLI. To avoid bloating the image with model and encoder binaries (e.g. GGUF files), the container expects `/app/data` to be mounted as a volume. Reuse that volume across invocations.

```sh
docker build -t semantics-2026-mirror .
```

#### Recipe A — cloud providers (Anthropic / OpenAI / Gemini)

API tokens are injected via `--env-file`:

```sh
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/config.cloud.yaml:/app/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  semantics-2026-mirror --config /app/config.yaml
```

#### Recipe B — Ollama running on the host

Ollama must already be running on the host (`ollama serve`). The container reaches the host daemon via the `OLLAMA_HOST` env var; the exact value depends on your platform.

**macOS / Windows (Docker Desktop):**

```sh
docker run --rm -it \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$(pwd)/config.local.yaml:/app/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  semantics-2026-mirror --config /app/config.yaml
```

**Linux:**

```sh
docker run --rm -it \
  --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v "$(pwd)/config.local.yaml:/app/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  semantics-2026-mirror --config /app/config.yaml
```

Alternatively on Linux, instead of `--network=host`:

```sh
--add-host=host.docker.internal:host-gateway \
-e OLLAMA_HOST=http://host.docker.internal:11434
```
