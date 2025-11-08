# bookworm contains a vulnerability apparently, using nightly
FROM python:3.13-slim

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=.cache/uv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv Python package manager
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN --mount=type=ssh \
    --mount=type=cache,target=$UV_CACHE_DIR \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY . /app

# Sync Python dependencies
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --frozen

VOLUME /app/data

ENTRYPOINT ["uv", "run", "main.py"]
