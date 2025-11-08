# Reproduction Package for NLL2FR (JURIX) 2025

This repository contains the code implementation, input data, and generated outputs for
the pipelines proposed in the relative workshop paper.

## Usage

### Configuration

The experimental setup can be configured by changing the provided [`config.yaml`](./config.yaml), or by creating a new configuration YAML file and passing its path as a command line argument when running the package.

### Executing Experiments

#### Local Execution

Having the [uv](https://docs.astral.sh/uv/) package manager installed one can simply run the following commands:

```sh
uv sync             # install dependencies
uv run main.py      # run the experiments
```

#### Docker Execution

The package also contains a [`Dockerfile`](./Dockerfile) for containerized execution. To avoid cluttering the image with model and encoder binaries (e.g. GGUF files) the container use a volume. The user is encouraged to reuse such volume for every invocation as shown in the following command:

```sh
docker build -t nll2fr-pipelines .

docker run --rm \
  -it \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  nll2fr-pipelines --config /app/config.yaml
```
