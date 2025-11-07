
import asyncio as aio
import logging
from pathlib import Path
from typing import Any, Awaitable

from sentence_transformers import SentenceTransformer

from src.config import Config, EmbeddingModelConfig


def get_encoder_path(cfg: EmbeddingModelConfig) -> Path:
    return Path(f"data/encoders/{cfg.org}/{cfg.name}")


def get_encoder(cfg: EmbeddingModelConfig) -> SentenceTransformer:
    path = get_encoder_path(cfg)
    return SentenceTransformer(str(path))


async def download_encoder(cfg: EmbeddingModelConfig) -> None:
    dest = get_encoder_path(cfg)
    fullname = f"{cfg.org}/{cfg.name}"

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logging.info(f"✓ Skipping existing file: {dest}")
        return

    logging.info(f"Downloading {fullname} → {dest}")

    encoder = SentenceTransformer(fullname, trust_remote_code=True)
    encoder.save(str(dest))

    logging.info(f"✓ Done: {dest}")


async def download_encoders(cfg: Config) -> None:
    logging.info("Downloading 'sentence_transformers' encoders.")

    tasks: list[Awaitable[Any]] = []
    for transformer in cfg.encodings:
        tasks.append(download_encoder(transformer))

    await aio.gather(*tasks)

    logging.info("All encoders downloads finished.")
