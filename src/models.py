import asyncio as aio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, cast, override

import aiofiles
import aiohttp
from llama_cpp import CreateCompletionResponse, Llama
from openai import OpenAI
from tqdm import tqdm

from src.config import CloudModelConfig, Config, LocalModelConfig, ModelConfig


class InferenceModel(ABC):
    """
    Base class that defines the minimal API for text generation.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> str:
        """
        Return a single generated text for the given prompt.

        Parameters
        ----------
        prompt: str
            Input prompt.

        Returns
        -------
        str
            Generated text.
        """
        ...


class LocalInferenceModel(InferenceModel):
    """
    Local inference using a compiled llama.cpp model.
    """

    def __init__(self, name: str, cfg: LocalModelConfig) -> None:
        super().__init__(name)

        self.path = get_model_path(name)
        self.model = Llama(model_path=str(self.path))

    @override
    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> str:
        result = self.model(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return cast(CreateCompletionResponse, result)["choices"][0]["text"]


openai_client: OpenAI


class OpenAIInferenceModel(InferenceModel):
    """
    Cloud inference using the OpenAI API.
    """

    def __init__(self, name: str, cfg: CloudModelConfig) -> None:
        super().__init__(name)

        global openai_client
        if not openai_client:
            openai_client = OpenAI()

        self.cfg = cfg

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> str:
        resp = openai_client.completions.create(
            model=self.cfg.model_id,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
        )
        return resp.choices[0].text


def get_model(cfg: ModelConfig) -> InferenceModel:
    """
    Return an inference model instance.

    Parameters
    ----------
    backend : str
        "local" or "openai".
    config : dict
        Arguments forwarded to the concrete constructor.
    """
    match cfg.meta.kind:
        case "local":
            local_meta = cast(LocalModelConfig, cfg.meta)
            return LocalInferenceModel(cfg.name, local_meta)
        case "cloud":
            cloud_meta = cast(CloudModelConfig, cfg.meta)
            match cloud_meta.provider:
                case "openai":
                    return OpenAIInferenceModel(cfg.name, cloud_meta)
                case _:
                    raise ValueError(f"Unknown provider {cloud_meta.provider}")
    raise ValueError()


def get_model_path(name: str) -> Path:
    return Path("data/models") / f"{name}.gguf"


async def download_model(url: str, dest: Path, chunk_size: int = 64 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logging.info(f"✓ Skipping existing file: {dest}")
        return

    logging.info(f"Downloading {url} → {dest}")

    try:
        async with aiohttp.ClientSession(
            # downloads are several GBs large and take time
            timeout=aiohttp.ClientTimeout(total=None, sock_read=30)
        ) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()

                total = int(resp.headers.get("Content-Length") or 0)

                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,  # e.g. 1.5 MiB instead of 1572864 B
                    unit_divisor=1024,
                    desc=dest.name,
                    dynamic_ncols=True,  # auto‑adjust width
                    colour="cyan"
                ) as pbar:
                    async with aiofiles.open(dest, mode="wb") as fp:
                        async for chunk in resp.content.iter_chunked(chunk_size):
                            await fp.write(chunk)
                            pbar.update(len(chunk))

    except Exception as e:
        # we don't want partial downloads to be interpreted as completed
        dest.unlink()
        raise e

    logging.info(f"✓ Done: {dest}")


async def download_models(cfg: Config) -> None:
    logging.info("Downloading GGUF models.")

    tasks: list[Awaitable[Any]] = []
    for model in cfg.models:
        if model.meta.kind != "local":
            logging.warning(f"Skipping cloud model entry: {model.name}")
            continue

        dest = get_model_path(model.name)
        tasks.append(download_model(model.meta.url.encoded_string(), dest))

    await aio.gather(*tasks)

    logging.info("All models downloads finished.")
