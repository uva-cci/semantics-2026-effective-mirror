import asyncio as aio
import logging
import sys

from src.config import load_config
from src.embeddings import download_encoders
from src.mirroring import MirroringPipeline
from src.models import download_models
from src.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


async def _main() -> None:
    cfg = load_config()

    await aio.gather(download_models(cfg), download_encoders(cfg))

    logging.info("Starting experiment")

    pipeline: Pipeline
    match cfg.pipeline:
        case "mirroring":
            pipeline = MirroringPipeline(cfg)
            pipeline.run()
        case _:
            logging.error(f"unsupported pipeline type {cfg.pipeline}")

    logging.info("Experiment completed")


def main() -> None:
    aio.run(_main())


if __name__ == "__main__":
    main()
