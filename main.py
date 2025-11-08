import asyncio as aio
import json
import logging
import sys

from src.config import load_config
from src.embeddings import download_encoders
from src.mirroring import MirroringPipeline
from src.models import download_models
from src.pipeline import Scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


async def _main() -> None:
    cfg = load_config()

    await aio.gather(download_models(cfg), download_encoders(cfg))

    scenarios: list[Scenario] = []
    with open(cfg.scenarios, "r") as f:
        raw = json.load(f)
        for scenario in raw:
            scenarios.append(Scenario(**scenario))

    for name in cfg.pipelines:
        logging.info(f"Starting experiment pipeline: {name}")

        match name:
            case "mirroring":
                MirroringPipeline(cfg, scenarios).run()
            case _:
                logging.error(f"Unimplemented pipeline type {name}")

        logging.info("Experiment completed")


def main() -> None:
    aio.run(_main())


if __name__ == "__main__":
    main()
