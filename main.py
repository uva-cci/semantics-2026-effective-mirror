import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.pipeline import MirroringPipeline, Scenario


def _main() -> None:
    args = parse_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = load_config(args.config)

    scenarios: list[Scenario] = []
    with open(cfg.scenarios, "r") as f:
        raw = json.load(f)
        for scenario in raw:
            scenarios.append(Scenario(**scenario))

    MirroringPipeline(cfg, scenarios).run()
    logging.info("Experiment completed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mirror",
        description="Testing normative specification languages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=False,
        default=Path("config.yaml"),
        metavar="FILE",
        help="Path to the configuration file in YAML format.",
    )

    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Enable debug logging output.",
    )

    return parser.parse_args(argv)


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
