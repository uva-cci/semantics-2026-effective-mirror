import argparse
import json
import logging
import sys
from pathlib import Path

from src.analyze import run_analyze
from src.config import load_config
from src.pipeline import MirroringPipeline, Scenario


def _cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.output is not None:
        cfg.output = args.output

    scenarios: list[Scenario] = []
    with open(cfg.scenarios, "r") as f:
        raw = json.load(f)
        for scenario in raw:
            scenarios.append(Scenario(**scenario))

    MirroringPipeline(cfg, scenarios).run()
    logging.info("Experiment completed")


def _cmd_analyze(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    input_path: Path = args.input
    output_path: Path = args.output or (
        Path("outputs") / f"{input_path.stem}.scores.csv"
    )
    run_analyze(cfg, input_path, output_path)
    logging.info(f"Analysis written to {output_path}")


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

    parser.add_argument(
        "--show-llama-server",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Show stdout/stderr forwarded from managed llama-server "
            "subprocesses. Muted by default to keep pipeline progress legible. "
            "Implied by --debug."
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Produce datapoints (encode → decode → re-encode) as NDJSON cells.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_run.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        metavar="FILE",
        help=(
            "NDJSON output path. Overrides `output:` in the config. Resume "
            "against a file requires a stable path here or in the config; "
            "the timestamp default never resumes."
        ),
    )
    p_run.set_defaults(func=_cmd_run)

    p_analyze = sub.add_parser(
        "analyze",
        help="Score an NDJSON produced by `mirror run` and emit a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_analyze.add_argument(
        "input",
        type=Path,
        metavar="INPUT_NDJSON",
        help="Path to the NDJSON file to score.",
    )
    p_analyze.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        metavar="FILE",
        help=(
            "Path for the scores CSV. Defaults to "
            "`outputs/<input-stem>.scores.csv`. Always rewritten in full."
        ),
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Child-logger filtering runs before propagation, so raising the level here
    # drops the forwarded stdout (INFO) / stderr (WARNING) lines before they
    # ever reach the root handler. The manager's own lifecycle log lines stay
    # on the root logger and remain visible.
    if not args.debug and not args.show_llama_server:
        logging.getLogger("llama_server.subprocess").setLevel(logging.ERROR + 1)

    args.func(args)


if __name__ == "__main__":
    main()
