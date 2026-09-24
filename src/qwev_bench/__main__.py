"""Commands that import GPU dependencies only for an actual benchmark."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwev: Qwen3-VL decision-ready latency on one GPU")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a CUDA benchmark, or validate without loading a model")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True, help="New JSONL file; existing results are never overwritten")
    run.add_argument("--model", help="Override the model in the configuration")
    run.add_argument("--dry-run", action="store_true", help="Validate and print the resolved config; no GPU/downloads")
    fixtures = sub.add_parser("fixtures", help="Generate reproducible synthetic RGB inputs without ML dependencies")
    fixtures.add_argument("--output", default="assets")
    summarize = sub.add_parser("summarize", help="Aggregate measured JSONL results into CSV")
    summarize.add_argument("inputs", nargs="+")
    summarize.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fixtures":
            from .fixtures import generate_fixtures
            for path in generate_fixtures(Path(args.output)):
                print(path)
            return 0
        if args.command == "summarize":
            from .reporting import summarize_files
            count = summarize_files(args.inputs, Path(args.output))
            print(f"Wrote {count} groups to {args.output}")
            return 0
        config = load_config(args.config, model=args.model)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            print("Validated only: no model loaded, no results written.", file=sys.stderr)
            return 0
        output = Path(args.output)
        if output.exists():
            raise ValueError(f"Output already exists: {output}; choose a new filename")
        from .runner import run as run_benchmark
        return run_benchmark(config, output)
    except (ValueError, OSError, ImportError) as error:
        print(f"qwev-bench: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
