"""Retry selected failed datasets using the existing model entrypoint settings."""
import argparse
import json
import runpy
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS.parent))
sys.path.insert(0, str(RUNS))
from _shared.common import run
from run_models import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=dict(METHODS))
    parser.add_argument("--datasets", required=True, help="JSON array of failed dataset names")
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    entrypoint = RUNS / args.model / dict(METHODS)[args.model]
    settings = runpy.run_path(str(entrypoint))["SETTINGS"].copy()
    settings["datasets"] = json.loads(args.datasets)
    settings["recovery_of"] = str(args.previous.resolve())
    if args.model == "BM25+CE":
        settings.update(batch_size=8, query_batch_size=8)
    if args.model == "ColBERT":
        settings["rank_batch_size"] = 256
        settings["reuse_index_from"] = {
            name: str((args.previous / name).resolve()) for name in settings["datasets"]
            if (args.previous / name / "indexes/beir-index").is_dir()
        }
    # GenQ training batch/loss are preserved. Generation chunks from the failed
    # attempt are reused; another GenQ process must finish before invoking this.
    return run(settings, entrypoint)


if __name__ == "__main__":
    raise SystemExit(main())
