"""Prepare isolated, native DeepCT or ColBERT v1 runtimes (no Docker)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS.parent))
sys.path.insert(0, str(RUNS))
from _shared.external import REPOSITORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["deepct", "colbert"])
    parser.add_argument("--source-only", action="store_true", help="Clone pinned source without installing an environment")
    args = parser.parse_args()
    method = {"deepct": "DeepCT", "colbert": "ColBERT"}[args.method]
    if not (3, 9) <= sys.version_info[:2] <= (3, 11):
        parser.error("Run setup from Python 3.9–3.11; Python 3.9 is recommended for ColBERT's torch 1.13 wheels")
    base = RUNS / method
    repository, commit = REPOSITORIES[method]
    source = base / "vendor"
    if not source.exists():
        subprocess.run(["git", "clone", repository, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Existing checkout has different commit: {actual}; refusing to overwrite")
    if args.source_only:
        print("Pinned source:", source, commit)
        return
    environment = base / "environment"
    python = environment / "bin/python"
    if not python.is_file():
        venv.create(environment, with_pip=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(base / "requirements-external.txt")], check=True)
    (environment / "beir-runtime.json").write_text(json.dumps({
        "method": method, "repository": repository, "commit": commit,
        "python": str(python), "requirements": str(base / "requirements-external.txt"),
    }, indent=2))
    print("Prepared:", python)
    print("The runner preflight will validate imports/CUDA; no model weights or datasets were downloaded by setup.")


if __name__ == "__main__":
    main()
