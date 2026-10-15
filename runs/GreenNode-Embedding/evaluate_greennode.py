"""Evaluate the exact GreenNode checkpoint; an approved HF account may be required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
from _shared.multilingual_dense import cli, default_settings

SETTINGS = default_settings("GreenNode-Embedding")
DATASETS = SETTINGS["datasets"]
MODEL_NAME = SETTINGS["model"]

if __name__ == "__main__":
    raise SystemExit(cli(SETTINGS, __file__))
