"""Evaluate multilingual E5 with query: and passage: prefixes on nine datasets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
from _shared.multilingual_dense import cli, default_settings

SETTINGS = default_settings("Multilingual-E5-Large")
DATASETS = SETTINGS["datasets"]
MODEL_NAME = SETTINGS["model"]

if __name__ == "__main__":
    raise SystemExit(cli(SETTINGS, __file__))
