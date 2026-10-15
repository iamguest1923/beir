"""Evaluate BAAI/bge-m3 dense embeddings on nine local BEIR datasets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
from _shared.multilingual_dense import cli, default_settings

# Same corpus/query/qrels -> retrieval -> metrics/ranking flow as runs/ANCE.
SETTINGS = default_settings("BGE-M3")
DATASETS = SETTINGS["datasets"]
MODEL_NAME = SETTINGS["model"]

if __name__ == "__main__":
    raise SystemExit(cli(SETTINGS, __file__))
