"""Evaluate DeepCT on six public BEIR datasets. Run this file directly."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
from _shared.common import run

# SETTINGS: edit here, like runs/ANCE/evaluate_ance.py. No required CLI arguments.
DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid"]
DEVICE = None  # Auto-select CUDA/CPU; external DeepCT/ColBERT use their separate runtime.
BATCH_SIZE = 16
CORPUS_CHUNK_SIZE = 10000
MAX_LENGTH = 512
K_VALUES = [1, 3, 5, 10, 100, 1000]
SEED = 42
REUSE_CACHE = True
KEEP_INDEX = False
CPU_THREADS = 4
EXISTING_DATA_DIRS = []  # ANCE, BM25 and example data folders are searched automatically.
EXTERNAL_PYTHON = None  # Default: runs/DeepCT/environment/bin/python
EXTERNAL_REPO = None    # Default: runs/DeepCT/vendor (pinned by setup_external.py)
BACKEND = "pytorch-converted"  # Original DeepCT weights converted by macavaney; "tensorflow-original" is also available.
MODEL = "macavaney/deepct"
SCALING = 100
SMOOTHING = "none"
KEEP_ALL_TERMS = True
EXTRA = {
    "backend": BACKEND, "model": MODEL,
    "external_python": EXTERNAL_PYTHON, "external_repo": EXTERNAL_REPO,
    "scaling": SCALING, "smoothing": SMOOTHING, "keep_all_terms": KEEP_ALL_TERMS,
    "bm25_k1": 0.9, "bm25_b": 0.4, "bm25_fields": {"contents": 1.0},
}

SETTINGS = {
    "method": 'DeepCT', "datasets": DATASETS, "device": DEVICE,
    "batch_size": BATCH_SIZE, "corpus_chunk_size": CORPUS_CHUNK_SIZE,
    "max_length": MAX_LENGTH, "k_values": K_VALUES, "seed": SEED,
    "reuse_cache": REUSE_CACHE, "keep_index": KEEP_INDEX,
    "cpu_threads": CPU_THREADS, "query_batch_size": 32,
    "existing_data_dirs": EXISTING_DATA_DIRS, **EXTRA,
}

if __name__ == "__main__":
    raise SystemExit(run(SETTINGS, __file__))
