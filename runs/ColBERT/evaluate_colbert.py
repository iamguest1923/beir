"""Evaluate ColBERT on six public BEIR datasets. Run this file directly."""
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
BATCH_SIZE = 32
CORPUS_CHUNK_SIZE = 10000
MAX_LENGTH = 300
K_VALUES = [1, 3, 5, 10, 100, 1000]
SEED = 42
REUSE_CACHE = True
KEEP_INDEX = False
CPU_THREADS = 4
EXISTING_DATA_DIRS = []  # ANCE, BM25 and example data folders are searched automatically.
EXTERNAL_PYTHON = None  # Default: runs/ColBERT/environment/bin/python
EXTERNAL_REPO = None    # Default: runs/ColBERT/vendor
# IVF partitions are explicit starting values, not claimed to be the authors' exact values.
PARTITIONS = {"nfcorpus": 96, "scifact": 96, "arguana": 128,
              "scidocs": 256, "fiqa": 512, "trec-covid": 1024}
EXTRA = {
    "external_python": EXTERNAL_PYTHON, "external_repo": EXTERNAL_REPO,
    "query_max_length": 32, "partitions": PARTITIONS,
    "nprobe": 32, "faiss_depth": 100, "return_depth": 100, "index_sample": 0.3,
}

SETTINGS = {
    "method": 'ColBERT', "datasets": DATASETS, "device": DEVICE,
    "batch_size": BATCH_SIZE, "corpus_chunk_size": CORPUS_CHUNK_SIZE,
    "max_length": MAX_LENGTH, "k_values": K_VALUES, "seed": SEED,
    "reuse_cache": REUSE_CACHE, "keep_index": KEEP_INDEX,
    "cpu_threads": CPU_THREADS, "query_batch_size": 32,
    "existing_data_dirs": EXISTING_DATA_DIRS, **EXTRA,
}

if __name__ == "__main__":
    raise SystemExit(run(SETTINGS, __file__))
