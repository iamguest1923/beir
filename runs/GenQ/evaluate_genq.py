"""Evaluate GenQ on six public BEIR datasets. Run this file directly."""
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
MAX_LENGTH = 512
K_VALUES = [1, 3, 5, 10, 100, 1000]
SEED = 42
REUSE_CACHE = True
KEEP_INDEX = False
CPU_THREADS = 4
EXISTING_DATA_DIRS = []  # ANCE, BM25 and example data folders are searched automatically.
MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"
GENERATION_MODEL = "BeIR/query-gen-msmarco-t5-base-v1"
QUESTIONS_PER_PASSAGE = 5
MAX_TRAINING_DOCUMENTS = 100000
TRAIN_BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-5
# Paper does not fully specify every training hyperparameter; these defaults are explicit.
EXTRA = {
    "model": MODEL, "generation_model": GENERATION_MODEL,
    "questions_per_passage": QUESTIONS_PER_PASSAGE, "generation_batch_size": 8,
    "generation_max_length": 64, "generation_top_k": 25, "generation_top_p": 0.95,
    "max_training_documents": MAX_TRAINING_DOCUMENTS, "train_batch_size": TRAIN_BATCH_SIZE,
    "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "train_steps_per_epoch": None,
}

SETTINGS = {
    "method": 'GenQ', "datasets": DATASETS, "device": DEVICE,
    "batch_size": BATCH_SIZE, "corpus_chunk_size": CORPUS_CHUNK_SIZE,
    "max_length": MAX_LENGTH, "k_values": K_VALUES, "seed": SEED,
    "reuse_cache": REUSE_CACHE, "keep_index": KEEP_INDEX,
    "cpu_threads": CPU_THREADS, "query_batch_size": 32,
    "existing_data_dirs": EXISTING_DATA_DIRS, **EXTRA,
}

if __name__ == "__main__":
    raise SystemExit(run(SETTINGS, __file__))
