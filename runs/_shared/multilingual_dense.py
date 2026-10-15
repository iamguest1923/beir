"""ANCE-style, normalized dense evaluation with model-specific preprocessing."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone

from .common import (ROOT, file_sha256, prepare_dataset, reuse_completed,
                     score_and_save, signature, write_json)

LOG = logging.getLogger(__name__)
DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid",
            "uit-viquad", "vihealthqa", "zalo-legal"]
VI_DATASETS = {"uit-viquad", "vihealthqa", "zalo-legal"}
SPECS = {
    "BGE-M3": {
        "model": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "max_length": 512, "native_max_length": 8192, "query_prefix": "", "passage_prefix": "",
        "word_segment": False, "batch_size": 2, "pooling": "cls", "dimension": 1024,
        "retrieval_mode": "dense_only",
    },
    "Multilingual-E5-Large": {
        "model": "intfloat/multilingual-e5-large", "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        "max_length": 512, "native_max_length": 512, "query_prefix": "query: ", "passage_prefix": "passage: ",
        "word_segment": False, "batch_size": 2, "pooling": "mean", "dimension": 1024,
        "retrieval_mode": "dense_only",
    },
    "GreenNode-Embedding": {
        "model": "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1",
        "revision": "1d3dddb3862292dab4bd3eddf0664c0335ad5843",
        "max_length": 512, "native_max_length": 8192, "query_prefix": "", "passage_prefix": "",
        "word_segment": False, "batch_size": 2, "pooling": "cls", "dimension": 1024,
        "retrieval_mode": "dense_only",
    },
    "Vietnamese-BiEncoder": {
        "model": "bkai-foundation-models/vietnamese-bi-encoder",
        "revision": "84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0",
        "max_length": 256, "native_max_length": 256, "query_prefix": "", "passage_prefix": "",
        "word_segment": True, "batch_size": 8, "pooling": "mean", "dimension": 768,
        "retrieval_mode": "dense_only",
        "training_overlap_note": "Model card includes 80% of Zalo Legal 2021 training data. "
                                 "Local Zalo train/test have 24 overlapping query IDs; do not claim strict zero-shot.",
    },
}


class AccessRequired(RuntimeError):
    pass


def is_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or type(exc).__name__ == "OutOfMemoryError")


def fetch_snapshot(spec, offline=False):
    """Download only ST inference files, preferring one safe weight format."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        if offline:
            return Path(snapshot_download(spec["model"], revision=spec["revision"], local_files_only=True))
        files = HfApi().list_repo_files(spec["model"], revision=spec["revision"])
        weights = ["model.safetensors"] if "model.safetensors" in files else ["pytorch_model.bin"]
        if not any(f in files for f in weights):
            raise ValueError("Expected a supported, unsharded Sentence Transformers checkpoint")
        chosen = [f for f in files if (f in weights or f.endswith((".json", ".txt", ".model", ".bpe")))
                  and not f.startswith(("onnx/", "openvino/"))]
        return Path(snapshot_download(spec["model"], revision=spec["revision"],
                                      allow_patterns=chosen, max_workers=2))
    except GatedRepoError as exc:
        raise AccessRequired(f"Access required for {spec['model']}. Request access, then run hf auth login "
                             "on this machine using the approved account; never put tokens in source/logs.") from exc
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) in (401, 403):
            raise AccessRequired(f"Hugging Face denied access to {spec['model']}; authenticate an approved account.") from exc
        raise


def prepare_texts(texts, spec, role, dataset, segmenter=None):
    if role not in {"query", "passage"}:
        raise ValueError("role must be query or passage")
    if spec["word_segment"] and dataset in VI_DATASETS:
        if segmenter is None:
            from pyvi import ViTokenizer
            segmenter = ViTokenizer.tokenize
        texts = [segmenter(text) if text.strip() else text for text in texts]
    return [spec[role + "_prefix"] + text for text in texts]


class NormalizedEncoder:
    def __init__(self, model, spec, dataset, batch_size, event_file, segmenter=None):
        self.model, self.spec, self.dataset = model, spec, dataset
        self.effective_batch_size = batch_size
        self.event_file, self.segmenter = event_file, segmenter
        self.recoveries = []

    def _encode(self, texts, role, batch_size, **kwargs):
        texts = prepare_texts(texts, self.spec, role, self.dataset, self.segmenter)
        while True:
            batch = min(batch_size, self.effective_batch_size)
            try:
                # Prefixes are applied explicitly once. Suppress any ST default prompt.
                return self.model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                         prompt="", **kwargs)
            except Exception as exc:
                if not is_oom(exc) or batch == 1:
                    raise
                import torch
                self.effective_batch_size = max(1, batch // 2)
                self.recoveries.append({"role": role, "old_batch": batch,
                                       "new_batch": self.effective_batch_size, "reason": "CUDA OOM"})
                write_json(self.event_file, self.recoveries)
                LOG.warning("CUDA OOM: reducing encoding batch %d -> %d; model/length/data stay fixed",
                            batch, self.effective_batch_size)
                gc.collect()
                torch.cuda.empty_cache()

    def encode_queries(self, queries, batch_size=2, **kwargs):
        return self._encode(queries, "query", batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size=2, **kwargs):
        from beir.retrieval.models.util import extract_corpus_sentences
        texts = extract_corpus_sentences(corpus, sep=" ")
        return self._encode(texts, "passage", batch_size, **kwargs)


def checked_inputs(path):
    return {f: file_sha256(path / f) for f in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]}


def model_metadata(spec, dataset, snapshot, encoder):
    return {**spec, "snapshot": str(snapshot), "resolved_revision": snapshot.name,
            "precision": "float32", "score_function": "dot_on_l2_normalized_vectors",
            "title_text_separator": " ", "default_prompt_disabled": True,
            "word_segment_applied": spec["word_segment"] and dataset in VI_DATASETS,
            "word_segmenter": "pyvi.ViTokenizer.tokenize" if spec["word_segment"] and dataset in VI_DATASETS else None,
            "effective_batch_size": encoder.effective_batch_size,
            "oom_recoveries": encoder.recoveries}


def run(settings, entrypoint, offline=False, prefetch=False, check_model=False):
    import torch
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch
    from sentence_transformers import SentenceTransformer

    spec = dict(settings)
    datasets = spec.pop("datasets")
    if not datasets or set(datasets) - set(DATASETS) or len(set(datasets)) != len(datasets):
        raise ValueError("Choose unique dataset names from the nine supported datasets")
    torch.set_num_threads(spec.get("cpu_threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    method = spec["method"]
    base = Path(entrypoint).resolve().parent
    if prefetch or check_model:
        snapshot = fetch_snapshot(spec, offline)
        if check_model:
            import numpy as np
            model = SentenceTransformer(str(snapshot), device="cpu", trust_remote_code=False)
            model.max_seq_length = spec["max_length"]
            model.float()
            encoder = NormalizedEncoder(model, spec, "uit-viquad", 2, base / "model_checks/oom.json")
            emb = encoder.encode_queries(["Thủ đô của Việt Nam là gì?", "What is information retrieval?"],
                                         batch_size=2, convert_to_numpy=True, show_progress_bar=False)
            if emb.shape != (2, spec["dimension"]) or not np.isfinite(emb).all():
                raise ValueError("Invalid model output shape or nonfinite embeddings")
            if not np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4):
                raise ValueError("Embeddings are not normalized")
            write_json(base / "model_checks/check.json", {"status": "SUCCESS", "benchmark_result": False,
                "revision": snapshot.name, "shape": list(emb.shape), "norms": np.linalg.norm(emb, axis=1).tolist()})
        print(json.dumps({"model": spec["model"], "snapshot": str(snapshot), "check_model": check_model}), flush=True)
        return 0

    run_id = os.environ.get("BEIR_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    if Path(run_id).name != run_id:
        raise ValueError("Invalid run ID")
    session = base / "results" / run_id
    session.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(session / "run.log", encoding="utf-8")], force=True)
    versions = {}
    for name in ["torch", "transformers", "sentence-transformers", "huggingface-hub", "pyvi", "pytrec-eval-terrier"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {"runner": file_sha256(__file__), "entrypoint": file_sha256(entrypoint),
               "common": file_sha256(Path(__file__).with_name("common.py")),
               "exact_search": file_sha256(ROOT / "beir/retrieval/search/dense/exact_search.py")}
    # Batch size is execution detail; reduce it on OOM without changing the model or token limit.
    semantic = {k: v for k, v in spec.items() if k not in {"batch_size", "cpu_threads", "device"}}
    config = {**spec, "datasets": datasets, "run_id": run_id, "split": "test", "python": sys.version,
              "versions": versions, "source_sha256": sources, "normalization": "L2",
              "cache_signature": signature({"settings": semantic, "source": sources, "versions": versions})}
    write_json(session / "config.json", config)
    shutil.copyfile(__file__, session / "multilingual_dense.py")
    shutil.copyfile(entrypoint, session / Path(entrypoint).name)
    model = None
    snapshot = None
    failures = 0
    access_required = False
    fields = ["dataset", "split", "model", "status", "NDCG@10", "Recall@100", "error"]
    with (session / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in datasets:
            output = session / name
            output.mkdir()
            row = {"dataset": name, "split": "test", "model": method, "status": "FAILED"}
            write_json(output / "config.json", {**config, "dataset": name})
            started = time.monotonic()
            try:
                path = prepare_dataset(name, base)
                hashes = checked_inputs(path)
                cached = reuse_completed(base, output, name, config["cache_signature"], hashes)
                if cached is not None:
                    row.update(status="SUCCESS", **{"NDCG@10": cached["ndcg"]["NDCG@10"],
                                                    "Recall@100": cached["recall"]["Recall@100"]})
                    continue
                if model is None:
                    LOG.info("Loading %s at immutable revision %s", spec["model"], spec["revision"])
                    snapshot = fetch_snapshot(spec, offline)
                    model = SentenceTransformer(str(snapshot), device=spec.get("device"), trust_remote_code=False)
                    model.max_seq_length = spec["max_length"]
                    model.float()
                    write_json(session / "model.json", {"snapshot": str(snapshot), "revision": snapshot.name,
                                                         "device": str(model.device), "precision": "float32"})
                LOG.info("Evaluating %s on %s", method, name)
                corpus, queries, qrels = GenericDataLoader(str(path)).load(split="test")
                if not corpus or not queries or not qrels:
                    raise ValueError("Empty corpus, query split or qrels")
                encoder = NormalizedEncoder(model, spec, name, spec["batch_size"], output / "recovery.json")
                search = DenseRetrievalExactSearch(encoder, batch_size=spec["batch_size"],
                                                   corpus_chunk_size=spec["corpus_chunk_size"])
                results = EvaluateRetrieval(search, k_values=spec["k_values"], score_function="dot").retrieve(corpus, queries)
                metrics = score_and_save(output, results, corpus, queries, qrels, spec["k_values"])
                write_json(output / "model.json", model_metadata(spec, name, snapshot, encoder))
                write_json(output / "data.json", {"path": str(path), "sha256": hashes,
                    "documents": len(corpus), "queries": len(queries), "elapsed_seconds": time.monotonic()-started,
                    "artifact_sha256": {f: file_sha256(output / f) for f in ["metrics.json", "ranking.run.trec"]}})
                row.update(status="SUCCESS", **{"NDCG@10": metrics["ndcg"]["NDCG@10"],
                                                "Recall@100": metrics["recall"]["Recall@100"]})
                del results, corpus, queries, qrels
            except AccessRequired as exc:
                access_required = True
                row.update(status="WAITING_ACCESS", error=str(exc))
                LOG.error("%s", exc)
            except Exception as exc:
                failures += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
                LOG.exception("%s failed on %s", method, name)
            finally:
                write_json(output / "status.json", row)
                writer.writerow(row)
                stream.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if access_required:
                break
    status = "WAITING_ACCESS" if access_required else "SUCCESS" if not failures else "PARTIAL_FAILURE"
    write_json(session / "status.json", {"status": status, "failed": failures, "datasets": datasets})
    return 3 if access_required else int(bool(failures))


def cli(settings, entrypoint):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    settings = dict(settings)
    if args.datasets:
        settings["datasets"] = args.datasets
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        settings["batch_size"] = args.batch_size
    if args.device:
        settings["device"] = args.device
    try:
        return run(settings, entrypoint, args.offline, args.prefetch, args.check_model)
    except AccessRequired as exc:
        print(str(exc), file=sys.stderr)
        return 3


def default_settings(method):
    return {**SPECS[method], "method": method, "datasets": DATASETS.copy(), "device": "cuda",
            "cpu_threads": 2, "corpus_chunk_size": 10000, "k_values": [1, 3, 5, 10, 100, 1000]}
