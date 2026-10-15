"""Evaluate ANCE on public BEIR datasets. Edit SETTINGS below, then run this file."""
from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import logging
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution without accidentally importing a different BEIR checkout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch

# SETTINGS: keep configuration in this file, like the upstream examples.
PUBLIC_DATASETS = [
    "nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid",
    "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity",
    "nq", "hotpotqa", "fever", "climate-fever", "msmarco",
]
DATASETS = PUBLIC_DATASETS.copy()  # For a short run: ["nfcorpus"]
MODEL_NAME = "sentence-transformers/msmarco-roberta-base-ance-firstp"
MODEL_REVISION = None  # Optional Hugging Face commit SHA for a fixed checkpoint.
DEVICE = None  # Auto-select; alternatively "cuda" or "cpu".
BATCH_SIZE = 16
CORPUS_CHUNK_SIZE = 10000
MAX_LENGTH = 512
K_VALUES = [1, 3, 5, 10, 100, 1000]
SPLITS = {"msmarco": "dev"}  # MS MARCO dev: 6,980 queries; test: 43.
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets"
RESULTS_DIR = BASE / "results"
EXISTING_DATA_DIRS = [
    ROOT / "examples/retrieval/evaluation/dense/datasets",
    ROOT / "examples/retrieval/evaluation/lexical/datasets",
    ROOT / "examples/retrieval/evaluation/sparse/datasets",
    ROOT / "examples/retrieval/evaluation/reranking/datasets",
]
DELETE_DOWNLOADED_ZIP = True  # After successful extraction; never delete reused data.
SUBSETS = [
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
]
LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def valid_folder(path, dataset, split):
    folders = [path / name for name in SUBSETS] if dataset == "cqadupstack" else [path]
    return all(
        all((folder / name).is_file() for name in (
            "corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"
        )) for folder in folders
    )


def prepare_dataset(dataset, split):
    for parent in [DATA_DIR, *EXISTING_DATA_DIRS]:
        candidate = parent / dataset
        if valid_folder(candidate, dataset, split):
            LOGGER.info("Reusing dataset: %s", candidate)
            return candidate

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / f"{dataset}.zip"
    if archive.exists() and not zipfile.is_zipfile(archive):
        bad = archive.with_name(f"{archive.name}.invalid-{uuid.uuid4().hex[:8]}")
        archive.rename(bad)
        LOGGER.warning("Moved invalid ZIP to %s", bad)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        util.download_url(url, str(partial))
        if not zipfile.is_zipfile(partial):
            raise ValueError(f"Download is not a complete ZIP: {partial}")
        partial.replace(archive)

    staging = DATA_DIR / f".extract-{dataset}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive) as handle:
            required = sum(info.file_size for info in handle.infolist())
            if shutil.disk_usage(DATA_DIR).free < required + 1024**3:
                raise OSError(f"Need {required / 1024**3:.1f} GiB plus 1 GiB free to extract {dataset}")
            for info in handle.infolist():
                if not (staging / info.filename).resolve().is_relative_to(staging.resolve()):
                    raise ValueError("ZIP contains a path outside the extraction directory")
        util.unzip(str(archive), str(staging))
        extracted = staging / dataset
        if not valid_folder(extracted, dataset, split):
            raise ValueError(f"Dataset is missing required BEIR files for split={split}")
        destination = DATA_DIR / dataset
        if destination.exists():
            destination.rename(DATA_DIR / f"{dataset}.incomplete-{uuid.uuid4().hex[:8]}")
        shutil.move(str(extracted), str(destination))
        if DELETE_DOWNLOADED_ZIP:
            archive.unlink()
        return destination
    finally:
        shutil.rmtree(staging)


def evaluate_folder(encoder, data_path, output, split):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split=split)
    if not corpus or not queries or not qrels:
        raise ValueError("Empty corpus, queries or qrels")
    if data_path.name == "msmarco" and split == "dev" and len(queries) != 6980:
        raise ValueError(f"Expected 6,980 MS MARCO dev queries, got {len(queries)}")
    # SparseSearch is not used: this is exact dot-product search with full vectors.
    search = DenseRetrievalExactSearch(
        encoder, batch_size=BATCH_SIZE, corpus_chunk_size=CORPUS_CHUNK_SIZE
    )
    evaluator = EvaluateRetrieval(search, k_values=K_VALUES, score_function="dot")
    results = evaluator.retrieve(corpus, queries)
    ndcg, mean_ap, recall, precision = evaluator.evaluate(qrels, results, K_VALUES)
    mrr = evaluator.evaluate_custom(qrels, results, K_VALUES, metric="mrr")
    util.save_results(str(output / "metrics.json"), ndcg, mean_ap, recall, precision, mrr)
    util.save_runfile(str(output / "ranking.run.trec"), results)
    metrics = dict(ndcg=ndcg, map=mean_ap, recall=recall, precision=precision, mrr=mrr)
    write_json(output / "data.json", {
        "path": str(data_path), "split": split, "documents": len(corpus),
        "queries": len(queries), "elapsed_seconds": time.perf_counter() - started,
    })
    return metrics


def main():
    unknown = set(DATASETS) - set(PUBLIC_DATASETS)
    if unknown:
        raise ValueError(f"Unsupported dataset names: {sorted(unknown)}")
    if not DATASETS or min(BATCH_SIZE, CORPUS_CHUNK_SIZE, *K_VALUES) < 1:
        raise ValueError("Dataset list must not be empty and sizes must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    session = RESULTS_DIR / run_id
    session.mkdir(parents=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO,
        handlers=[LoggingHandler(), logging.FileHandler(session / "run.log", encoding="utf-8")],
        force=True,
    )
    versions = {}
    for package in ("beir", "torch", "transformers", "sentence-transformers", "pytrec-eval-terrier"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = dict(
        run_id=run_id, model=MODEL_NAME, requested_revision=MODEL_REVISION,
        datasets=DATASETS, splits=SPLITS, score_function="dot", max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE, corpus_chunk_size=CORPUS_CHUNK_SIZE, k_values=K_VALUES,
        device=DEVICE, python=sys.version, versions=versions, beir_source=str(ROOT),
    )
    write_json(session / "config.json", config)
    shutil.copyfile(__file__, session / "evaluate_ance.py")
    kwargs = {"revision": MODEL_REVISION} if MODEL_REVISION else {}
    LOGGER.info("Session: %s", session)
    encoder = SentenceBERT(MODEL_NAME, max_length=MAX_LENGTH, device=DEVICE, **kwargs)
    config["device"] = str(encoder.q_model.device)
    config["resolved_revision"] = getattr(encoder.q_model[0].auto_model.config, "_commit_hash", None)
    write_json(session / "config.json", config)
    failures = 0
    with (session / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dataset", "split", "model", "status", "NDCG@10", "Recall@100", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for dataset in DATASETS:
            split = SPLITS.get(dataset, "test")
            output = session / dataset
            output.mkdir()
            handler = logging.FileHandler(output / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(handler)
            row = dict(dataset=dataset, split=split, model=MODEL_NAME, status="FAILED")
            write_json(output / "config.json", {**config, "dataset": dataset, "split": split})
            try:
                path = prepare_dataset(dataset, split)
                if dataset == "cqadupstack":
                    scores = [evaluate_folder(encoder, path / name, output / name, split) for name in SUBSETS]
                    metrics = {
                        group: {key: sum(item[group][key] for item in scores) / len(scores)
                                for key in scores[0][group]}
                        for group in scores[0]
                    }
                    write_json(output / "metrics.json", metrics)
                    write_json(output / "aggregation.json", {"method": "unweighted mean over 12 subsets", "subsets": SUBSETS})
                else:
                    metrics = evaluate_folder(encoder, path, output, split)
                row.update(status="SUCCESS", **{
                    "NDCG@10": metrics["ndcg"]["NDCG@10"],
                    "Recall@100": metrics["recall"]["Recall@100"],
                })
            except Exception as exc:
                LOGGER.exception("Failed dataset %s", dataset)
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
            finally:
                write_json(output / "status.json", row)
                writer.writerow(row)
                stream.flush()
                logging.getLogger().removeHandler(handler)
                handler.close()
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    LOGGER.info("Finished: %d succeeded, %d failed. Summary: %s", len(DATASETS) - failures, failures, session / "summary.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
