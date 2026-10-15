"""Evaluate Anserini BM25 on public BEIR datasets. Edit SETTINGS below, then run this file."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import shutil
import sys
import time
import uuid
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution without accidentally importing a different BEIR checkout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.base import BaseSearch

# SETTINGS: keep configuration in this file, like the upstream examples.
PUBLIC_DATASETS = [
    "nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid",
    "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity",
    "nq", "hotpotqa", "fever", "climate-fever", "msmarco",
]
DATASETS = PUBLIC_DATASETS.copy()  # For a short run: ["nfcorpus"]
MODEL_NAME = "anserini-bm25-multifield"
PYSERINI_VERSION = "0.12.0"  # Compatible with Python 3.9 / Java 11.
JAVA_HEAP = "4g"
K1 = 0.9
B = 0.4
FIELDS = {"contents": 1.0, "title": 1.0}
BATCH_SIZE = 32  # Queries per batch; BM25 uses CPU, not GPU.
THREADS = 4
INDEX_TIMEOUT = 86400
KEEP_INDEX = False  # Delete only this run's temporary index after successful search.
K_VALUES = [1, 3, 5, 10, 100, 1000]
SPLITS = {"msmarco": "dev"}  # MS MARCO dev: 6,980 queries; test: 43.
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets"
RESULTS_DIR = BASE / "results"
EXISTING_DATA_DIRS = [
    ROOT / "runs/ANCE/datasets",
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


def load_backend():
    """Initialize the local JVM before datasets are loaded or downloaded."""
    version = importlib.metadata.version("pyserini")
    if version != PYSERINI_VERSION:
        raise RuntimeError(f"Expected pyserini=={PYSERINI_VERSION}, found {version}")
    if shutil.which("java") is None:
        raise RuntimeError("Java 11 is required; set JAVA_HOME and PATH before running")
    os.environ.setdefault("JAVA_TOOL_OPTIONS", f"-Xmx{JAVA_HEAP}")
    from pyserini.search import SimpleSearcher
    return SimpleSearcher


class AnseriniBM25Search(BaseSearch):
    """Build and search a local Anserini index through Pyserini."""

    def __init__(self, output):
        self.output = output

    def encode(self, *args, **kwargs):
        raise NotImplementedError("BM25 indexes terms, not dense embeddings; use retrieve().")

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError("Use retrieve() with the corpus and queries.")

    def search(self, corpus, queries, top_k, score_function=None, **kwargs):
        searcher_class = load_backend()
        folder = self.output / "index_input"
        folder.mkdir()
        index = self.output / "index"
        if index.exists():
            raise ValueError("Index already exists; use a new output directory")
        searcher = None
        success = False
        try:
            with (folder / "corpus.jsonl").open("w", encoding="utf-8") as stream:
                for doc_id, doc in corpus.items():
                    stream.write(json.dumps({
                        "id": doc_id, "title": doc.get("title") or "",
                        "contents": doc.get("text") or "",
                    }) + "\n")
            command = [
                sys.executable, "-m", "pyserini.index", "-collection", "JsonCollection",
                "-generator", "DefaultLuceneDocumentGenerator", "-threads", str(THREADS),
                "-input", str(folder), "-index", str(index),
                "-storePositions", "-storeDocvectors",
            ]
            empty_count = sum(not (doc.get("text") or "").strip() for doc in corpus.values())
            if empty_count:
                # Anserini 0.12's default generator rejects empty contents before
                # inspecting title. Retain original fields, including truly empty
                # documents, without adding artificial searchable terms.
                import pyserini
                jar = Path(pyserini.__file__).parent / "resources/jars/anserini-0.12.0-fatjar.jar"
                source = BASE / "java/BeirMultifieldGenerator.java"
                classes = self.output / "index_generator"
                classes.mkdir()
                subprocess.run(["javac", "--release", "11", "-cp", str(jar), "-d", str(classes), str(source)],
                               check=True, capture_output=True, text=True)
                shutil.copyfile(source, classes / source.name)
                command = ["java", "-cp", os.pathsep.join([str(jar), str(classes)]),
                           "io.anserini.index.IndexCollection", *command[3:]]
                command[command.index("-generator") + 1] = "BeirMultifieldGenerator"
                write_json(self.output / "index_generator.json", {
                    "empty_contents_retained": empty_count,
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "policy": "Keep original empty contents and title fields; no synthetic terms",
                })
            write_json(self.output / "index_command.json", command)
            LOGGER.info("Indexing %d documents; log: %s", len(corpus), self.output / "indexing.log")
            with (self.output / "indexing.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=INDEX_TIMEOUT)
            searcher = searcher_class(str(index))
            if searcher.num_docs != len(corpus):
                raise ValueError(f"Indexed document count differs: {searcher.num_docs} vs {len(corpus)}")
            searcher.set_bm25(k1=K1, b=B)
            results = {}
            qids = list(queries)
            for start in range(0, len(qids), BATCH_SIZE):
                batch = qids[start:start + BATCH_SIZE]
                rows = searcher.batch_search(
                    queries=[queries[qid] for qid in batch], qids=batch, k=top_k + 1,
                    threads=THREADS, fields=FIELDS,
                )
                if set(rows) != set(batch):
                    raise ValueError("Searcher returned incomplete query coverage")
                for qid in batch:
                    ranked = sorted(rows[qid], key=lambda hit: (hit.score, hit.docid), reverse=True)
                    results[qid] = dict([
                        (hit.docid, float(hit.score)) for hit in ranked if hit.docid != qid
                    ][:top_k])
                LOGGER.info("Searched queries %d/%d", min(start + BATCH_SIZE, len(qids)), len(qids))
            success = True
            return results
        finally:
            if searcher is not None:
                searcher.close()
            shutil.rmtree(folder, ignore_errors=True)
            if success and not KEEP_INDEX:
                shutil.rmtree(index)


def evaluate_folder(data_path, output, split):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split=split)
    if not corpus or not queries or not qrels:
        raise ValueError("Empty corpus, queries or qrels")
    if data_path.name == "msmarco" and split == "dev" and len(queries) != 6980:
        raise ValueError(f"Expected 6,980 MS MARCO dev queries, got {len(queries)}")
    evaluator = EvaluateRetrieval(AnseriniBM25Search(output), k_values=K_VALUES)
    results = evaluator.retrieve(corpus, queries)
    ndcg, mean_ap, recall, precision = evaluator.evaluate(qrels, results, K_VALUES)
    mrr = evaluator.evaluate_custom(qrels, results, K_VALUES, metric="mrr")
    util.save_results(str(output / "metrics.json"), ndcg, mean_ap, recall, precision, mrr)
    util.save_runfile(str(output / "ranking.run.trec"), results)
    write_json(output / "data.json", {
        "path": str(data_path), "split": split, "documents": len(corpus),
        "queries": len(queries), "elapsed_seconds": time.perf_counter() - started,
    })
    return dict(ndcg=ndcg, map=mean_ap, recall=recall, precision=precision, mrr=mrr)


def main():
    unknown = set(DATASETS) - set(PUBLIC_DATASETS)
    if unknown:
        raise ValueError(f"Unsupported dataset names: {sorted(unknown)}")
    if not DATASETS or min(BATCH_SIZE, THREADS, *K_VALUES) < 1:
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
    for package in ("beir", "pyserini", "pyjnius", "pytrec-eval-terrier"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = dict(
        run_id=run_id, model=MODEL_NAME, datasets=DATASETS, splits=SPLITS,
        k1=K1, b=B, fields=FIELDS, batch_size=BATCH_SIZE, threads=THREADS,
        k_values=K_VALUES, python=sys.version, versions=versions, beir_source=str(ROOT),
        backend="local-pyserini", java_heap=JAVA_HEAP, keep_index=KEEP_INDEX,
    )
    write_json(session / "config.json", config)
    shutil.copyfile(__file__, session / "evaluate_bm25.py")
    LOGGER.info("Session: %s", session)
    try:
        load_backend()
        java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
        config["java_version"] = java.stderr.strip() or java.stdout.strip()
        config["java_tool_options"] = os.environ.get("JAVA_TOOL_OPTIONS")
        write_json(session / "config.json", config)
    except Exception as exc:
        LOGGER.error("Local BM25 backend unavailable: %s. See runs/BM25/README.md", exc)
        write_json(session / "status.json", {"status": "FAILED", "error": str(exc)})
        return 1
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
                    scores = [evaluate_folder(path / name, output / name, split) for name in SUBSETS]
                    metrics = {
                        group: {key: sum(item[group][key] for item in scores) / len(scores)
                                for key in scores[0][group]}
                        for group in scores[0]
                    }
                    write_json(output / "metrics.json", metrics)
                    write_json(output / "aggregation.json", {"method": "unweighted mean over 12 subsets", "subsets": SUBSETS})
                else:
                    metrics = evaluate_folder(path, output, split)
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
    LOGGER.info("Finished: %d succeeded, %d failed. Summary: %s", len(DATASETS) - failures, failures, session / "summary.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
