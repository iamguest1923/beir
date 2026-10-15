from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid"]
NEW_DATA = ROOT / "new/data/datasets"
LOG = logging.getLogger(__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_dataset(path):
    return all((path / name).is_file() for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"))


def local_datasets():
    return sorted(path.name for path in NEW_DATA.iterdir() if path.is_dir() and valid_dataset(path)) if NEW_DATA.is_dir() else []


def prepare_dataset(name, base, extra_dirs=()):
    parents = [NEW_DATA, base / "datasets", *map(Path, extra_dirs), RUNS / "ANCE/datasets", RUNS / "BM25/datasets"]
    parents += sorted(RUNS.glob("*/datasets"))
    parents += sorted((ROOT / "examples/retrieval/evaluation").glob("*/datasets"))
    for parent in dict.fromkeys(parents):
        candidate = parent / name
        if valid_dataset(candidate):
            LOG.info("Reusing dataset: %s", candidate)
            return candidate
    if name not in DATASETS:
        raise FileNotFoundError(f"Local benchmark {name} is missing valid BEIR test files under {NEW_DATA}")
    target = base / "datasets"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{name}.zip"
    if archive.exists() and not zipfile.is_zipfile(archive):
        archive.rename(archive.with_name(archive.name + ".invalid-" + uuid.uuid4().hex[:8]))
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        util.download_url(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
            str(partial),
        )
        if not zipfile.is_zipfile(partial):
            raise ValueError(f"Incomplete or invalid dataset ZIP: {partial}")
        partial.replace(archive)
    temporary = target / (".extract-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        with zipfile.ZipFile(archive) as source:
            required = sum(info.file_size for info in source.infolist())
            if shutil.disk_usage(target).free < required + 1024**3:
                raise OSError(f"Need {required / 1024**3:.1f} GiB plus 1 GiB free to extract {name}")
            for info in source.infolist():
                if not (temporary / info.filename).resolve().is_relative_to(temporary.resolve()):
                    raise ValueError("Unsafe ZIP path")
        util.unzip(str(archive), str(temporary))
        extracted = temporary / name
        if not valid_dataset(extracted):
            raise ValueError(f"Missing BEIR test files in {archive}")
        destination = target / name
        if destination.exists():
            destination.rename(target / (name + ".incomplete-" + uuid.uuid4().hex[:8]))
        shutil.move(str(extracted), str(destination))
        archive.unlink()
        return destination
    finally:
        shutil.rmtree(temporary)


def validate_results(results, corpus, queries):
    if set(results) != set(queries):
        raise ValueError("Result query coverage does not match the evaluation queries")
    for qid, hits in results.items():
        for doc, score in hits.items():
            if doc not in corpus or not math.isfinite(float(score)):
                raise ValueError(f"Invalid result for query={qid}: {doc}, {score}")


def score_and_save(output, results, corpus, queries, qrels, k_values):
    validate_results(results, corpus, queries)
    ndcg, mean_ap, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")
    util.save_results(str(output / "metrics.json"), ndcg, mean_ap, recall, precision, mrr)
    util.save_runfile(str(output / "ranking.run.trec"), results)
    return dict(ndcg=ndcg, map=mean_ap, recall=recall, precision=precision, mrr=mrr)


def execute(command, output_log, cwd=None, env=None, timeout=None):
    command = list(map(str, command))
    write_json(Path(output_log).with_suffix(".command.json"), command)
    LOG.info("Executing %s; log: %s", " ".join(command), output_log)
    with Path(output_log).open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def reuse_completed(base, output, name, cache_signature, hashes):
    """Reuse only finished outputs with matching data/config/source and intact artifacts."""
    for previous in sorted((base / "results").glob(f"*/{name}"), reverse=True):
        if previous == output:
            continue
        try:
            cfg = json.loads((previous / "config.json").read_text())
            state = json.loads((previous / "status.json").read_text())
            data = json.loads((previous / "data.json").read_text())
            if (state["status"] != "SUCCESS" or cfg.get("cache_signature") != cache_signature
                    or data["sha256"] != hashes):
                continue
            for artifact in ("metrics.json", "ranking.run.trec"):
                if file_sha256(previous / artifact) != data["artifact_sha256"][artifact]:
                    raise ValueError("Changed completed artifact")
            metrics = json.loads((previous / "metrics.json").read_text())
            for filename in ("metrics.json", "ranking.run.trec", "data.json", "model.json", "training.json",
                             "generation.json", "deepct.json"):
                if (previous / filename).is_file():
                    shutil.copyfile(previous / filename, output / filename)
            write_json(output / "reused_from.json", {"directory": str(previous)})
            LOG.info("Reused completed %s: %s", name, previous)
            return metrics
        except (OSError, ValueError, KeyError):
            continue
    return None


def run(settings, entrypoint):
    from .adapters import build_adapter

    settings = dict(settings)
    settings["datasets"] = ["trec-covid" if name == "trec" else name for name in settings["datasets"]]
    supported = DATASETS + local_datasets()
    if not settings["datasets"] or set(settings["datasets"]) - set(supported):
        raise ValueError(f"Choose only these datasets: {supported}")
    if len(set(settings["datasets"])) != len(settings["datasets"]):
        raise ValueError("Duplicate datasets")
    if min(settings["batch_size"], settings["corpus_chunk_size"], *settings["k_values"]) < 1:
        raise ValueError("Batch/chunk sizes and k-values must be positive")
    if not {10, 100}.issubset(settings["k_values"]):
        raise ValueError("Include k=10 and k=100 for summary.csv")
    base = Path(entrypoint).resolve().parent
    run_id = os.environ.get("BEIR_RUN_ID") or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if Path(run_id).name != run_id:
        raise ValueError("Invalid run ID")
    session = base / "results" / run_id
    session.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[LoggingHandler(), logging.FileHandler(session / "run.log", encoding="utf-8")],
                        force=True)
    config = {**settings, "run_id": run_id, "split": "test", "beir_source": str(ROOT), "python": sys.version}
    source_hashes = {path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")}
    source_hashes["entrypoint"] = file_sha256(entrypoint)
    config["source_sha256"] = source_hashes
    config["cache_signature"] = signature({"settings": settings, "source": source_hashes})
    config["versions"] = {}
    for package in ("beir", "torch", "transformers", "sentence-transformers", "pyserini", "numpy", "pytrec-eval-terrier"):
        try:
            config["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config["versions"][package] = None
    write_json(session / "config.json", config)
    shutil.copyfile(entrypoint, session / Path(entrypoint).name)
    # Shared source matters as well as the entrypoint for reproducibility.
    snapshot = session / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    LOG.info("%s session: %s", settings["method"], session)
    try:
        adapter = build_adapter(settings, base)
        adapter.preflight()
    except Exception as exc:
        LOG.exception("Prerequisite check failed before dataset processing")
        write_json(session / "status.json", {"status": "FAILED", "phase": "preflight", "error": str(exc)})
        return 1
    failures = 0
    fields = ["dataset", "split", "model", "status", "NDCG@10", "Recall@100", "error"]
    with (session / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for name in settings["datasets"]:
            output = session / name
            output.mkdir()
            handler = logging.FileHandler(output / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(handler)
            row = dict(dataset=name, split="test", model=settings["method"], status="FAILED")
            write_json(output / "config.json", {**config, "dataset": name})
            started = time.perf_counter()
            try:
                data_path = prepare_dataset(name, base, settings.get("existing_data_dirs", []))
                hashes = {filename: file_sha256(data_path / filename)
                          for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")}
                previous = reuse_completed(base, output, name, config["cache_signature"], hashes) if settings.get("reuse_cache", True) else None
                if previous is not None:
                    row.update(status="SUCCESS", **{"NDCG@10": previous["ndcg"]["NDCG@10"],
                                                    "Recall@100": previous["recall"]["Recall@100"]})
                    continue
                corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split="test")
                if not corpus or not queries or not qrels:
                    raise ValueError("Empty dataset")
                context = dict(dataset=name, data_path=data_path, hashes=hashes, output=output,
                               cache=base / "cache" / name)
                results = adapter.retrieve(corpus, queries, context)
                metrics = score_and_save(output, results, corpus, queries, qrels, settings["k_values"])
                write_json(output / "data.json", {"path": str(data_path), "sha256": hashes,
                           "documents": len(corpus), "queries": len(queries),
                           "elapsed_seconds": time.perf_counter() - started,
                           "artifact_sha256": {filename: file_sha256(output / filename)
                                               for filename in ("metrics.json", "ranking.run.trec")}})
                row.update(status="SUCCESS", **{"NDCG@10": metrics["ndcg"]["NDCG@10"],
                                                "Recall@100": metrics["recall"]["Recall@100"]})
                del results, corpus, queries, qrels
            except Exception as exc:
                LOG.exception("%s failed on %s", settings["method"], name)
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
            finally:
                write_json(output / "status.json", row)
                writer.writerow(row)
                stream.flush()
                logging.getLogger().removeHandler(handler)
                handler.close()
                gc.collect()
                if "torch" in sys.modules:
                    torch = sys.modules["torch"]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    write_json(session / "status.json", {"status": "SUCCESS" if not failures else "PARTIAL_FAILURE",
               "succeeded": len(settings["datasets"]) - failures, "failed": failures})
    LOG.info("Finished %s: %d succeeded, %d failed; %s", settings["method"],
             len(settings["datasets"]) - failures, failures, session / "summary.csv")
    return int(bool(failures))
