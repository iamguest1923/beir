"""Audit real runs and fill missing model/dataset evaluations, preserving old runs.

Use the BEIR Python environment. --audit-only never starts an evaluation.
The supervisor has one CPU lane (BM25) and one serial GPU lane.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

RUNS = Path(__file__).resolve().parent
ROOT = RUNS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNS))
MODELS = {
    "BM25": "evaluate_bm25.py", "ANCE": "evaluate_ance.py",
    "TAS-B": "evaluate_tasb.py", "DPR": "evaluate_dpr.py",
    "DeepCT": "evaluate_deepct.py", "SPARTA": "evaluate_sparta.py",
    "BM25+CE": "evaluate_bm25_ce.py", "ColBERT": "evaluate_colbert.py",
    "GenQ": "evaluate_genq.py", "docT5query": "evaluate_doct5query.py",
}
CORE = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid"]
FILES = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_paths():
    from _shared.common import NEW_DATA, local_datasets, prepare_dataset
    paths = {name: prepare_dataset(name, RUNS / "BM25") for name in CORE}
    paths.update({name: NEW_DATA / name for name in local_datasets()})
    return paths


def dataset_info(paths):
    data = {}
    for name, folder in paths.items():
        docs = set()
        with (folder / "corpus.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["_id"] in docs:
                    raise ValueError(f"Duplicate document ID in {name}: {row['_id']}")
                docs.add(row["_id"])
        queries = set()
        with (folder / "qrels/test.tsv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                queries.add(row["query-id"])
        data[name] = {"path": str(folder.resolve()), "documents": docs, "queries": queries,
                      "sha256": {f: digest(folder / f) for f in FILES}}
    return data


def verify(folder, name, info, memo):
    """Check source, coverage, artifact hashes and recompute saved metrics."""
    from beir.retrieval.evaluation import EvaluateRetrieval
    status, config, meta, metrics = [load(folder / f) for f in
                                    ["status.json", "config.json", "data.json", "metrics.json"]]
    if status.get("status") != "SUCCESS":
        raise ValueError("Dataset status is not SUCCESS")
    if config.get("dataset") != name or status.get("split", config.get("split", "test")) != "test":
        raise ValueError("Dataset/split mismatch")
    if meta.get("documents") != len(info["documents"]) or meta.get("queries") != len(info["queries"]):
        raise ValueError("Dataset counts differ")
    if "sha256" in meta:
        if meta["sha256"] != info["sha256"]:
            raise ValueError("Dataset checksums differ")
    else:
        # Legacy BM25/ANCE did not save input hashes. Check their recorded files
        # and disclose this limitation in the exported audit.
        if any(digest(Path(meta["path"]) / f) != info["sha256"][f] for f in FILES):
            raise ValueError("Legacy input files differ from selected corpus")
    hashes = {f: digest(folder / f) for f in ["metrics.json", "ranking.run.trec"]}
    for f, expected in meta.get("artifact_sha256", {}).items():
        if f in hashes and hashes[f] != expected:
            raise ValueError(f"Artifact checksum mismatch: {f}")
    if not math.isclose(float(status["NDCG@10"]), metrics["ndcg"]["NDCG@10"], abs_tol=1e-10):
        raise ValueError("Status/metrics disagree")
    key = str(folder.resolve())
    fingerprint = {"artifact_sha256": hashes, "input_sha256": info["sha256"]}
    if memo.get(key, {}).get("fingerprint") == fingerprint:
        return memo[key]
    # TREC runfiles have no representation for a query with zero hits. Restore
    # such queries before recomputing metrics, including them in the denominator.
    results = {qid: {} for qid in info["queries"]}
    with (folder / "ranking.run.trec").open(encoding="utf-8") as stream:
        for line in stream:
            qid, _, did, _, score, _ = line.split()
            if qid not in info["queries"] or did not in info["documents"]:
                raise ValueError(f"Ranking has unknown query/document: {qid}/{did}")
            if not math.isfinite(float(score)) or did in results[qid]:
                raise ValueError("Ranking has duplicate pair or nonfinite score")
            results[qid][did] = float(score)
    rels = {}
    with (Path(info["path"]) / "qrels/test.tsv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            rels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    ks = sorted(int(k.split("@")[1]) for k in metrics["ndcg"])
    values = EvaluateRetrieval.evaluate(rels, results, ks)
    recalculated = dict(zip(["ndcg", "map", "recall", "precision"], values))
    recalculated["mrr"] = EvaluateRetrieval.evaluate_custom(rels, results, ks, metric="mrr")
    for group, scores in metrics.items():
        for metric, score in scores.items():
            if not math.isfinite(score) or not math.isclose(score, recalculated[group][metric], abs_tol=1e-5):
                raise ValueError(f"Recomputed metric differs: {group}/{metric}")
    record = {"fingerprint": fingerprint, "verified_at": datetime.now(timezone.utc).isoformat(),
              "checks": "input files, IDs, ranking coverage, checksums, recomputed metrics",
              "queries_with_zero_saved_hits": sorted(q for q, hits in results.items() if not hits),
              "legacy_input_hashes_absent": "sha256" not in meta}
    memo[key] = record
    return record


def audit(paths, info, memo):
    rows = []
    for model in MODELS:
        for name in paths:
            candidates = sorted((RUNS / model / "results").glob(f"*/{name}/status.json"), reverse=True)
            errors = []
            selected = None
            for p in candidates:
                try:
                    if load(p).get("status") != "SUCCESS":
                        continue
                    checks = verify(p.parent, name, info[name], memo)
                    metrics = load(p.parent / "metrics.json")
                    selected = {"model": model, "dataset": name, "status": "SUCCESS",
                                "NDCG@10": metrics["ndcg"]["NDCG@10"],
                                "Recall@100": metrics["recall"]["Recall@100"],
                                "directory": str(p.parent.resolve()), "error": "",
                                "legacy_input_hashes_absent": checks["legacy_input_hashes_absent"]}
                    break
                except Exception as exc:
                    errors.append(f"{p.parent}: {type(exc).__name__}: {exc}")
            if selected is None:
                latest = load(candidates[0]) if candidates else {}
                selected = {"model": model, "dataset": name, "status": "PENDING",
                            "NDCG@10": "", "Recall@100": "", "directory": "",
                            "error": "; ".join(errors) or latest.get("error", "No verified completed result"),
                            "legacy_input_hashes_absent": ""}
            rows.append(selected)
    return rows


def export(folder, rows, info, memo):
    folder.mkdir(parents=True, exist_ok=True)
    def csvfile(filename, contents):
        temp = folder / (filename + ".tmp")
        with temp.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.writer(stream).writerows(contents)
        temp.replace(folder / filename)
    fields = ["model", "dataset", "status", "NDCG@10", "Recall@100", "directory", "error", "legacy_input_hashes_absent"]
    csvfile("coverage.csv", [fields, *[[r[k] for k in fields] for r in rows]])
    for metric, filename in [("NDCG@10", "ndcg_at_10.csv"), ("Recall@100", "recall_at_100.csv")]:
        lookup = {(r["model"], r["dataset"]): r for r in rows}
        csvfile(filename, [["dataset", *MODELS], *[
            [name, *[lookup[m, name][metric] if lookup[m, name]["status"] == "SUCCESS" else "_" for m in MODELS]]
            for name in info]])
    write(folder / "verified_artifacts.json", memo)
    write(folder / "inputs.json", {name: {k: v for k, v in d.items() if k not in {"documents", "queries"}}
                                  | {"documents": len(d["documents"]), "queries": len(d["queries"])}
                                  for name, d in info.items()})


def worker(model, datasets):
    from _shared.common import run
    entrypoint = RUNS / model / MODELS[model]
    module = runpy.run_path(str(entrypoint))
    if model in {"BM25", "ANCE"}:
        settings = {"method": model, "datasets": datasets, "device": None,
                    "batch_size": module["BATCH_SIZE"], "corpus_chunk_size": module.get("CORPUS_CHUNK_SIZE", 10000),
                    "max_length": module.get("MAX_LENGTH", 512), "k_values": module["K_VALUES"],
                    "seed": 42, "reuse_cache": True, "keep_index": False,
                    "cpu_threads": 4, "query_batch_size": 32}
        if model == "ANCE":
            settings["model"] = module["MODEL_NAME"]
        else:
            settings.update(bm25_k1=module["K1"], bm25_b=module["B"], bm25_fields=module["FIELDS"])
    else:
        settings = module["SETTINGS"].copy()
        settings["datasets"] = datasets
    if model == "BM25+CE":
        settings.update(batch_size=8, query_batch_size=8)
    if model == "ColBERT":
        settings["partitions"] = {**settings["partitions"], "uit-viquad": 96, "vihealthqa": 128, "zalo-legal": 512}
        settings["rank_batch_size"] = 256
    # Record this dispatch source as part of the actual run configuration.
    settings["completion_dispatch_sha256"] = digest(__file__)
    return run(settings, entrypoint)


def supervise(folder, paths, info, memo, rows):
    lock = (RUNS / ".completion-queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Respect the original queue's lock as well, preventing overlapping writers.
    original_lock = (RUNS / ".model-queue.lock").open("a")
    fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for model in MODELS:
        missing = [r["dataset"] for r in rows if r["model"] == model and r["status"] != "SUCCESS"]
        if missing:
            jobs.append({"model": model, "datasets": missing, "status": "PENDING"})
    state = {"status": "RUNNING", "pid": os.getpid(), "directory": str(folder), "jobs": jobs,
             "expected_pairs": len(rows), "verified_pairs": sum(r["status"] == "SUCCESS" for r in rows)}
    write(folder / "state.json", state)
    running = {}
    while any(j["status"] in {"PENDING", "RUNNING"} for j in jobs):
        occupied = {j["lane"] for j, _ in running.values()}
        for job in jobs:
            lane = "CPU" if job["model"] == "BM25" else "GPU"
            if job["status"] != "PENDING" or lane in occupied:
                continue
            run_id = folder.name + "_" + job["model"].replace("+", "_")
            env = os.environ.copy()
            env.update(BEIR_RUN_ID=run_id, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
            log = folder / (job["model"].replace("+", "_") + ".log")
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", job["model"],
                                            "--datasets-json", json.dumps(job["datasets"])],
                                           env=env, stdout=stream, stderr=subprocess.STDOUT)
            job.update(status="RUNNING", lane=lane, pid=process.pid, started=time.time(), log=str(log),
                       directory=str(RUNS / job["model"] / "results" / run_id))
            running[process.pid] = (job, process)
            occupied.add(lane)
            write(folder / "state.json", state)
            print("Started", job["model"], job["datasets"], "PID", process.pid, flush=True)
        for pid, (job, process) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            job.update(status="SUCCESS" if code == 0 else "FAILED", exit_code=code,
                       elapsed_seconds=time.time() - job["started"])
            del running[pid]
            rows = audit(paths, info, memo)
            export(folder, rows, info, memo)
            state["verified_pairs"] = sum(r["status"] == "SUCCESS" for r in rows)
            write(folder / "state.json", state)
            print("Finished", job["model"], job["status"], "verified", state["verified_pairs"], flush=True)
        time.sleep(5)
    rows = audit(paths, info, memo)
    export(folder, rows, info, memo)
    state.update(status="COMPLETE" if all(r["status"] == "SUCCESS" for r in rows) else "INCOMPLETE",
                 verified_pairs=sum(r["status"] == "SUCCESS" for r in rows))
    write(folder / "state.json", state)
    return int(state["status"] != "COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--datasets-json")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, json.loads(args.datasets_json))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_completion_" + uuid.uuid4().hex[:8]
    folder = (args.output or RUNS / "completion_queues" / run_id).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths()
    info = dataset_info(paths)
    memo = load(folder / "verified_artifacts.json") if (folder / "verified_artifacts.json").is_file() else {}
    rows = audit(paths, info, memo)
    export(folder, rows, info, memo)
    write(RUNS / "completion_queues/latest.json", {"directory": str(folder)})
    print("Audit", sum(r["status"] == "SUCCESS" for r in rows), "/", len(rows), "verified; report", folder, flush=True)
    if args.audit_only:
        return 0
    return supervise(folder, paths, info, memo, rows)


if __name__ == "__main__":
    raise SystemExit(main())
