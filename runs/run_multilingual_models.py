"""Run the four requested embedding models in parallel, with bounded recovery.

The earlier benchmark queue is left untouched. GreenNode is retried after the
user authenticates an approved Hugging Face account on this machine.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS.parent))
from _shared.common import write_json
from _shared.multilingual_dense import DATASETS, SPECS
from complete_benchmark import dataset_paths, dataset_info, verify

METHODS = {
    "BGE-M3": "evaluate_bge_m3.py",
    "Multilingual-E5-Large": "evaluate_multilingual_e5.py",
    "GreenNode-Embedding": "evaluate_greennode.py",
    "Vietnamese-BiEncoder": "evaluate_vietnamese_biencoder.py",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(folder, jobs, inputs, memo):
    rows = []
    for job in jobs:
        method = job["model"]
        for name in DATASETS:
            record = {"model": method, "dataset": name, "split": "test",
                      "status": "MISSING_RESULT" if job["status"] == "SUCCESS" else job["status"],
                      "NDCG@10": "", "Recall@100": "", "directory": "", "error": ""}
            for attempt in reversed(job["attempts"]):
                output = Path(attempt["directory"]) / name
                state = output / "status.json"
                if not state.exists():
                    continue
                result = read(state)
                if result.get("status") == "SUCCESS":
                    try:
                        verify(output, name, inputs[name], memo)
                        metrics = read(output / "metrics.json")
                        record.update(status="SUCCESS", **{"NDCG@10": metrics["ndcg"]["NDCG@10"],
                                      "Recall@100": metrics["recall"]["Recall@100"]}, directory=str(output), error="")
                        break
                    except Exception as exc:
                        record.update(status="INVALID_RESULT", error=f"{type(exc).__name__}: {exc}")
                elif not record["error"]:
                    record.update(error=result.get("error", ""), directory=str(output))
                    # A failed earlier attempt may now be retrying or awaiting authentication.
                    record["status"] = job["status"]
            rows.append(record)
    def csvfile(name, data):
        tmp = folder / (name + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(data)
        tmp.replace(folder / name)
    fields = ["model", "dataset", "split", "status", "NDCG@10", "Recall@100", "directory", "error"]
    csvfile("coverage.csv", [fields, *[[r[k] for k in fields] for r in rows]])
    lookup = {(r["model"], r["dataset"]): r for r in rows}
    for metric, filename in [("NDCG@10", "ndcg_at_10.csv"), ("Recall@100", "recall_at_100.csv")]:
        csvfile(filename, [["dataset", *METHODS], *[
            [d, *[lookup[m, d][metric] if lookup[m, d]["status"] == "SUCCESS" else "_" for m in METHODS]]
            for d in DATASETS]])
    write_json(folder / "verified_artifacts.json", memo)
    return rows


def recovery(job, attempt, exit_code, now):
    """Retry resources/network problems without changing checkpoint or token limit."""
    result_file = Path(attempt["directory"]) / "status.json"
    result = read(result_file) if result_file.exists() else {}
    attempt.update(exit_code=exit_code, result=result, elapsed_seconds=now-attempt["started"])
    if exit_code == 3 or result.get("status") == "WAITING_ACCESS":
        job.update(status="WAITING_ACCESS", next_retry=now+60)
        return
    if exit_code == 0 and result.get("status") == "SUCCESS":
        job["status"] = "SUCCESS"
        return
    # Retry failed/incomplete datasets only. Completed datasets are checksum-audited.
    failed = []
    oom = False
    for name in attempt["datasets"]:
        p = Path(attempt["directory"]) / name / "status.json"
        d = read(p) if p.exists() else {}
        if d.get("status") != "SUCCESS":
            failed.append(name)
            oom |= "out of memory" in d.get("error", "").lower()
    job["remaining"] = failed or attempt["datasets"]
    job["failed_attempts"] += 1
    if job["failed_attempts"] <= 3:
        if oom:
            job["batch_size"] = max(1, job["batch_size"] // 2)
        job.update(status="RETRYING", next_retry=now+(120 if oom else 30)*job["failed_attempts"])
        job["last_recovery"] = "Reduced batch / released worker GPU memory" if oom else "Fresh process and resumed completed datasets/cache"
    else:
        job["status"] = "FAILED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 4:
        parser.error("--max-parallel must be between 1 and 4")
    lock = (RUNS / ".multilingual-queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_multilingual_" + uuid.uuid4().hex[:8]
    folder = RUNS / "multilingual_queues" / run_id
    folder.mkdir(parents=True)
    inputs = dataset_info(dataset_paths())
    jobs = [{"model": method, "status": "PENDING", "remaining": DATASETS.copy(),
             "batch_size": SPECS[method]["batch_size"], "attempts": [], "failed_attempts": 0}
            for method in METHODS]
    state = {"status": "RUNNING", "mode": "PARALLEL", "max_parallel": args.max_parallel,
             "pid": os.getpid(), "directory": str(folder), "expected_pairs": 36, "verified_pairs": 0,
             "jobs": jobs, "preserves_existing_queue": True}
    write_json(RUNS / "multilingual_queues/latest.json", {"directory": str(folder)})
    write_json(folder / "inputs.json", {n: {"path": d["path"], "sha256": d["sha256"],
        "documents": len(d["documents"]), "queries": len(d["queries"])} for n, d in inputs.items()})
    memo = {}
    running = {}
    def stop(signum, frame):
        for process, _, _ in running.values():
            process.terminate()
        state["status"] = "INTERRUPTED"
        write_json(folder / "state.json", state)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_collect = 0
    while True:
        now = time.time()
        for job in jobs:
            if len(running) >= args.max_parallel:
                break
            if job["status"] not in {"PENDING", "RETRYING", "WAITING_ACCESS"} or now < job.get("next_retry", 0):
                continue
            if job["status"] == "WAITING_ACCESS":
                from huggingface_hub import get_token
                if not get_token():
                    job["next_retry"] = now+60
                    continue
            method = job["model"]
            idx = len(job["attempts"])+1
            model_id = run_id+"_"+method+f"_try{idx}"
            out = RUNS / method / "results" / model_id
            log = folder / f"{method}.try{idx}.log"
            command = [sys.executable, "-u", str(RUNS / method / METHODS[method]),
                       "--datasets", *job["remaining"], "--batch-size", str(job["batch_size"]), "--device", "cuda"]
            env = os.environ.copy()
            env.update(BEIR_RUN_ID=model_id, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="2", HF_HUB_DOWNLOAD_TIMEOUT="180", HF_HUB_ETAG_TIMEOUT="30")
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            attempt = {"pid": process.pid, "directory": str(out), "log": str(log),
                       "datasets": job["remaining"].copy(), "started": now, "batch_size": job["batch_size"], "command": command}
            job["attempts"].append(attempt)
            job.update(status="RUNNING", pid=process.pid)
            running[process.pid] = (process, job, attempt)
            write_json(folder / "state.json", state)
            print("Started", method, "PID", process.pid, "attempt", idx, flush=True)
        for pid, (process, job, attempt) in list(running.items()):
            code = process.poll()
            if code is not None:
                recovery(job, attempt, code, time.time())
                del running[pid]
                print("Worker exited", job["model"], code, "status", job["status"], flush=True)
        if now-last_collect >= 30 or not running:
            rows = collect(folder, jobs, inputs, memo)
            state["verified_pairs"] = sum(r["status"] == "SUCCESS" for r in rows)
            # The audit remains authoritative over a worker's exit code.
            for job in jobs:
                selected = [r for r in rows if r["model"] == job["model"]]
                if all(r["status"] == "SUCCESS" for r in selected):
                    job["status"] = "SUCCESS"
                elif job["status"] == "SUCCESS":
                    job.update(status="FAILED", error="Worker completed but audit rejected at least one result")
            write_json(folder / "state.json", state)
            last_collect = time.time()
        if not running and all(job["status"] in {"SUCCESS", "FAILED"} for job in jobs):
            state["status"] = "COMPLETE" if state["verified_pairs"] == 36 else "INCOMPLETE"
            write_json(folder / "state.json", state)
            return int(state["status"] != "COMPLETE")
        # WAITING_ACCESS is kept active; login on disk can unblock it without restarting the queue.
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
