"""Run the eight requested methods sequentially; never use sudo or Docker."""
from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS.parent))
sys.path.insert(0, str(RUNS))
from _shared.common import write_json

METHODS = [
    ("DeepCT", "evaluate_deepct.py"), ("SPARTA", "evaluate_sparta.py"),
    ("docT5query", "evaluate_doct5query.py"), ("DPR", "evaluate_dpr.py"),
    ("TAS-B", "evaluate_tasb.py"), ("GenQ", "evaluate_genq.py"),
    ("ColBERT", "evaluate_colbert.py"), ("BM25+CE", "evaluate_bm25_ce.py"),
]


def collect(queue, jobs):
    fields = ["model", "dataset", "status", "NDCG@10", "Recall@100", "error", "directory"]
    latest_rows = {}
    for job in jobs:
        folder = Path(job["directory"])
        for status_file in sorted(folder.glob("*/status.json")):
            status = json.loads(status_file.read_text())
            latest_rows[(status["model"], status["dataset"])] = {
                key: status.get(key, "") for key in fields[:-1]
            } | {"directory": str(status_file.parent)}
    with (queue / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(latest_rows.values())


def main():
    lock_file = (RUNS / ".model-queue.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A model queue is already running; refusing to start another")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    queue = RUNS / "model_queues" / run_id
    queue.mkdir(parents=True)
    jobs = []
    state = {"status": "RUNNING", "directory": str(queue), "pid": os.getpid(), "jobs": jobs}
    write_json(RUNS / "model_queues/latest.json", {"directory": str(queue)})
    write_json(queue / "state.json", state)
    print("Queue:", queue, flush=True)
    for name, filename in METHODS:
        job_id = run_id + "_" + name.replace("+", "_")
        output = RUNS / name / "results" / job_id
        job = {"model": name, "status": "RUNNING", "directory": str(output), "started": time.time()}
        jobs.append(job)
        write_json(queue / "state.json", state)
        env = os.environ.copy()
        env["BEIR_RUN_ID"] = job_id
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["OMP_NUM_THREADS"] = "4"
        log_file = queue / (name.replace("+", "_") + ".log")
        print("Starting:", name, flush=True)
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(RUNS / name / filename)], env=env, stdout=log, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            write_json(queue / "state.json", state)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                state["status"] = "INTERRUPTED"
                write_json(queue / "state.json", state)
                raise
        job.update(status="SUCCESS" if code == 0 else "FAILED", exit_code=code,
                   elapsed_seconds=time.time() - job["started"])
        if code:
            with log_file.open(errors="replace") as log:
                job["last_log_lines"] = list(deque(log, maxlen=20))
        collect(queue, jobs)
        write_json(queue / "state.json", state)
        print("Finished:", name, job["status"], "seconds=", round(job["elapsed_seconds"]), flush=True)
    state["status"] = "COMPLETE" if all(job["status"] == "SUCCESS" for job in jobs) else "COMPLETE_WITH_FAILURES"
    collect(queue, jobs)
    write_json(queue / "state.json", state)
    print(state["status"], queue / "summary.csv", flush=True)
    return int(state["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
