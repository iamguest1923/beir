"""Observe current jobs and retry failed datasets once, without progress polling.

Repairs for distinct stopped models start together. A still-running model is
left alone; its failed datasets are retried only after that process exits.
"""
import argparse
import csv
import fcntl
import json
import os
import select
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS.parent))
sys.path.insert(0, str(RUNS))
from _shared.common import write_json
from run_models import collect
from run_models_parallel import pidfd


def failed_datasets(job):
    rows = [json.loads(p.read_text()) for p in Path(job["directory"]).glob("*/status.json")]
    return [row["dataset"] for row in rows if row["status"] != "SUCCESS"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    guard = (args.state.parent / ".recovery.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = json.loads(args.state.read_text())
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_recovery_" + uuid.uuid4().hex[:8]
    folder = RUNS / "model_queues" / name
    folder.mkdir(parents=True)
    state = dict(status="RUNNING", mode="RECOVERY", pid=os.getpid(), directory=str(folder),
                 previous_state=str(args.state.resolve()), jobs=old["jobs"])
    pending = {}
    poll = select.poll()

    def save():
        collect(folder, state["jobs"])
        write_json(folder / "state.json", state)

    def register(job, process=None):
        try:
            fd = pidfd(job["pid"])
        except ProcessLookupError:
            return False
        pending[fd] = (job, process)
        poll.register(fd, select.POLLIN)
        return True

    def retry(job):
        datasets = failed_datasets(job)
        if not datasets:
            return
        method = job["model"]
        job_id = name + "_" + method.replace("+", "_")
        log = folder / (method.replace("+", "_") + ".log")
        env = os.environ.copy()
        env.update(BEIR_RUN_ID=job_id, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        command = [sys.executable, str(RUNS / "retry_model.py"), "--model", method,
                   "--datasets", json.dumps(datasets), "--previous", job["directory"]]
        with log.open("w") as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
        new = dict(model=method, status="RUNNING", phase="recovery", pid=process.pid,
                   directory=str(RUNS / method / "results" / job_id), datasets=datasets,
                   started=time.time(), log=str(log))
        state["jobs"].append(new)
        if not register(new, process):
            new.update(status="FAILED", error="Process exited before registration")
        print("Retrying", method, datasets, "PID", process.pid, flush=True)

    for job in list(state["jobs"]):
        if job["status"] == "RUNNING" and register(job):
            continue
        retry(job)
    save()
    write_json(RUNS / "model_queues/latest.json", {"directory": str(folder)})
    print("Recovery session:", folder, flush=True)
    while pending:
        for fd, _ in poll.poll():
            job, process = pending.pop(fd)
            poll.unregister(fd)
            os.close(fd)
            code = process.wait() if process is not None else None
            path = Path(job["directory"]) / "status.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            job.update(status="SUCCESS" if result.get("status") == "SUCCESS" and code in (None, 0) else "FAILED",
                       exit_code=code, result=result)
            print("Finished", job["model"], job.get("phase", "original"), job["status"], flush=True)
            if job.get("phase") != "recovery":
                retry(job)
            save()
    with (folder / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    state["status"] = "COMPLETE" if len(rows) == 48 and all(r["status"] == "SUCCESS" for r in rows) else "COMPLETE_WITH_FAILURES"
    save()


if __name__ == "__main__":
    main()
