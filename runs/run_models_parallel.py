"""Start remaining methods together; observe process exits with Linux pidfds.

Optional --attach-state preserves a running model from a stopped sequential
supervisor. Stop that supervisor and its retry watcher before using this flag.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import platform
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
from run_models import METHODS, collect


def pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Linux pidfd fallback requires x86_64/aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open")
    return fd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-state", type=Path)
    args = parser.parse_args()
    lock = (RUNS / ".model-queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_parallel_" + uuid.uuid4().hex[:8]
    folder = RUNS / "model_queues" / run_id
    folder.mkdir(parents=True)
    state = dict(status="RUNNING", mode="PARALLEL", pid=os.getpid(), directory=str(folder), jobs=[])
    poll = select.poll()
    pending = {}
    attached = set()
    previous = json.loads(args.attach_state.read_text()) if args.attach_state else {"jobs": []}

    def register(job, process=None):
        try:
            fd = pidfd(job["pid"])
        except ProcessLookupError:
            fd = None
        state["jobs"].append(job)
        if fd is not None:
            poll.register(fd, select.POLLIN)
            pending[fd] = (job, process)
        else:
            job["status"] = "FAILED"
            job["error"] = "Process exited before observation; inspect dataset artifacts"

    for old in previous["jobs"]:
        if old["status"] == "RUNNING":
            job = {**old, "attached": True, "status": "RUNNING"}
            register(job)
            attached.add(job["model"])
    completed = {job["model"] for job in previous["jobs"] if job["status"] == "SUCCESS"}
    for name, filename in METHODS:
        if name in attached or name in completed:
            continue
        job_id = run_id + "_" + name.replace("+", "_")
        output = RUNS / name / "results" / job_id
        env = os.environ.copy()
        env.update(BEIR_RUN_ID=job_id, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        log = folder / (name.replace("+", "_") + ".log")
        with log.open("w") as stream:
            process = subprocess.Popen([sys.executable, str(RUNS / name / filename)],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT)
        job = dict(model=name, status="RUNNING", directory=str(output),
                   started=time.time(), pid=process.pid, log=str(log))
        register(job, process)
        print("Started", name, "PID", process.pid, flush=True)
        write_json(folder / "state.json", state)
    write_json(folder / "state.json", state)
    write_json(RUNS / "model_queues/latest.json", {"directory": str(folder)})
    print("Parallel session:", folder, flush=True)
    while pending:
        # No repeated log reads or progress polling: wake on process exit only.
        for fd, _ in poll.poll():
            job, process = pending.pop(fd)
            poll.unregister(fd)
            os.close(fd)
            code = process.wait() if process is not None else None
            status_file = Path(job["directory"]) / "status.json"
            result = json.loads(status_file.read_text()) if status_file.exists() else {}
            success = result.get("status") == "SUCCESS" and code in (0, None)
            job.update(status="SUCCESS" if success else "FAILED", exit_code=code,
                       elapsed_seconds=time.time() - job["started"], result=result)
            collect(folder, state["jobs"])
            write_json(folder / "state.json", state)
            print("Finished", job["model"], job["status"], flush=True)
    state["status"] = "COMPLETE" if all(job["status"] == "SUCCESS" for job in state["jobs"]) else "COMPLETE_WITH_FAILURES"
    collect(folder, state["jobs"])
    write_json(folder / "state.json", state)
    return int(state["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
