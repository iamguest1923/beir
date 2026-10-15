"""Wait for the existing queue, then retry failed methods sequentially once.

Uses a Linux pidfd to wait for the supervisor instead of repeatedly reading logs.
The failed method's dataset cache/completed artifacts are reused where valid.
"""
import fcntl
import ctypes
import json
import os
import select
import subprocess
import sys
import time
import uuid
from pathlib import Path

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS.parent))
sys.path.insert(0, str(RUNS))
from _shared.common import write_json
from run_models import METHODS, collect


def main():
    queue = Path(json.loads((RUNS / "model_queues/latest.json").read_text())["directory"])
    guard = (queue / ".retry.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((queue / "state.json").read_text())
    if state["status"] == "RUNNING":
        try:
            if hasattr(os, "pidfd_open"):
                fd = os.pidfd_open(state["pid"])
            else:
                # Some Conda Python builds omit the wrapper even though Linux supports it.
                import platform
                if platform.machine() not in ("x86_64", "aarch64"):
                    raise RuntimeError("pidfd syscall fallback supports x86_64/aarch64 only")
                libc = ctypes.CDLL(None, use_errno=True)
                fd = libc.syscall(434, state["pid"], 0)
                if fd < 0:
                    raise OSError(ctypes.get_errno(), "pidfd_open")
        except ProcessLookupError:
            fd = None
        if fd is not None:
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            print("Waiting for queue supervisor to finish, without log polling", flush=True)
            poll.poll()
            os.close(fd)
    lock = (RUNS / ".model-queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = json.loads((queue / "state.json").read_text())
    latest = {job["model"]: job for job in state["jobs"]}
    if state["status"] not in ("COMPLETE", "COMPLETE_WITH_FAILURES"):
        raise RuntimeError("Queue did not finish cleanly; inspect state before retrying")
    state["status"] = "RUNNING_RETRY"
    state["pid"] = os.getpid()
    write_json(queue / "state.json", state)
    for name, filename in METHODS:
        if name not in latest or latest[name]["status"] == "SUCCESS":
            continue
        job_id = queue.name + "_retry_" + name.replace("+", "_") + "_" + uuid.uuid4().hex[:6]
        output = RUNS / name / "results" / job_id
        job = {"model": name, "status": "RUNNING", "phase": "retry",
               "directory": str(output), "started": time.time()}
        state["jobs"].append(job)
        write_json(queue / "state.json", state)
        env = os.environ.copy()
        env.update(BEIR_RUN_ID=job_id, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        print("Retrying", name, flush=True)
        with (queue / (name.replace("+", "_") + ".retry.log")).open("w") as stream:
            process = subprocess.Popen([sys.executable, str(RUNS / name / filename)], env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            write_json(queue / "state.json", state)
            code = process.wait()
        job.update(status="SUCCESS" if code == 0 else "FAILED", exit_code=code,
                   elapsed_seconds=time.time() - job["started"])
        collect(queue, state["jobs"])
        write_json(queue / "state.json", state)
        print("Retried", name, job["status"], flush=True)
    latest = {job["model"]: job for job in state["jobs"]}
    state["status"] = "COMPLETE" if all(job["status"] == "SUCCESS" for job in latest.values()) else "COMPLETE_WITH_FAILURES"
    collect(queue, state["jobs"])
    write_json(queue / "state.json", state)


if __name__ == "__main__":
    main()
