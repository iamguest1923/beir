"""Read compact queue state and only new log bytes; safe to invoke occasionally."""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent


def main():
    latest = RUNS / "model_queues/latest.json"
    if not latest.exists():
        print("No model queue has started")
        return
    queue = Path(json.loads(latest.read_text())["directory"])
    state = json.loads((queue / "state.json").read_text())
    print("Queue:", queue.name, state["status"])
    offsets_file = queue / ".status-offsets.json"
    offsets = json.loads(offsets_file.read_text()) if offsets_file.exists() else {}
    groups = {}
    for job in state["jobs"]:
        groups.setdefault(job["model"], []).append(job)
    for model, jobs in groups.items():
        latest_rows = {}
        for job in jobs:
            for file in sorted(Path(job["directory"]).glob("*/status.json")):
                row = json.loads(file.read_text())
                latest_rows[row["dataset"]] = row
        rows = sorted(latest_rows.values(), key=lambda row: row["dataset"])
        successful = sum(row["status"] == "SUCCESS" for row in rows)
        status = "SUCCESS" if successful == 6 else (
            "RUNNING" if any(job["status"] == "RUNNING" for job in jobs) else "PARTIAL_FAILURE")
        print(model, status, [(row["dataset"], row["status"], row.get("NDCG@10")) for row in rows])
    for job in state["jobs"]:
        output = Path(job["directory"])
        log = output / "run.log"
        if job["status"] == "RUNNING" and log.exists():
            key = str(log)
            size = log.stat().st_size
            start = max(offsets.get(key, 0), size - 16384)
            if start > size:
                start = 0
            with log.open("rb") as stream:
                stream.seek(start)
                text = stream.read().decode(errors="replace")
            offsets[key] = size
            lines = [line for line in text.splitlines() if any(word in line for word in (
                "ERROR", "Failed", "failed", "Generated/reused", "Encoding Batch", "Executing", "Reusing", "Finished"
            ))]
            for line in lines[-5:]:
                print(" ", line[:400])
    offsets_file.write_text(json.dumps(offsets))


if __name__ == "__main__":
    main()
