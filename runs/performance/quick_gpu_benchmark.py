"""Short, FP32 encoding benchmark; does not interrupt existing workloads."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = {
    "ANCE": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "TAS-B": "sentence-transformers/msmarco-distilbert-base-tas-b",
}
BATCHES = [16, 128, 32, 64]


def telemetry():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks.current.sm,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip().splitlines()[0]
    values = [float(part.strip()) for part in output.split(",")]
    return dict(zip(["temperature_c", "gpu_util_percent", "memory_mib", "sm_mhz", "power_w"], values))


def worker(connection, model_name, documents):
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from beir.retrieval.models import SentenceBERT
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = SentenceBERT(model_name, max_length=512, device="cuda")
        model.q_model.float()
        connection.send({"ready": True})
        while True:
            command = connection.recv()
            if command is None:
                break
            try:
                batch = command["batch"]
                model.encode_corpus(documents[:min(batch, 32)], batch_size=batch,
                                    show_progress_bar=False, convert_to_tensor=True)
                torch.cuda.synchronize()
                connection.send({"warmed": True})
                if connection.recv() != "go":
                    break
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(command["repeats"]):
                    output = model.encode_corpus(documents, batch_size=batch,
                                                 show_progress_bar=False, convert_to_tensor=True)
                    del output
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                connection.send({"seconds": seconds,
                                 "documents_per_second": len(documents) * command["repeats"] / seconds,
                                 "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                                 "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2})
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                connection.send({"error": "CUDA out of memory"})
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def receive(connection, timeout=240):
    if not connection.poll(timeout):
        raise TimeoutError("Benchmark worker timed out")
    value = connection.recv()
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def main():
    data = ROOT / "examples/retrieval/evaluation/dense/datasets/nfcorpus/corpus.jsonl"
    documents = []
    with data.open() as stream:
        for line in stream:
            row = json.loads(line)
            documents.append({"title": row.get("title") or "", "text": row["text"]})
    documents.sort(key=lambda doc: len(doc["title"] + doc["text"]), reverse=True)
    # Fixed mix of long, middle and short passages; both models use the same input.
    middle = len(documents) // 2
    documents = documents[:128] + documents[middle:middle + 64] + documents[-64:]
    run = Path(__file__).resolve().parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir()
    result = {"models": MODELS, "documents": len(documents), "precision": "float32",
              "max_length": 512, "background_processes": subprocess.check_output([
                  "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"
              ], text=True), "sweep": [], "pair_trials": [], "telemetry": []}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                result["telemetry"].append({"time": time.time(), **telemetry()})
            except Exception:
                pass
            stop.wait(1)

    def save():
        (run / "results.json").write_text(json.dumps(result, indent=2))

    def check_temperature():
        state = telemetry()
        if state["temperature_c"] >= 92:
            raise RuntimeError("Stopping benchmark: GPU reached 92 C")

    context = mp.get_context("spawn")
    processes, connections = {}, {}
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        for name, model_name in MODELS.items():
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, model_name, documents))
            process.start()
            processes[name], connections[name] = process, parent
            receive(parent)
            print(name, "loaded", flush=True)

        for name, connection in connections.items():
            for batch in BATCHES:
                check_temperature()
                connection.send({"batch": batch, "repeats": 2})
                receive(connection)
                connection.send("go")
                measurement = {"model": name, "batch": batch, **receive(connection)}
                result["sweep"].append(measurement)
                save()
                print(json.dumps(measurement), flush=True)
        best = {name: max((r for r in result["sweep"] if r["model"] == name),
                          key=lambda r: r["documents_per_second"])["batch"] for name in MODELS}
        result["best_batches"] = best

        for mode in ["sequential", "concurrent", "concurrent", "sequential"]:
            check_temperature()
            # Warm both models outside the timer; only encoding is compared.
            for name, connection in connections.items():
                connection.send({"batch": best[name], "repeats": 2})
                receive(connection)
            before = telemetry()
            started = time.perf_counter()
            if mode == "sequential":
                for connection in connections.values():
                    connection.send("go")
                    receive(connection)
            else:
                for connection in connections.values():
                    connection.send("go")
                for connection in connections.values():
                    receive(connection)
            seconds = time.perf_counter() - started
            trial = {"mode": mode, "seconds": seconds,
                     "total_documents_per_second": len(documents) * 2 * len(MODELS) / seconds,
                     "before": before, "after": telemetry()}
            result["pair_trials"].append(trial)
            save()
            print(json.dumps(trial), flush=True)
        result["pair_medians_seconds"] = {
            mode: statistics.median(r["seconds"] for r in result["pair_trials"] if r["mode"] == mode)
            for mode in ["sequential", "concurrent"]
        }
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        stop.set()
        sampler.join(timeout=10)
        for connection in connections.values():
            try:
                connection.send(None)
            except Exception:
                pass
        for process in processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        save()
        print("Saved:", run / "results.json", flush=True)


if __name__ == "__main__":
    main()
