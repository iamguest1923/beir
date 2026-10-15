"""Export latest successful local evaluation results in the BEIR paper's layout.

Reads metrics/status/config artifacts only; never reads full logs or rankings,
downloads models, or starts/stops evaluations. Missing scores are exported as _.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(__file__).resolve().parent
MODELS = ["BM25", "DeepCT", "SPARTA", "docT5query", "DPR", "ANCE", "TAS-B", "GenQ", "ColBERT", "BM25+CE"]
DATASETS = [
    ("MS MARCO", "msmarco"), ("TREC-COVID", "trec-covid"), ("BioASQ", "bioasq"),
    ("NFCorpus", "nfcorpus"), ("NQ", "nq"), ("HotpotQA", "hotpotqa"),
    ("FiQA-2018", "fiqa"), ("Signal-1M (RT)", "signal1m"),
    ("TREC-NEWS", "trec-news"), ("Robust04", "robust04"), ("ArguAna", "arguana"),
    ("Touche-2020", "webis-touche2020"), ("CQADupStack", "cqadupstack"),
    ("Quora", "quora"), ("DBPedia", "dbpedia-entity"), ("SCIDOCS", "scidocs"),
    ("FEVER", "fever"), ("Climate-FEVER", "climate-fever"), ("SciFact", "scifact"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_successes():
    selected, warnings = {}, []
    for model in MODELS:
        sessions = RUNS / model / "results"
        if not sessions.exists():
            continue
        for session in sorted(sessions.iterdir()):
            if not session.is_dir():
                continue
            # Restrict to real runner sessions; model_checks and benchmark fixtures
            # live elsewhere and must never be interpreted as benchmark results.
            try:
                started = datetime.strptime(session.name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            for metric_path in sorted(session.glob("*/metrics.json")):
                folder = metric_path.parent
                try:
                    status_path, config_path = folder / "status.json", folder / "config.json"
                    if not status_path.exists() or load(status_path).get("status") != "SUCCESS":
                        continue
                    status, config, metrics = load(status_path), load(config_path), load(metric_path)
                    dataset = config["dataset"]
                    if dataset != folder.name:
                        raise ValueError("Dataset folder/config mismatch")
                    split = status.get("split", config.get("split", "_"))
                    expected_split = "dev" if dataset == "msmarco" else "test"
                    if split != expected_split:
                        raise ValueError(f"Expected split {expected_split}, found {split}")
                    value = float(metrics["ndcg"]["NDCG@10"])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid NDCG@10")
                    if "NDCG@10" in status and not math.isclose(value, float(status["NDCG@10"]), abs_tol=1e-10):
                        raise ValueError("Metric/status score mismatch")
                    data_path = folder / "data.json"
                    data = load(data_path) if data_path.exists() else {}
                    expected_hash = data.get("artifact_sha256", {}).get("metrics.json")
                    actual_hash = sha256(metric_path)
                    if expected_hash and actual_hash != expected_hash:
                        raise ValueError("Metrics checksum mismatch")
                    record = dict(model=model, dataset=dataset, split=split, score=value,
                                  metrics=metrics, metrics_path=metric_path, config_path=config_path,
                                  status_path=status_path, run_id=session.name, metrics_sha256=actual_hash,
                                  started=started, timestamp=metric_path.stat().st_mtime_ns)
                    key = model, dataset
                    previous = selected.get(key)
                    if previous is None or (started, record["timestamp"], session.name) > (
                        previous["started"], previous["timestamp"], previous["run_id"]
                    ):
                        selected[key] = record
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    warnings.append(f"{metric_path}: {exc}")
    return selected, warnings


def write_csv(path, rows):
    # UTF-8 BOM helps Excel recognize the encoding; standard csv handles commas.
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        csv.writer(stream).writerows(rows)


def export(output):
    selected, warnings = latest_successes()
    known = {dataset for _, dataset in DATASETS}
    unexpected = {dataset for _, dataset in selected} - known
    if unexpected:
        raise ValueError(f"Successful datasets missing from table layout: {sorted(unexpected)}")
    output.mkdir(parents=True, exist_ok=False)
    for filename, precision in (("ndcg_at_10.csv", 3), ("ndcg_at_10_full_precision.csv", None)):
        rows = [["Dataset", *MODELS]]
        for label, dataset in DATASETS:
            row = [label]
            for model in MODELS:
                record = selected.get((model, dataset))
                row.append("_" if record is None else (
                    repr(record["score"]) if precision is None else f"{record['score']:.3f}"
                ))
            rows.append(row)
        # Do not substitute a mean over a smaller, different set of datasets for
        # the full benchmark comparison printed at the bottom of the paper.
        rows.append(["Avg. Performance vs. BM25", *(["_"] * len(MODELS))])
        write_csv(output / filename, rows)
    sources = [["Dataset", "Model", "Split", "NDCG@10", "Run", "MetricsFile", "ConfigFile", "StatusFile", "MetricsSHA256"]]
    all_metrics = [["Dataset", "Model", "Split", "MetricGroup", "Metric", "Value", "Run"]]
    for label, dataset in DATASETS:
        for model in MODELS:
            record = selected.get((model, dataset))
            if record is None:
                continue
            sources.append([label, model, record["split"], repr(record["score"]), record["run_id"],
                            str(record["metrics_path"]), str(record["config_path"]), str(record["status_path"]),
                            record["metrics_sha256"]])
            for group, values in record["metrics"].items():
                for metric, value in values.items():
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Invalid metric {group}/{metric} in {record['metrics_path']}")
                    all_metrics.append([label, model, record["split"], group, metric, repr(value), record["run_id"]])
    write_csv(output / "sources.csv", sources)
    write_csv(output / "all_metrics.csv", all_metrics)
    counts = Counter(model for model, _ in selected)
    generated = datetime.now(timezone.utc).isoformat()
    readme = [
        "# Kết quả BEIR từ các lượt chạy thực tế", "", f"Thời điểm xuất (UTC): {generated}", "",
        "- ndcg_at_10.csv: bảng nDCG@10 theo thứ tự dataset/model trong ảnh, làm tròn 3 chữ số.",
        "- ndcg_at_10_full_precision.csv: cùng bảng, giữ nguyên độ chính xác lưu trong metrics.json.",
        "- sources.csv: nguồn artifact, split, run ID và SHA-256 cho từng kết quả.",
        "- all_metrics.csv: tất cả metric/k đã lưu cho các kết quả được chọn.", "",
        "Chỉ chọn status SUCCESS ở cấp dataset và lấy lượt thành công mới nhất theo thời điểm bắt đầu session.",
        "Kết quả dataset hoàn tất trong session bị dừng vẫn được giữ. Không lấy điểm từ bài báo hoặc smoke test.",
        "Ô chưa có kết quả thành công dùng ký tự _. GenQ chưa có dataset đánh giá hoàn tất.",
        "MS MARCO dùng dev; các dataset còn lại dùng test. CQADupStack là trung bình đều 12 subsets theo runner.",
        "Dòng Avg. Performance vs. BM25 để _ do chưa đủ dữ liệu cho mức trung bình toàn benchmark như bảng mẫu.",
        "Điểm phản ánh code/cấu hình đã chạy; không khẳng định trùng cấu hình hay điểm của bài báo.", "",
        f"Tổng kết quả model/dataset: {len(selected)}.", "", "| Model | Dataset thành công |", "|---|---:|",
        *[f"| {model} | {counts[model]} |" for model in MODELS],
    ]
    if warnings:
        readme.extend(["", "Artifact bị loại vì không hợp lệ:", *warnings])
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    # Verify shape, placeholder policy, and that each populated score has provenance.
    with (output / "ndcg_at_10.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.reader(stream))
    assert len(table) == len(DATASETS) + 2
    assert all(len(row) == len(MODELS) + 1 for row in table)
    assert all(cell != "" for row in table for cell in row)
    assert sum(cell != "_" for row in table[1:-1] for cell in row[1:]) == len(selected)
    assert len(sources) - 1 == len(selected)
    print(json.dumps({"directory": str(output), "successful_model_dataset_results": len(selected),
                      "counts": dict(counts), "metric_values": len(all_metrics) - 1, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNS / "reports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    export(parser.parse_args().output.resolve())
