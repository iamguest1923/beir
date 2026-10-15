"""Conversion regression tests; no network access or model downloads."""
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import Benchmark, convert_qa, convert_retrieval, prepare, sha256


def jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")


def test_viquad_context_dedup_unanswerable_and_split_ids(tmp_path):
    source = tmp_path / "source"
    row = {"id": "1", "question": "Thủ đô?", "context": "Hà Nội là thủ đô.",
           "answers": {"text": "Hà Nội", "answer_start": 0}}
    jsonl(source / "train.json", [row])  # Real mirror uses JSONL in .json.
    jsonl(source / "test.jsonl", [row, {**row, "id": "2", "context": "Hà  Nội là thủ đô."},
                                    {**row, "id": "3", "answers": None, "is_impossible": True}])
    info = prepare("uit-viquad", tmp_path / "out", source_dir=source)
    assert info["documents"] == 1
    assert info["queries"] == 3
    assert info["statistics"]["unanswerable_queries_skipped"] == 1
    assert info["query_text_overlap_between_splits"] == {"test/train": 1}
    for relative, digest in info["artifact_sha256"].items():
        assert sha256(tmp_path / "out/uit-viquad" / relative) == digest
    with pytest.raises(FileExistsError):
        prepare("uit-viquad", tmp_path / "out", source_dir=source)


def test_squad_nested_and_bad_answer(tmp_path):
    path = tmp_path / "test.json"
    data = {"data": [{"title": "Wiki", "paragraphs": [{"context": "Hà Nội", "qas": [
        {"id": "q", "question": "Ở đâu?", "answers": [{"text": "Hà Nội"}]}]}]}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    b = Benchmark()
    convert_qa("uit-viquad", tmp_path, b)
    b.validate()
    data["data"][0]["paragraphs"][0]["qas"][0]["answers"] = [{"text": "Huế"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Answer not found"):
        convert_qa("uit-viquad", tmp_path, Benchmark())


def test_health_csv_with_quoted_multiline_answer(tmp_path):
    for split in ["train", "val", "test"]:
        with (tmp_path / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "question", "answer", "link"])
            writer.writeheader()
            writer.writerow(dict(id=1, question="Câu hỏi?", answer="Dòng 1,\nDòng 2", link="url"))
    b = Benchmark()
    convert_qa("vihealthqa", tmp_path, b)
    b.validate()
    assert len(b.corpus) == 1 and len(b.queries) == 3
    assert set(b.qrels) == {"train", "dev", "test"}
    assert next(iter(b.corpus.values()))["title"] == ""


def test_csconda_parquet_answer_and_context_override(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"question": "Hỏi?", "answer": "Trả lời", "context": "Tài liệu"},
        {"question": "Hỏi nữa?", "answer": "Trả  lời", "context": "Tài liệu"},
    ]), tmp_path / "test-00000.parquet")
    b = Benchmark()
    convert_qa("csconda", tmp_path, b)
    assert len(b.corpus) == 1 and len(b.qrels["test"]) == 2
    assert next(iter(b.corpus.values()))["text"] == "Trả lời"
    b = Benchmark()
    convert_qa("csconda", tmp_path, b, context_field="context")
    assert next(iter(b.corpus.values()))["text"] == "Tài liệu"


def test_native_retrieval_multiple_positives_duplicate_ids_and_title_only(tmp_path):
    jsonl(tmp_path / "corpus.jsonl", [{"_id": "law+1", "title": "Điều 1", "text": ""},
                                     {"_id": "law+2", "text": "Điều 2"}])
    jsonl(tmp_path / "queries.jsonl", [{"_id": "q", "text": "Câu hỏi?"}] * 2)
    jsonl(tmp_path / "qrels/test.jsonl", [
        {"query-id": "q", "corpus-id": d, "score": 1} for d in ["law+1", "law+2", "law+1"]])
    b = Benchmark()
    convert_retrieval(tmp_path, b)
    b.validate()
    assert len(b.qrels["test"]) == 2
    assert b.stats["duplicate_queries"] == b.stats["duplicate_qrels"] == 1
    assert b.stats["title_only_documents"] == 1
    b.relevance("test", "q", "missing")
    with pytest.raises(ValueError, match="Unknown qrel ID"):
        b.validate()


def test_local_beir_vihealthqa_preserves_original_relevance(tmp_path):
    source = tmp_path / "source"
    jsonl(source / "corpus.jsonl", [{"_id": "d", "text": "Passage"}])
    jsonl(source / "queries.jsonl", [{"_id": "q", "text": "Query"}])
    (source / "qrels").mkdir()
    (source / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\td\t2\n")
    info = prepare("vihealthqa", tmp_path / "out", source_dir=source)
    assert info["conversion"] == "preserve_source_ids_and_relevance"
    assert (tmp_path / "out/vihealthqa/qrels/test.tsv").read_text().endswith("q\td\t2\n")


def test_conflicting_ids_invalid_scores_and_no_positives():
    b = Benchmark()
    b.document("d", "Text")
    b.query("q", "Question")
    with pytest.raises(ValueError, match="Conflicting query"):
        b.query("q", "Other")
    for score in [1.5, True, -2]:
        with pytest.raises(ValueError, match="Invalid relevance"):
            b.relevance("test", "q", "d", score)
    b.relevance("test", "q", "d", 0)
    with pytest.raises(ValueError, match="without positive"):
        b.validate()
