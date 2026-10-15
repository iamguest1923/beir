"""Audit and dispatch regression checks without downloading model weights."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS.parent))
from complete_benchmark import digest, verify, worker
from _shared.common import score_and_save


class CompletionTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        (source / "qrels").mkdir(parents=True)
        (source / "corpus.jsonl").write_text('{"_id":"d","title":"","text":"body"}\n')
        (source / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n{"_id":"q2","text":"stopword"}\n')
        (source / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td\t1\nq2\td\t1\n")
        output = root / "result"
        output.mkdir()
        metrics = score_and_save(output, {"q1": {"d": 1.0}, "q2": {}}, {"d": {}},
                                 {"q1": "query", "q2": "stopword"}, {"q1": {"d": 1}, "q2": {"d": 1}}, [10, 100])
        hashes = {f: digest(source / f) for f in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]}
        for name, data in {
            "status.json": {"status": "SUCCESS", "split": "test", "NDCG@10": metrics["ndcg"]["NDCG@10"]},
            "config.json": {"dataset": "nfcorpus"},
            "data.json": {"documents": 1, "queries": 2, "path": str(source), "sha256": hashes},
        }.items():
            (output / name).write_text(json.dumps(data))
        info = {"path": str(source), "documents": {"d"}, "queries": {"q1", "q2"}, "sha256": hashes}
        return output, info

    def test_audit_restores_empty_queries_and_recomputes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, info = self.fixture(Path(tmp))
            record = verify(output, "nfcorpus", info, {})
            self.assertEqual(record["queries_with_zero_saved_hits"], ["q2"])
            self.assertEqual(json.loads((output / "metrics.json").read_text())["ndcg"]["NDCG@10"], 0.5)

    def test_audit_rejects_changed_run_even_with_cached_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, info = self.fixture(Path(tmp))
            memo = {}
            verify(output, "nfcorpus", info, memo)
            (output / "ranking.run.trec").write_text("q1 Q0 unknown 0 1.0 beir\n")
            with self.assertRaisesRegex(ValueError, "unknown query/document"):
                verify(output, "nfcorpus", info, memo)

    def test_dispatch_preserves_model_and_selects_only_missing_datasets(self):
        for model in ["BM25", "ANCE", "ColBERT", "GenQ"]:
            with patch("_shared.common.run", return_value=0) as run:
                self.assertEqual(worker(model, ["uit-viquad", "zalo-legal"]), 0)
                settings, entrypoint = run.call_args.args
                self.assertEqual(settings["datasets"], ["uit-viquad", "zalo-legal"])
                self.assertEqual(settings["method"], model)
                self.assertEqual(entrypoint.parent.name, model)
                if model == "GenQ":
                    self.assertEqual(settings["questions_per_passage"], 5)
                    self.assertEqual(settings["train_batch_size"], 32)
                if model == "ColBERT":
                    self.assertIn("zalo-legal", settings["partitions"])


if __name__ == "__main__":
    unittest.main()
