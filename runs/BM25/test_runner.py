"""Offline adapter tests; real retrieval is checked with a dataset run."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import subprocess

spec = importlib.util.spec_from_file_location("bm25_runner", Path(__file__).with_name("evaluate_bm25.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.data = self.base / "data"
        (self.data / "qrels").mkdir(parents=True)
        for name, rows in {
            "corpus.jsonl": [
                {"_id": "doc-a", "title": "A", "text": "alpha"},
                {"_id": "doc-b", "title": "B", "text": "beta"},
            ],
            "queries.jsonl": [{"_id": qid, "text": "alpha"} for qid in ["q1", "q2"]],
        }.items():
            (self.data / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.data / "qrels/test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\tdoc-a\t1\nq2\tdoc-a\t1\n"
        )
        self.closed = False
        owner = self

        class FakeSearcher:
            num_docs = 2

            def __init__(self, index):
                self.index = index

            def set_bm25(self, k1, b):
                owner.assertEqual((k1, b), (0.9, 0.4))

            def batch_search(self, queries, qids, k, threads, fields):
                owner.assertEqual(fields, {"contents": 1.0, "title": 1.0})
                return {qid: [SimpleNamespace(docid=doc, score=score) for doc, score in
                              [(qid, 99.0), ("doc-a", 2.0), ("doc-b", 1.0)]] for qid in qids}

            def close(self):
                owner.closed = True

        self.searcher = FakeSearcher

    def index(self, command, **kwargs):
        self.assertEqual(command[1:3], ["-m", "pyserini.index"])
        Path(command[command.index("-index") + 1]).mkdir()
        return subprocess.CompletedProcess(command, 0)

    def evaluate(self, output):
        with patch.object(runner, "load_backend", return_value=self.searcher):
            with patch.object(runner.subprocess, "run", side_effect=self.index):
                return runner.evaluate_folder(self.data, output, "test")

    def test_beir_evaluation_and_output(self):
        output = self.base / "output"
        metrics = self.evaluate(output)
        self.assertEqual(metrics["ndcg"]["NDCG@10"], 1.0)
        self.assertEqual(metrics["recall"]["Recall@100"], 1.0)
        self.assertEqual(len((output / "ranking.run.trec").read_text().splitlines()), 4)
        self.assertFalse((output / "index_input").exists())
        self.assertFalse((output / "index").exists())
        self.assertTrue(self.closed)

    def test_index_count_mismatch_is_not_evaluated(self):
        self.searcher.num_docs = 1
        output = self.base / "output"
        with self.assertRaisesRegex(ValueError, "count differs"):
            self.evaluate(output)
        self.assertFalse((output / "metrics.json").exists())
        self.assertFalse((output / "index_input").exists())
        self.assertTrue((output / "index").exists())
        self.assertTrue(self.closed)

    def test_missing_queries_are_not_silently_scored(self):
        self.searcher.batch_search = lambda *args, **kwargs: {"q1": []}
        with self.assertRaisesRegex(ValueError, "query coverage"):
            self.evaluate(self.base / "output")
        self.assertTrue(self.closed)

    def test_indexing_failure_does_not_search(self):
        output = self.base / "output"
        with patch.object(runner, "load_backend", return_value=self.searcher):
            with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "index")):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.evaluate_folder(self.data, output, "test")
        self.assertFalse((output / "metrics.json").exists())
        self.assertFalse((output / "index_input").exists())


if __name__ == "__main__":
    unittest.main()
