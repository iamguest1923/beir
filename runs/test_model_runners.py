"""Small numerical/contract checks; do not substitute these for full dataset runs."""
import importlib.util
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS.parent))
sys.path.insert(0, str(RUNS))
from _shared.adapters import ConcreteSparseSearch, generated_queries, deepct_term_weights
from _shared.common import validate_results, score_and_save, reuse_completed, file_sha256
from _shared.external import parse_colbert_ranking


class RunnerTests(unittest.TestCase):
    def test_native_bm25_retains_title_only_and_empty_documents(self):
        from _shared.adapters import bm25_module
        module = bm25_module()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "KEEP_INDEX", True):
            output = Path(tmp)
            corpus = {"title-only": {"title": "quasar", "text": ""},
                      "body": {"title": "", "text": "planet"},
                      "empty": {"title": "", "text": ""}}
            results = module.AnseriniBM25Search(output).search(
                corpus, {"q1": "quasar", "q2": "planet"}, 3)
            self.assertIn("title-only", results["q1"])
            self.assertIn("body", results["q2"])
            searcher = module.load_backend()(str(output / "index"))
            try:
                self.assertEqual(searcher.num_docs, 3)
            finally:
                searcher.close()

    def test_cross_encoder_import_on_python39(self):
        from beir.reranking.models import CrossEncoder
        self.assertTrue(callable(CrossEncoder))

    def test_deepct_merges_wordpieces_and_keeps_first_piece_weight(self):
        value = deepct_term_weights(["[CLS]", "run", "##ning", "run", "[SEP]"],
                                   [0, 0.2, 0.9, 0.1, 0], keep_all_terms=False)
        self.assertEqual(value, {"running": 20, "run": 10})
        self.assertEqual(deepct_term_weights(["a"], [-1.0], keep_all_terms=True), {"a": 1})

    def test_generation_passes_attention_mask(self):
        import torch
        from beir.generation.models import QGenModel
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": torch.tensor([[1, 0]]), "attention_mask": torch.tensor([[1, 0]])}

            def batch_decode(self, *args, **kwargs):
                return ["question"]
        class Model:
            def generate(self, **kwargs):
                self.mask = kwargs["attention_mask"]
                return [[1]]
        model = QGenModel.__new__(QGenModel)
        model.model, model.tokenizer, model.device, model.gen_prefix = Model(), Tokenizer(), "cpu", ""
        model.generate([{"title": "", "text": "a"}], 1, 10, 8, 0.95)
        self.assertEqual(model.model.mask.tolist(), [[1, 0]])

    def test_completed_result_cache_rejects_changed_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            previous = base / "results/old/nfcorpus"
            previous.mkdir(parents=True)
            (previous / "config.json").write_text(json.dumps({"cache_signature": "same"}))
            (previous / "status.json").write_text(json.dumps({"status": "SUCCESS"}))
            (previous / "metrics.json").write_text(json.dumps({"ndcg": {"NDCG@10": 0.3}}))
            (previous / "ranking.run.trec").write_text("q Q0 d 1 1.0 test\n")
            hashes = {"corpus.jsonl": "same-corpus"}
            meta = {"sha256": hashes, "artifact_sha256": {
                name: file_sha256(previous / name) for name in ("metrics.json", "ranking.run.trec")
            }}
            (previous / "data.json").write_text(json.dumps(meta))
            output = base / "results/new/nfcorpus"
            output.mkdir(parents=True)
            self.assertIsNotNone(reuse_completed(base, output, "nfcorpus", "same", hashes))
            (previous / "ranking.run.trec").write_text("changed")
            other = base / "results/third/nfcorpus"
            other.mkdir(parents=True)
            self.assertIsNone(reuse_completed(base, other, "nfcorpus", "same", hashes))

    def test_eight_entrypoints_use_only_requested_datasets(self):
        methods = {
            "DeepCT": "evaluate_deepct.py", "SPARTA": "evaluate_sparta.py",
            "docT5query": "evaluate_doct5query.py", "DPR": "evaluate_dpr.py",
            "TAS-B": "evaluate_tasb.py", "GenQ": "evaluate_genq.py",
            "ColBERT": "evaluate_colbert.py", "BM25+CE": "evaluate_bm25_ce.py",
        }
        for name, filename in methods.items():
            settings = runpy.run_path(str(RUNS / name / filename))["SETTINGS"]
            self.assertEqual(settings["datasets"], ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid"])
            self.assertEqual(settings["method"], name)

    def test_sparse_search_handles_small_corpus_and_known_scores(self):
        class Model:
            def encode_corpus(self, corpus, **kwargs):
                return csr_matrix([[3.0, 0.0], [0.0, 2.0]])

            def encode_query(self, query):
                return [0] if query == "alpha" else [1]
        corpus = {"d1": {"title": "", "text": "alpha"}, "d2": {"title": "", "text": "beta"}}
        results = ConcreteSparseSearch(Model()).search(corpus, {"q1": "alpha", "q2": "beta"}, 1000, "dot")
        self.assertEqual(results["q1"]["d1"], 3.0)
        self.assertEqual(results["q2"]["d2"], 2.0)

    def test_colbert_maps_integer_ids_and_preserves_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ranking.tsv"
            p.write_text("0\t0\t1\n0\t1\t2\n1\t1\t1\n")
            results = parse_colbert_ranking(p, ["q1", "d2"], ["q1", "q2"], 100)
            self.assertEqual(results, {"q1": {"d2": -2.0}, "q2": {"d2": -1.0}})
            p.write_text("0\t0\t1\n0\t0\t2\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_colbert_ranking(p, ["d1"], ["q1"], 100)

    def test_query_cache_reuses_chunks_and_invalidates_settings(self):
        from types import SimpleNamespace
        class Model:
            calls = 0
            device = "cpu"
            tokenizer = SimpleNamespace(model_max_length=512)
            model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fixed"), to=lambda device: None, eval=lambda: None)

            def generate(self, corpus, ques_per_passage, **kwargs):
                self.calls += 1
                return ["generated query"] * len(corpus) * ques_per_passage
        settings = runpy.run_path(str(RUNS / "GenQ/evaluate_genq.py"))["SETTINGS"].copy()
        settings.update(generation_batch_size=1, questions_per_passage=2)
        corpus = {"d1": {"title": "", "text": "alpha"}, "d2": {"title": "", "text": "beta"}}
        with tempfile.TemporaryDirectory() as tmp:
            context = {"cache": Path(tmp) / "cache", "output": Path(tmp) / "output",
                       "hashes": {"corpus.jsonl": "fixed"}}
            model = Model()
            first, _, _ = generated_queries(corpus, context, settings, model)
            second, _, _ = generated_queries(corpus, context, settings, model)
            self.assertEqual(first, second)
            self.assertEqual(model.calls, 2)
            settings["generation_top_k"] += 1
            generated_queries(corpus, context, settings, model)
            self.assertEqual(model.calls, 4)

    def test_metric_output_uses_real_beir_evaluator(self):
        corpus = {"d1": {}, "d2": {}}
        queries = {"q1": "", "q2": ""}
        results = {"q1": {"d1": 2.0, "d2": 1.0}, "q2": {"d2": 2.0, "d1": 1.0}}
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            metrics = score_and_save(Path(tmp), results, corpus, queries, qrels, [10, 100])
            self.assertEqual(metrics["ndcg"]["NDCG@10"], 1.0)
            self.assertEqual(len((Path(tmp) / "ranking.run.trec").read_text().splitlines()), 4)
        with self.assertRaises(ValueError):
            validate_results({"q1": {"d1": np.nan}}, corpus, {"q1": ""})


if __name__ == "__main__":
    unittest.main()
