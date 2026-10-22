"""Preprocessing, recovery and real BEIR metric contracts; no model downloads."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS.parent))
from _shared.multilingual_dense import SPECS, DATASETS, NormalizedEncoder, prepare_texts, fetch_snapshot
from _shared.common import score_and_save
from run_multilingual_models import METHODS, collect, recovery


class MultilingualTests(unittest.TestCase):
    def test_e5_prefixes_both_languages_and_distinguishes_roles(self):
        spec = SPECS["Multilingual-E5-Large"]
        self.assertEqual(prepare_texts(["Hà Nội", "English text"], spec, "query", "uit-viquad"),
                         ["query: Hà Nội", "query: English text"])
        self.assertEqual(prepare_texts(["body"], spec, "passage", "nfcorpus"), ["passage: body"])
        self.assertEqual(prepare_texts(["body"], SPECS["BGE-M3"], "query", "nfcorpus"), ["body"])

    def test_bkai_segments_vietnamese_but_does_not_rewrite_english(self):
        segment = lambda text: text.replace("sinh viên", "sinh_viên")
        spec = SPECS["Vietnamese-BiEncoder"]
        self.assertEqual(prepare_texts(["sinh viên"], spec, "query", "uit-viquad", segment), ["sinh_viên"])
        self.assertEqual(prepare_texts(["sinh viên"], spec, "passage", "nfcorpus", segment), ["sinh viên"])
        from pyvi import ViTokenizer
        self.assertIn("sinh_viên", ViTokenizer.tokenize("sinh viên học đại học"))

    def test_corpus_encoder_preserves_title_and_applies_prefix_once(self):
        class Model:
            def encode(self, texts, **kwargs):
                self.texts, self.options = texts, kwargs
                return np.array([[1., 0.] for _ in texts])
        m = Model()
        enc = NormalizedEncoder(m, SPECS["Multilingual-E5-Large"], "nfcorpus", 2, Path("unused"))
        enc.encode_corpus([{"title": "Title", "text": "Body"}], batch_size=2)
        self.assertEqual(m.texts, ["passage: Title Body"])
        self.assertEqual(m.options["prompt"], "")
        self.assertTrue(m.options["normalize_embeddings"])

    def test_cuda_oom_reduces_only_batch_and_logs_recovery(self):
        class Model:
            def encode(self, texts, batch_size, **kwargs):
                if batch_size > 2:
                    raise RuntimeError("CUDA out of memory")
                return np.array([[1., 0.] for _ in texts])
        with tempfile.TemporaryDirectory() as tmp:
            enc = NormalizedEncoder(Model(), SPECS["BGE-M3"], "nfcorpus", 8, Path(tmp)/"recovery.json")
            value = enc.encode_queries(["a", "b"], batch_size=8)
            self.assertEqual(value.shape, (2, 2))
            self.assertEqual(enc.effective_batch_size, 2)
            self.assertEqual(len(json.loads((Path(tmp)/"recovery.json").read_text())), 2)
            self.assertEqual(enc.spec["max_length"], 512)

    def test_non_oom_is_not_hidden(self):
        class Model:
            def encode(self, *args, **kwargs):
                raise ValueError("wrong tokenizer")
        enc = NormalizedEncoder(Model(), SPECS["BGE-M3"], "nfcorpus", 2, Path("unused"))
        with self.assertRaisesRegex(ValueError, "wrong tokenizer"):
            enc.encode_queries(["query"])

    def test_snapshot_download_includes_phobert_bpe_and_only_one_weight_format(self):
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download", return_value="/tmp/model") as download:
            api.return_value.list_repo_files.return_value = ["model.safetensors", "pytorch_model.bin", "bpe.codes",
                "vocab.txt", "modules.json", "1_Pooling/config.json", "onnx/model.onnx", "pipeline.py"]
            fetch_snapshot(SPECS["Vietnamese-BiEncoder"])
            chosen = download.call_args.kwargs["allow_patterns"]
            self.assertIn("bpe.codes", chosen)
            self.assertIn("model.safetensors", chosen)
            self.assertNotIn("pytorch_model.bin", chosen)
            self.assertNotIn("pipeline.py", chosen)
            self.assertNotIn("onnx/model.onnx", chosen)

    def test_normalized_wrapper_runs_exact_search_and_real_metrics(self):
        from beir.retrieval.search.dense import DenseRetrievalExactSearch
        from beir.retrieval.evaluation import EvaluateRetrieval
        import torch
        class Model:
            def encode(self, texts, **kwargs):
                return torch.tensor([[1., 0.] if "cat" in t else [0., 1.] for t in texts])
        with tempfile.TemporaryDirectory() as tmp:
            corpus = {"d1": {"title": "", "text": "cat"}, "d2": {"title": "", "text": "dog"}}
            queries = {"q1": "cat", "q2": "dog"}
            enc = NormalizedEncoder(Model(), SPECS["Multilingual-E5-Large"], "nfcorpus", 2, Path(tmp)/"recovery.json")
            retriever = EvaluateRetrieval(DenseRetrievalExactSearch(enc, batch_size=2), k_values=[1, 10, 100], score_function="dot")
            results = retriever.retrieve(corpus, queries)
            metrics = score_and_save(Path(tmp), results, corpus, queries, {"q1": {"d1": 1}, "q2": {"d2": 1}}, [1, 10, 100])
            self.assertEqual(metrics["ndcg"]["NDCG@1"], 1)

    def test_access_wait_does_not_consume_transient_retry_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = {"failed_attempts": 0, "batch_size": 2}
            attempt = {"directory": tmp, "datasets": DATASETS, "started": 0}
            recovery(job, attempt, 3, 10)
            self.assertEqual(job["status"], "WAITING_ACCESS")
            self.assertEqual(job["failed_attempts"], 0)

    def test_missing_artifacts_are_never_exported_as_success(self):
        jobs = [{"model": model, "status": "SUCCESS", "attempts": []} for model in METHODS]
        with tempfile.TemporaryDirectory() as tmp:
            rows = collect(Path(tmp), jobs, {}, {})
            self.assertEqual(len(rows), 36)
            self.assertTrue(all(row["status"] == "MISSING_RESULT" for row in rows))

    def test_worker_recovery_retries_only_failed_dataset_and_reduces_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [("nfcorpus", {"status": "SUCCESS"}),
                                ("scifact", {"status": "FAILED", "error": "CUDA out of memory"})]:
                (root/name).mkdir()
                (root/name/"status.json").write_text(json.dumps(value))
            job = {"failed_attempts": 0, "batch_size": 2}
            attempt = {"directory": tmp, "datasets": ["nfcorpus", "scifact"], "started": 0}
            recovery(job, attempt, 1, 10)
            self.assertEqual(job["remaining"], ["scifact"])
            self.assertEqual(job["batch_size"], 1)
            self.assertEqual(job["status"], "RETRYING")


if __name__ == "__main__":
    unittest.main()
