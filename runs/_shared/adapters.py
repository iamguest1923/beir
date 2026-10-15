from __future__ import annotations

import functools
import gc
import importlib.util
import json
import logging
import random
from pathlib import Path

import numpy as np

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch

from .common import ROOT, execute, file_sha256, signature, validate_results, write_json

LOG = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def bm25_module():
    path = ROOT / "runs/BM25/evaluate_bm25.py"
    spec = importlib.util.spec_from_file_location("beir_local_bm25_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_bm25(settings):
    module = bm25_module()
    module.K1 = settings.get("bm25_k1", 0.9)
    module.B = settings.get("bm25_b", 0.4)
    module.FIELDS = settings.get("bm25_fields", {"contents": 1.0, "title": 1.0})
    module.THREADS = settings.get("cpu_threads", 4)
    module.BATCH_SIZE = settings.get("query_batch_size", 32)
    module.KEEP_INDEX = settings.get("keep_index", False)
    return module


def bm25_retrieve(corpus, queries, context, settings, tag):
    """Reuse the already tested local BM25 runner, including index lifecycle."""
    module = configure_bm25(settings)
    output = context["output"] / tag
    output.mkdir(parents=True, exist_ok=True)
    config = {"hashes": context["hashes"], "tag": tag, "k1": module.K1, "b": module.B,
              "fields": module.FIELDS, "depth": settings.get("candidate_k", max(settings["k_values"])),
              "pyserini": module.PYSERINI_VERSION}
    config["backend_sha256"] = file_sha256(module.__file__)
    config["empty_generator_sha256"] = file_sha256(ROOT / "runs/BM25/java/BeirMultifieldGenerator.java")
    # Use cache only for this exact data/preprocessing configuration.
    cache = context["cache"] / (tag + "-" + signature(config))
    file = cache / "results.json"
    if settings.get("reuse_cache", True) and file.is_file():
        results = json.loads(file.read_text())
        validate_results(results, corpus, queries)
        LOG.info("Reusing %s candidates: %s", tag, file)
        return results
    results = module.AnseriniBM25Search(output).search(corpus, queries, config["depth"])
    validate_results(results, corpus, queries)
    write_json(cache / "config.json", config)
    write_json(file, results)
    return results


class Adapter:
    def __init__(self, settings, base):
        self.s = settings
        self.base = base

    def preflight(self):
        import torch
        torch.set_num_threads(self.s.get("cpu_threads", 4))
        torch.backends.cuda.matmul.allow_tf32 = False
        if self.s.get("device") == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")


class DenseAdapter(Adapter):
    encoder = None

    def retrieve(self, corpus, queries, context):
        from beir.retrieval.models import SentenceBERT
        from beir.retrieval.search.dense import DenseRetrievalExactSearch
        if self.encoder is None:
            name = self.s["model"]
            if isinstance(name, list):
                name = tuple(name)
            self.encoder = SentenceBERT(name, max_length=self.s["max_length"],
                                        sep=self.s.get("separator", " "), device=self.s.get("device"))
            self.encoder.q_model.float()
            self.encoder.doc_model.float()
        write_json(context["output"] / "model.json", {
            "models": self.s["model"], "device": str(self.encoder.q_model.device),
            "query_revision": getattr(self.encoder.q_model[0].auto_model.config, "_commit_hash", None),
            "document_revision": getattr(self.encoder.doc_model[0].auto_model.config, "_commit_hash", None),
        })
        search = DenseRetrievalExactSearch(self.encoder, batch_size=self.s["batch_size"],
                                           corpus_chunk_size=self.s["corpus_chunk_size"])
        return EvaluateRetrieval(search, k_values=self.s["k_values"], score_function="dot").retrieve(corpus, queries)


class BM25Adapter(Adapter):
    def preflight(self):
        super().preflight()
        configure_bm25(self.s).load_backend()

    def retrieve(self, corpus, queries, context):
        return bm25_retrieve(corpus, queries, context, self.s, "bm25")


class ConcreteSparseSearch(SparseSearch):
    # Upstream BaseSearch added abstract methods that SparseSearch does not implement.
    def encode(self, *args, **kwargs):
        raise NotImplementedError("Use retrieve() for sparse retrieval")

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError("Use retrieve() for sparse retrieval")

    def search(self, corpus, queries, top_k, score_function, **kwargs):
        return super().search(corpus, queries, min(top_k, len(corpus)), score_function, **kwargs)


class CachedSparseModel:
    def __init__(self, model, path, reuse):
        self.model, self.path, self.reuse = model, path, reuse

    def encode_query(self, query):
        return self.model.encode_query(query)

    def encode_corpus(self, corpus, **kwargs):
        from scipy.sparse import load_npz, save_npz
        if self.reuse and self.path.is_file():
            LOG.info("Reusing SPARTA sparse matrix: %s", self.path)
            return load_npz(self.path)
        value = self.model.encode_corpus(corpus, **kwargs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".tmp.npz")
        save_npz(temporary, value)
        temporary.replace(self.path)
        return value


class SpartaAdapter(Adapter):
    encoder = None

    def retrieve(self, corpus, queries, context):
        import torch
        from beir.retrieval.models import SPARTA
        if self.encoder is None:
            self.encoder = SPARTA(self.s["model"], max_length=self.s["max_length"],
                                  sparse_vector_dim=self.s["sparse_dimension"])
            if self.s.get("device"):
                self.encoder.device = self.s["device"]
                self.encoder.model.to(self.encoder.device)
            with torch.no_grad():
                self.encoder.bert_input_embeddings = self.encoder._bert_input_embeddings().detach()
        meta = {"model": self.s["model"], "max_length": self.s["max_length"],
                "sparse_dimension": self.s["sparse_dimension"], "corpus_sha256": context["hashes"]["corpus.jsonl"],
                "revision": getattr(self.encoder.model.config, "_commit_hash", None)}
        write_json(context["output"] / "model.json", meta)
        cached = CachedSparseModel(self.encoder, context["cache"] / signature(meta) / "corpus.npz",
                                   self.s.get("reuse_cache", True))
        search = ConcreteSparseSearch(cached, batch_size=self.s["batch_size"])
        return EvaluateRetrieval(search, k_values=self.s["k_values"]).retrieve(corpus, queries)


def generated_queries(corpus, context, settings, model=None):
    """Atomic per-chunk cache; deterministic seeds make interrupted generation resumable."""
    import torch
    from beir.generation.models import QGenModel
    from transformers import set_seed

    if model is None:
        model = QGenModel(settings["generation_model"], device=settings.get("device"))
    model.model.to(model.device)
    model.model.eval()
    model.tokenizer.model_max_length = settings["max_length"]
    config = {key: settings[key] for key in (
        "generation_model", "questions_per_passage", "generation_batch_size", "generation_max_length",
        "generation_top_k", "generation_top_p", "max_length", "seed",
    )}
    config["revision"] = getattr(model.model.config, "_commit_hash", None)
    config["selected_corpus"] = signature(list(corpus))
    config["corpus_sha256"] = context["hashes"]["corpus.jsonl"]
    cache = context["cache"] / ("generated-" + signature(config))
    write_json(cache / "config.json", config)
    doc_ids = list(corpus)
    generated = {}
    for start in range(0, len(doc_ids), settings["generation_batch_size"]):
        ids = doc_ids[start:start + settings["generation_batch_size"]]
        file = cache / f"chunk-{start:09d}.json"
        if settings.get("reuse_cache", True) and file.is_file():
            chunk = json.loads(file.read_text())
            if set(chunk) != set(ids) or any(len(value) != settings["questions_per_passage"] for value in chunk.values()):
                raise ValueError(f"Invalid generated-query cache: {file}")
        else:
            set_seed(settings["seed"] + start)
            values = model.generate(
                [corpus[doc] for doc in ids], ques_per_passage=settings["questions_per_passage"],
                max_length=settings["generation_max_length"], top_k=settings["generation_top_k"],
                top_p=settings["generation_top_p"],
            )
            count = settings["questions_per_passage"]
            if len(values) != len(ids) * count:
                raise ValueError("Generator returned incorrect number of queries")
            chunk = {doc: values[i * count:(i + 1) * count] for i, doc in enumerate(ids)}
            write_json(file, chunk)
        generated.update(chunk)
        if start % (settings["generation_batch_size"] * 32) == 0 or start + len(ids) == len(doc_ids):
            LOG.info("Generated/reused questions for %d/%d documents", start + len(ids), len(doc_ids))
    write_json(context["output"] / "generation.json", {**config, "cache": str(cache), "documents": len(doc_ids)})
    return generated, model, signature(config)


class DocT5Adapter(Adapter):
    generator = None

    def preflight(self):
        super().preflight()
        configure_bm25(self.s).load_backend()

    def retrieve(self, corpus, queries, context):
        generated, self.generator, key = generated_queries(corpus, context, self.s, self.generator)
        expanded = {doc: {"title": value["title"],
                          "text": value["text"] + " " + " ".join(generated[doc])}
                    for doc, value in corpus.items()}
        return bm25_retrieve(expanded, queries, context, self.s, "doct5-" + key)


def deepct_term_weights(tokens, logits, scale=100, smoothing="none", keep_all_terms=True):
    """Match the original DeepCT wordpiece merging/rounding (first piece's weight)."""
    words, weights = [], []
    for token, weight in zip(tokens, logits):
        if token.startswith("##") and words:
            words[-1] += token[2:]
        else:
            words.append(token)
            weights.append(float(weight))
    result = {}
    for token, weight in zip(words, weights):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        weight = max(0.0, weight)
        if smoothing == "sqrt":
            weight = np.sqrt(weight)
        count = int(np.round(scale * weight))
        if count < 1:
            if not keep_all_terms:
                continue
            count = 1
        result[token] = max(count, result.get(token, 0))
    return result


class PytorchDeepCTAdapter(Adapter):
    encoder = None

    def preflight(self):
        super().preflight()
        configure_bm25(self.s).load_backend()

    def retrieve(self, corpus, queries, context):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        if self.encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.s["model"])
            self.device = self.s.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder = AutoModelForTokenClassification.from_pretrained(self.s["model"]).float().eval().to(self.device)
        config = {"model": self.s["model"], "revision": getattr(self.encoder.config, "_commit_hash", None),
                  "corpus_sha256": context["hashes"]["corpus.jsonl"], "max_length": self.s["max_length"],
                  "scale": self.s["scaling"], "smoothing": self.s["smoothing"],
                  "keep_all_terms": self.s["keep_all_terms"]}
        cache = context["cache"] / ("deepct-pytorch-" + signature(config))
        ids = list(corpus)
        expanded = {}
        for start in range(0, len(ids), self.s["batch_size"]):
            batch = ids[start:start + self.s["batch_size"]]
            path = cache / f"chunk-{start:09d}.json"
            if self.s.get("reuse_cache", True) and path.is_file():
                chunk = json.loads(path.read_text())
                if set(chunk) != set(batch):
                    raise ValueError("DeepCT cache does not match document IDs")
            else:
                texts = [(corpus[doc]["title"] + " " + corpus[doc]["text"]).strip() or "." for doc in batch]
                tokens = self.tokenizer(texts, padding=True, truncation=True,
                                        max_length=self.s["max_length"], return_tensors="pt")
                with torch.no_grad():
                    logits = self.encoder(**{key: value.to(self.device) for key, value in tokens.items()}).logits.squeeze(-1).cpu().numpy()
                chunk = {}
                for idx, doc in enumerate(batch):
                    pieces = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][idx])
                    counts = deepct_term_weights(pieces, logits[idx], self.s["scaling"],
                                                self.s["smoothing"], self.s["keep_all_terms"])
                    chunk[doc] = {"title": "", "text": " ".join(word for word, count in counts.items() for _ in range(count))}
                write_json(path, chunk)
            expanded.update(chunk)
            if start % (self.s["batch_size"] * 32) == 0 or start + len(batch) == len(ids):
                LOG.info("DeepCT transformed/reused %d/%d documents", start + len(batch), len(ids))
        write_json(context["output"] / "model.json", {**config, "backend": "pytorch-converted-original-checkpoint",
                   "source": "https://huggingface.co/macavaney/deepct", "device": self.device})
        return bm25_retrieve(expanded, queries, context, self.s, "deepct-" + signature(config))


class CEAdapter(Adapter):
    reranker = None

    def preflight(self):
        super().preflight()
        configure_bm25(self.s).load_backend()

    def retrieve(self, corpus, queries, context):
        from beir.reranking import Rerank
        from beir.reranking.models import CrossEncoder
        candidates = bm25_retrieve(corpus, queries, context, self.s, "bm25-candidates")
        if self.reranker is None:
            model = CrossEncoder(self.s["model"], max_length=self.s["max_length"], device=self.s.get("device"))
            model.model.model.float()
            self.reranker = Rerank(model, batch_size=self.s["batch_size"])
        qids = list(queries)
        results = {}
        # Bound pair construction memory instead of materializing every query x 100 at once.
        for start in range(0, len(qids), self.s["query_batch_size"]):
            ids = qids[start:start + self.s["query_batch_size"]]
            results.update(self.reranker.rerank(corpus, {qid: queries[qid] for qid in ids},
                           {qid: candidates[qid] for qid in ids}, top_k=self.s["candidate_k"]))
        write_json(context["output"] / "model.json", {
            "model": self.s["model"], "candidate_depth": self.s["candidate_k"],
            "revision": getattr(self.reranker.cross_encoder.model.model.config, "_commit_hash", None),
            "note": "Recall above candidate depth is limited to the reranked candidates",
        })
        return results


class GenQAdapter(Adapter):
    generator = None

    def retrieve(self, corpus, queries, context):
        import torch
        from sentence_transformers import InputExample, SentenceTransformer, datasets, losses, util
        from transformers import set_seed
        from huggingface_hub import snapshot_download
        from beir.retrieval.models import SentenceBERT
        from beir.retrieval.search.dense import DenseRetrievalExactSearch

        ids = sorted(corpus)
        if len(ids) > self.s["max_training_documents"]:
            ids = random.Random(self.s["seed"]).sample(ids, self.s["max_training_documents"])
        selected = {doc: corpus[doc] for doc in ids}
        generated, self.generator, generation_key = generated_queries(selected, context, self.s, self.generator)
        # Avoid keeping a T5 model on GPU during retriever training.
        self.generator.model.cpu()
        torch.cuda.empty_cache()
        base_checkpoint = snapshot_download(self.s["model"])
        config = {key: self.s[key] for key in (
            "model", "train_batch_size", "epochs", "learning_rate", "seed", "max_length", "train_steps_per_epoch"
        )}
        config.update(generation=generation_key, loss="MultipleNegativesRankingLoss(dot,scale=1)",
                      base_revision=Path(base_checkpoint).name,
                      data_selection="seeded sample of sorted document IDs; no evaluation queries/qrels")
        checkpoint = context["cache"] / ("trained-" + signature(config))
        model_path = checkpoint / "model"
        if not (self.s.get("reuse_cache", True) and (checkpoint / "complete.json").is_file()):
            set_seed(self.s["seed"])
            samples = []
            for doc in ids:
                passage = (corpus[doc]["title"] + " " + corpus[doc]["text"]).strip()
                samples.extend(InputExample(texts=[q.strip(), passage])
                               for q in sorted(set(generated[doc])) if q.strip())
            if len(ids) < self.s["train_batch_size"]:
                raise ValueError("Training corpus is smaller than the no-duplicates training batch")
            model = SentenceTransformer(base_checkpoint, device=self.s.get("device"))
            model.max_seq_length = self.s["max_length"]
            model.float()
            loader = datasets.NoDuplicatesDataLoader(samples, batch_size=self.s["train_batch_size"])
            loss = losses.MultipleNegativesRankingLoss(model, scale=1.0, similarity_fct=util.dot_score)
            steps = self.s["train_steps_per_epoch"] or len(loader)
            checkpoint.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint / ("model-incomplete")
            if temporary.exists():
                import shutil
                shutil.rmtree(temporary)
            # old_fit is the maintained legacy training loop; BEIR TrainRetriever imports removed transformers.AdamW.
            model.old_fit(
                train_objectives=[(loader, loss)], epochs=self.s["epochs"], steps_per_epoch=steps,
                warmup_steps=int(steps * self.s["epochs"] * 0.1),
                optimizer_class=torch.optim.AdamW, optimizer_params={"lr": self.s["learning_rate"]},
                output_path=str(temporary), save_best_model=False, use_amp=False,
            )
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            temporary.replace(model_path)
            write_json(checkpoint / "complete.json", config)
            del model, loader, loss, samples
            gc.collect()
            torch.cuda.empty_cache()
        write_json(context["output"] / "training.json", {**config, "checkpoint": str(model_path),
                   "selected_documents": len(selected), "test_labels_used_for_training": False})
        encoder = SentenceBERT(str(model_path), max_length=self.s["max_length"], device=self.s.get("device"))
        search = DenseRetrievalExactSearch(encoder, batch_size=self.s["batch_size"],
                                           corpus_chunk_size=self.s["corpus_chunk_size"])
        return EvaluateRetrieval(search, k_values=self.s["k_values"], score_function="dot").retrieve(corpus, queries)


def build_adapter(settings, base):
    if settings["method"] in ("DPR", "TAS-B", "ANCE"):
        cls = DenseAdapter
    elif settings["method"] == "BM25":
        cls = BM25Adapter
    elif settings["method"] == "DeepCT" and settings.get("backend") == "pytorch-converted":
        cls = PytorchDeepCTAdapter
    else:
        from .external import DeepCTAdapter, ColBERTAdapter
        cls = {"SPARTA": SpartaAdapter, "docT5query": DocT5Adapter, "GenQ": GenQAdapter,
               "BM25+CE": CEAdapter, "DeepCT": DeepCTAdapter, "ColBERT": ColBERTAdapter}[settings["method"]]
    return cls(settings, base)
