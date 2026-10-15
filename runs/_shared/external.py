"""Concrete adapters for the original DeepCT and ColBERT v1 codebases."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from beir import util

from .adapters import Adapter, bm25_retrieve, configure_bm25
from .common import RUNS, execute, file_sha256, signature, write_json

REPOSITORIES = {
    "DeepCT": ("https://github.com/Nthakur20/DeepCT.git", "ad4658845d41f35924d03c3d176cc3fea337b222"),
    "ColBERT": ("https://github.com/Nthakur20/beir-ColBERT.git", "91190882deac1792c78b3c33d51be9edaa9c6805"),
}


def runtime(settings, method):
    base = RUNS / method
    python = Path(settings.get("external_python") or base / "environment/bin/python").expanduser().resolve()
    repo = Path(settings.get("external_repo") or base / "vendor").expanduser().resolve()
    if not python.is_file() or not (repo / ".git").exists():
        raise RuntimeError(f"Prepare {method} first: python code/beir/runs/setup_external.py {method.lower()}")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != REPOSITORIES[method][1]:
        raise ValueError(f"{method} source must be pinned to {REPOSITORIES[method][1]}, found {commit}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_NUM_INTRAOP_THREADS"] = str(settings.get("cpu_threads", 4))
    env["TF_NUM_INTEROP_THREADS"] = "2"
    # pip-installed CUDA libraries live inside this isolated environment.
    library_dirs = sorted((python.parent.parent / "lib").glob("python*/site-packages/nvidia/*/lib"))
    if library_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*map(str, library_dirs), env.get("LD_LIBRARY_PATH", "")])
    return python, repo, env


def asset(url, folder):
    """Extract a checkpoint ZIP atomically and record the downloaded archive hash."""
    marker = folder / "asset.json"
    if marker.is_file():
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.with_suffix(".zip.part")
    util.download_url(url, str(archive))
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Invalid checkpoint ZIP: {archive}")
    temporary = folder.with_name(folder.name + "-extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                if not (temporary / info.filename).resolve().is_relative_to(temporary.resolve()):
                    raise ValueError("Unsafe checkpoint ZIP path")
            source.extractall(temporary)
        write_json(temporary / "asset.json", {"url": url, "zip_sha256": file_sha256(archive)})
        if folder.exists():
            shutil.rmtree(folder)
        temporary.replace(folder)
        archive.unlink()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return folder


class DeepCTAdapter(Adapter):
    def preflight(self):
        super().preflight()
        configure_bm25(self.s).load_backend()
        self.python, self.repo, self.env = runtime(self.s, "DeepCT")
        subprocess.run([str(self.python), "-c", "import tensorflow as tf; assert tf.__version__.startswith('2.15.'); from deepct import run_deepct"],
                       cwd=self.repo, env=self.env, check=True, timeout=120)

    def retrieve(self, corpus, queries, context):
        # The original Google Storage BERT ZIP now returns AccessDenied. Only the
        # config/vocabulary are needed: the DeepCT checkpoint contains BERT weights.
        from huggingface_hub import hf_hub_download
        bert_config = Path(hf_hub_download("google-bert/bert-base-uncased", "config.json"))
        vocab = Path(hf_hub_download("google-bert/bert-base-uncased", "vocab.txt"))
        trained = asset("http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/outputs/marco.zip",
                        self.base / "assets/deepct-msmarco")
        ckpt_index = next(trained.rglob("model.ckpt-65816.index"))
        config = {
            "corpus": str(context["data_path"] / "corpus.jsonl"),
            "corpus_sha256": context["hashes"]["corpus.jsonl"],
            "bert_config": str(bert_config), "vocab": str(vocab),
            "checkpoint": str(ckpt_index.with_suffix("")), "max_length": self.s["max_length"],
            "batch_size": self.s["batch_size"], "scaling": self.s["scaling"],
            "smoothing": self.s["smoothing"], "keep_all_terms": self.s["keep_all_terms"],
            "repository_commit": REPOSITORIES["DeepCT"][1],
            "checkpoint_asset": json.loads((trained / "asset.json").read_text()),
            "bert_config_sha256": file_sha256(bert_config), "vocab_sha256": file_sha256(vocab),
        }
        cache = context["cache"] / ("deepct-" + signature(config))
        cache.mkdir(parents=True, exist_ok=True)
        config["output"] = str(cache)
        write_json(cache / "config.json", config)
        transformed = cache / "deepct.jsonl"
        if not (self.s.get("reuse_cache", True) and (cache / "complete.json").is_file()):
            execute([self.python, Path(__file__).with_name("deepct_worker.py"), cache / "config.json"],
                    context["output"] / "deepct.log", cwd=self.repo, env=self.env)
            if not transformed.is_file():
                raise ValueError("DeepCT did not produce deepct.jsonl")
        expanded = {}
        with transformed.open() as stream:
            for line in stream:
                row = json.loads(line)
                doc = str(row["id"])
                if doc in expanded or doc not in corpus:
                    raise ValueError("DeepCT emitted duplicate or unknown document IDs")
                expanded[doc] = {"title": "", "text": row["contents"]}
        if set(expanded) != set(corpus):
            raise ValueError("DeepCT output does not cover every corpus document")
        write_json(cache / "complete.json", {"sha256": file_sha256(transformed)})
        write_json(context["output"] / "deepct.json", config)
        return bm25_retrieve(expanded, queries, context, self.s, "deepct-" + signature(config))


def parse_colbert_ranking(path, document_ids, query_ids, depth):
    """The v1 logger exports qid/pid/rank, so preserve rank with synthetic scores."""
    results = {qid: {} for qid in query_ids}
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                raise ValueError("Malformed ColBERT ranking row")
            q, d, rank = map(int, fields[:3])
            if not (0 <= q < len(query_ids) and 0 <= d < len(document_ids)) or rank < 1:
                raise ValueError("Invalid ColBERT query/document/rank")
            if (q, d) in seen:
                raise ValueError("Duplicate ColBERT query/document pair")
            seen.add((q, d))
            qid, doc = query_ids[q], document_ids[d]
            if qid != doc:
                results[qid][doc] = -float(rank)
    return {qid: dict(sorted(rows.items(), key=lambda item: item[1], reverse=True)[:depth])
            for qid, rows in results.items()}


class ColBERTAdapter(Adapter):
    def preflight(self):
        super().preflight()
        self.python, self.repo, self.env = runtime(self.s, "ColBERT")
        subprocess.run([str(self.python), "-c",
                        "import torch, faiss; assert torch.cuda.is_available(), 'ColBERT v1 needs CUDA'; from colbert.modeling.colbert import ColBERT"],
                       cwd=self.repo, env=self.env, check=True, timeout=120)

    def retrieve(self, corpus, queries, context):
        weights = asset("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/models/ColBERT/msmarco.psg.l2.zip",
                        self.base / "assets/msmarco.psg.l2")
        checkpoint = next(weights.rglob("colbert-300000.dnn"))
        output = context["output"]
        collection, query_file = output / "collection.tsv", output / "queries.tsv"
        ids, qids = list(corpus), list(queries)
        clean = lambda text: " ".join(text.split())
        with collection.open("w", encoding="utf-8") as stream:
            for idx, doc in enumerate(ids):
                stream.write(f"{idx}\t{clean(corpus[doc]['title'] + ' ' + corpus[doc]['text'])}\n")
        with query_file.open("w", encoding="utf-8") as stream:
            for idx, qid in enumerate(qids):
                stream.write(f"{idx}\t{clean(queries[qid])}\n")
        write_json(output / "id_mapping.json", {"documents": ids, "queries": qids})
        partitions = self.s["partitions"][context["dataset"]]
        root, index_root = output / "experiments", output / "indexes"
        index_name = "beir-index"
        common = ["--root", root, "--experiment", context["dataset"], "--run", "beir-evaluation"]
        model = ["--checkpoint", checkpoint, "--doc_maxlen", self.s["max_length"],
                 "--query_maxlen", self.s["query_max_length"], "--similarity", "l2", "--mask-punctuation"]
        reused_index = False
        previous = self.s.get("reuse_index_from", {}).get(context["dataset"])
        if previous:
            previous = Path(previous).resolve()
            old_config = json.loads((previous / "config.json").read_text())
            keys = ("max_length", "query_max_length", "partitions", "index_sample", "batch_size")
            if any(old_config[key] != self.s[key] for key in keys):
                raise ValueError("Previous ColBERT index has different encoding/index settings")
            for filename in ("collection.tsv", "queries.tsv", "id_mapping.json"):
                if file_sha256(previous / filename) != file_sha256(output / filename):
                    raise ValueError(f"Previous ColBERT {filename} differs from current dataset")
            old_index = previous / "indexes" / index_name
            required = [old_index / f"ivfpq.{partitions}.faiss", old_index / "metadata.json"]
            if not all(path.is_file() and path.stat().st_size for path in required) or not list(old_index.glob("doclens.*.json")):
                raise ValueError("Previous ColBERT index is incomplete")
            metadata = json.loads((old_index / "metadata.json").read_text())
            # The original metadata stores the checkpoint used by the encoder.
            if str(checkpoint) not in json.dumps(metadata):
                raise ValueError("Previous ColBERT index checkpoint cannot be verified")
            index_root = old_index.parent
            reused_index = True
            write_json(output / "reused_index.json", {
                "directory": str(old_index), "validated": [*keys, "collection", "queries", "ids", "checkpoint"],
                "metadata_sha256": file_sha256(old_index / "metadata.json"),
                "faiss_sha256": file_sha256(required[0]),
            })
        # No distributed launcher is needed for the original code's single-GPU mode.
        if not reused_index:
            execute([self.python, "-m", "colbert.index", *common, *model,
                     "--bsize", self.s["batch_size"], "--index_root", index_root,
                     "--index_name", index_name, "--collection", collection],
                    output / "colbert-index.log", cwd=self.repo, env=self.env)
            execute([self.python, "-m", "colbert.index_faiss", *common,
                     "--index_root", index_root, "--index_name", index_name,
                     "--partitions", partitions, "--sample", self.s["index_sample"]],
                    output / "colbert-faiss.log", cwd=self.repo, env=self.env)
        ranking_dir = output / "original-ranking"
        rank_batch_size = self.s.get("rank_batch_size", 256)
        retrieve_env = {**self.env, "BEIR_COLBERT_RANK_BATCH_SIZE": str(rank_batch_size)}
        execute([self.python, Path(__file__).with_name("colbert_retrieve_worker.py"), *common, *model,
                 "--bsize", self.s["batch_size"], "--queries", query_file,
                 "--index_root", index_root, "--index_name", index_name,
                 "--partitions", partitions, "--nprobe", self.s["nprobe"],
                 "--faiss_depth", self.s["faiss_depth"], "--depth", self.s["return_depth"] + 1,
                 "--ranking_dir", ranking_dir], output / "colbert-retrieve.log", cwd=self.repo, env=retrieve_env)
        write_json(output / "model.json", {
            "checkpoint": str(checkpoint), "checkpoint_asset": json.loads((weights / "asset.json").read_text()),
            "source_commit": REPOSITORIES["ColBERT"][1], "backend": "original ColBERT v1 ANN + MaxSim",
            "similarity": "l2", "partitions": partitions,
            "score_representation": "negative exported rank; original v1 logger omits scores",
            "return_depth": self.s["return_depth"],
            "rank_batch_size": rank_batch_size, "reused_index": reused_index,
            "precision": "FP32 encoding (AMP disabled); upstream stores document token embeddings in FP16",
        })
        results = parse_colbert_ranking(ranking_dir / "ranking.tsv", ids, qids, self.s["return_depth"])
        if not reused_index and not self.s.get("keep_index", False):
            shutil.rmtree(index_root)
        return results
