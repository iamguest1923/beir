"""Download Vietnamese benchmarks and convert them to the project's BEIR format."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
# Immutable source snapshots, verified against the Hub API. ViQuAD is a mirror;
# the remaining repositories are the published QA source / retrieval conversion.
SOURCES = {
    "uit-viquad": ("vile99/vi_quad", "88a2838ff1a44a63db584a20de9f1fd017badf1d",
                    ["train.json", "validation.json", "test.json"]),
    "vihealthqa": ("tarudesu/ViHealthQA", "d90a62deedfc822a15de3f35535bb41474e4d3f4",
                   ["train.csv", "val.csv", "test.csv"]),
    "zalo-legal": ("GreenNode/zalo-ai-legal-text-retrieval-vn", "12d76d4d04b94ceada970fcfbe7fec20bfa97389",
                   ["corpus.jsonl", "queries.jsonl", "qrels/train.jsonl", "qrels/test.jsonl"]),
    "csconda": ("ura-hcmut/Vietnamese-Customer-Support-QA", "95da2df0715df14eaf184dcc2e5c33b77194cb5a",
                ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected a nonempty text field")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def identifier(value):
    if value is None or isinstance(value, bool) or not str(value).strip():
        raise ValueError(f"Invalid ID: {value!r}")
    value = str(value)
    if any(c in value for c in "\t\r\n"):
        raise ValueError("IDs must not contain tabs or newlines")
    return value


def rows(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires: pip install pyarrow") from exc
        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        content = path.read_text(encoding="utf-8")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            # The pinned ViQuAD mirror uses JSON Lines with a .json suffix.
            if exc.msg != "Extra data":
                raise
            for line in content.splitlines():
                if line.strip():
                    yield json.loads(line)
            return
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        elif isinstance(data, dict) and "question" in data:
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array or SQuAD data: {path}")
        for item in data:
            if "paragraphs" in item:
                for paragraph in item["paragraphs"]:
                    # Keep paragraphs without answerable questions as distractors.
                    yield {"context": paragraph["context"], "title": item.get("title", ""),
                           "_qas": paragraph["qas"]}
            else:
                yield item
    else:
        raise ValueError(f"Unsupported source file: {path}")


def download(name, cache, revision=None):
    repo, pinned, files = SOURCES[name]
    revision = revision or pinned
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character commit SHA")
    folder = Path(cache) / name / revision
    folder.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    for filename in files:
        target = folder / filename
        if target.is_file() and target.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        headers = {"User-Agent": "beir-vietnamese-data/1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        LOG.info("Downloading %s/%s", name, filename)
        partial = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as source:
                with partial.open("wb") as output:
                    shutil.copyfileobj(source, output)
            if not partial.stat().st_size:
                raise ValueError(f"Empty download: {filename}")
            partial.replace(target)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(f"Access denied for {repo}. Request access on its dataset page, "
                                   "set HF_TOKEN, or use --source-dir with authorized local files.") from exc
            raise
        finally:
            partial.unlink(missing_ok=True)
    return folder, {"repository": repo, "revision": revision, "kind": "huggingface"}


class Benchmark:
    def __init__(self):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.stats = {"duplicate_documents": 0, "duplicate_queries": 0,
                      "duplicate_qrels": 0, "unanswerable_queries_skipped": 0,
                      "title_only_documents": 0}
        self.used_files = set()

    def read(self, path):
        self.used_files.add(Path(path))
        return rows(path)

    def document(self, did, text, title=""):
        did = identifier(did)
        if not isinstance(title, str):
            raise ValueError("Document title must be a string")
        if not isinstance(text, str) or not (text.strip() or title.strip()):
            raise ValueError("Document needs nonempty text or title")
        row = {"_id": did, "title": title, "text": text}
        if did in self.corpus:
            if self.corpus[did] != row:
                raise ValueError(f"Conflicting document ID: {did}")
            self.stats["duplicate_documents"] += 1
        else:
            self.corpus[did] = row
            self.stats["title_only_documents"] += int(not text.strip())
        return did

    def passage(self, text, title=""):
        text = normalize(text)
        did = "ctx-" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Identity depends on context only, even when source titles differ.
        if did in self.corpus:
            self.stats["duplicate_documents"] += 1
            return did
        return self.document(did, text, title)

    def query(self, qid, text):
        qid = identifier(qid)
        normalize(text)
        row = {"_id": qid, "text": text}
        if qid in self.queries:
            if self.queries[qid] != row:
                raise ValueError(f"Conflicting query ID: {qid}")
            self.stats["duplicate_queries"] += 1
        self.queries[qid] = row
        return qid

    def relevance(self, split, qid, did, score=1):
        qid, did = identifier(qid), identifier(did)
        if isinstance(score, bool) or not re.fullmatch(r"-?\d+", str(score)) or int(score) < -1:
            raise ValueError(f"Invalid relevance score: {score}")
        rels = self.qrels.setdefault(split, {})
        key = (qid, did)
        if key in rels:
            if rels[key] != int(score):
                raise ValueError(f"Conflicting qrel: {key}")
            self.stats["duplicate_qrels"] += 1
        rels[key] = int(score)

    def validate(self):
        if not self.corpus or not self.queries or not self.qrels:
            raise ValueError("Empty corpus, queries or qrels")
        for split, rels in self.qrels.items():
            if not rels:
                raise ValueError(f"Empty qrels: {split}")
            positive = set()
            for (qid, did), score in rels.items():
                if qid not in self.queries or did not in self.corpus:
                    raise ValueError(f"Unknown qrel ID: {split}/{qid}/{did}")
                if score > 0:
                    positive.add(qid)
            if {q for q, _ in rels} - positive:
                raise ValueError(f"Queries without positive judgments: {split}")

    def write(self, output, metadata):
        self.validate()
        output = Path(output)
        # Never silently replace previously prepared benchmark data.
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
        try:
            for filename, data in [("corpus.jsonl", self.corpus), ("queries.jsonl", self.queries)]:
                with (temporary / filename).open("w", encoding="utf-8") as stream:
                    for key in sorted(data):
                        stream.write(json.dumps(data[key], ensure_ascii=False) + "\n")
            (temporary / "qrels").mkdir()
            for split, rels in sorted(self.qrels.items()):
                with (temporary / "qrels" / f"{split}.tsv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(["query-id", "corpus-id", "score"])
                    for (qid, did), score in sorted(rels.items()):
                        writer.writerow([qid, did, score])
            split_queries = {s: {q for q, _ in rels} for s, rels in self.qrels.items()}
            overlaps = {}
            for a in sorted(split_queries):
                for b in sorted(split_queries):
                    if a < b:
                        texts_a = {normalize(self.queries[q]["text"]) for q in split_queries[a]}
                        texts_b = {normalize(self.queries[q]["text"]) for q in split_queries[b]}
                        overlaps[f"{a}/{b}"] = len(texts_a & texts_b)
            metadata.update(documents=len(self.corpus), queries=len(self.queries), statistics=self.stats,
                            splits={s: {"queries": len(split_queries[s]), "qrels": len(v)}
                                    for s, v in self.qrels.items()},
                            query_text_overlap_between_splits=overlaps,
                            source_sha256={str(p): sha256(p) for p in sorted(self.used_files)},
                            artifact_sha256={str(p.relative_to(temporary)): sha256(p)
                                             for p in sorted(temporary.rglob("*")) if p.is_file()})
            (temporary / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return metadata


def qa_files(source):
    found = {}
    for path in sorted(Path(source).rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv", ".parquet"}:
            continue
        for split, aliases in {"train": ["train"], "dev": ["dev", "val", "valid", "validation"], "test": ["test"]}.items():
            if any(re.match(rf"^{alias}(?:$|[-_.])", path.stem, re.I) for alias in aliases):
                found.setdefault(split, []).append(path)
                break
    if not found:
        raise ValueError("No train/dev/val/validation/test JSON, JSONL, CSV or Parquet files found")
    return found


def convert_qa(name, source, benchmark, context_field=None, question_field="question"):
    for split, files in sorted(qa_files(source).items()):
        for path in files:
            for index, row in enumerate(benchmark.read(path)):
                text_field = context_field or ("answer" if name in {"vihealthqa", "csconda"} else "context")
                did = benchmark.passage(row[text_field], row.get("title", ""))
                questions = row["_qas"] if "_qas" in row else [row]
                for number, qa in enumerate(questions):
                    if name == "uit-viquad":
                        answers = qa.get("answers")
                        texts = answers.get("text", []) if isinstance(answers, dict) else [a["text"] for a in (answers or [])]
                        if isinstance(texts, str):
                            texts = [texts] if texts.strip() else []
                        if qa.get("is_impossible") or not texts:
                            benchmark.stats["unanswerable_queries_skipped"] += 1
                            continue
                        if not all(normalize(t) in normalize(row[text_field]) for t in texts):
                            raise ValueError(f"Answer not found in context: {path}:{index}")
                    source_id = qa.get("id", f"{path.name}:{index}:{number}")
                    qid = benchmark.query(f"{split}:{identifier(source_id)}", qa[question_field])
                    benchmark.relevance(split, qid, did)


def convert_retrieval(source, benchmark):
    source = Path(source)
    for row in benchmark.read(source / "corpus.jsonl"):
        benchmark.document(row["_id"], row["text"], row.get("title", ""))
    for row in benchmark.read(source / "queries.jsonl"):
        benchmark.query(row["_id"], row["text"])
    files = sorted((source / "qrels").glob("*"))
    for path in files:
        split = {"val": "dev", "validation": "dev", "valid": "dev"}.get(path.stem, path.stem)
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported qrels split: {path}")
        if path.suffix == ".tsv":
            benchmark.used_files.add(path)
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                    raise ValueError(f"Invalid qrels header: {path}")
                for row in reader:
                    benchmark.relevance(split, row["query-id"], row["corpus-id"], row["score"])
        elif path.suffix == ".jsonl":
            for row in benchmark.read(path):
                benchmark.relevance(split, row["query-id"], row["corpus-id"], row["score"])
        else:
            raise ValueError(f"Unsupported qrels file: {path}")


def prepare(name, output_dir=ROOT / "datasets", cache_dir=ROOT / "raw", source_dir=None,
            revision=None, context_field=None, question_field="question"):
    destination = Path(output_dir) / name
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new --output-dir")
    if source_dir:
        if revision:
            raise ValueError("--revision only applies to downloaded sources")
        source = Path(source_dir).resolve()
        provenance = {"kind": "local", "directory": str(source)}
    else:
        source, provenance = download(name, cache_dir, revision)
    benchmark = Benchmark()
    retrieval = (source / "corpus.jsonl").is_file() and (source / "queries.jsonl").is_file()
    if retrieval:
        convert_retrieval(source, benchmark)
    elif name == "zalo-legal":
        raise ValueError("Legal retrieval requires corpus.jsonl, queries.jsonl and qrels/{split}.tsv or .jsonl")
    else:
        convert_qa(name, source, benchmark, context_field, question_field)
    policy = "preserve_source_ids_and_relevance" if retrieval else "qa_to_retrieval; normalized_unique_passages; source_pair_positive"
    return benchmark.write(destination, {"dataset": name, "source": provenance, "format": "BEIR",
        "preparation_script_sha256": sha256(__file__),
        "source_variant": "ViQuAD mirror containing answerable and unanswerable questions" if name == "uit-viquad" else name,
        "conversion": policy, "context_field": context_field or ("answer" if name in {"vihealthqa", "csconda"} else "context"),
        "question_field": question_field, "corpus_scope": "all_source_splits",
        "split_policy": "preserve_source; val/validation mapped to dev; no synthetic test split",
        "evaluation_note": "QA-derived relevance is incomplete; results are not native QA scores. "
                           "Check split overlap and model training data before reporting held-out results."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=[*SOURCES, "all"], default=["all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "raw")
    parser.add_argument("--source-dir", type=Path, help="Local source; requires exactly one dataset")
    parser.add_argument("--revision", help="Override pinned source with a commit SHA; one dataset only")
    parser.add_argument("--context-field", help="Default: answer for ViHealthQA/CSConDa, context for ViQuAD")
    parser.add_argument("--question-field", default="question")
    args = parser.parse_args()
    names = list(SOURCES) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    if (args.source_dir or args.revision) and len(names) != 1:
        parser.error("--source-dir/--revision requires exactly one dataset")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    failures = []
    for name in names:
        try:
            info = prepare(name, args.output_dir, args.cache_dir, args.source_dir, args.revision,
                           args.context_field, args.question_field)
            LOG.info("%s: %d documents, %d queries; splits=%s", name, info["documents"], info["queries"], info["splits"])
        except Exception as exc:
            LOG.error("%s: %s", name, exc)
            failures.append(name)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
