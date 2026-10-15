"""Evaluate a prepared benchmark with existing BEIR models and metric code."""
import argparse
import json
from pathlib import Path
import sys

# Prefer this checkout, following the model runner convention.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Sentence Transformers model name or local checkpoint")
    source.add_argument("--results", type=Path, help="Existing JSON {query_id: {document_id: score}}")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    # Reuse the project's loader, output format, result validation and metrics.
    from beir.datasets.data_loader import GenericDataLoader
    from runs._shared.common import score_and_save

    corpus, queries, qrels = GenericDataLoader(str(args.data)).load(split=args.split)
    if args.results:
        results = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        from beir.retrieval import models
        from beir.retrieval.search.dense import DenseRetrievalExactSearch
        from beir.retrieval.evaluation import EvaluateRetrieval
        model = DenseRetrievalExactSearch(models.SentenceBERT(args.model), batch_size=args.batch_size)
        results = EvaluateRetrieval(model, score_function="cos_sim").retrieve(corpus, queries)
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = score_and_save(args.output, results, corpus, queries, qrels, [1, 3, 5, 10, 100])
    metadata = {"data": str(args.data.resolve()), "split": args.split, "model": args.model,
                "results": str(args.results) if args.results else None,
                "data_sha256": {name: sha256(args.data / name) for name in
                                ["corpus.jsonl", "queries.jsonl", f"qrels/{args.split}.tsv"]}}
    if args.results:
        metadata["results_sha256"] = sha256(args.results)
    (args.output / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
