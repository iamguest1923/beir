"""Execute unmodified ColBERT v1 with bounded, numerically equivalent MaxSim batches."""
import os
import runpy


def configure_ranker(batch_size):
    import colbert.ranking.index_ranker as module
    if batch_size < 1:
        raise ValueError("Rank batch size must be positive")
    module.BSIZE = batch_size
    original = module.IndexRanker.rank

    def rank(self, Q, pids, views=None, shift=0):
        scores = []
        for start in range(0, len(pids), batch_size):
            end = start + batch_size
            query = Q if Q.size(0) == 1 else Q[start:end]
            scores.extend(original(self, query, pids[start:end], views=views, shift=shift))
        return scores

    module.IndexRanker.rank = rank


if __name__ == "__main__":
    configure_ranker(int(os.environ.get("BEIR_COLBERT_RANK_BATCH_SIZE", "256")))
    runpy.run_module("colbert.retrieve", run_name="__main__")
