"""Quick standalone check: how good is BM25-only L1 retrieval on its own?
Establishes the baseline everything else (dense retrieval, L2 reranking) has
to beat, and quantifies the vocabulary-mismatch problem observed during dev
(narrative case facts vs. abstract statutory language)."""
import json
import time

import pandas as pd

from retrieval import load_article_corpus, BM25Retriever
from metrics import evaluate_ranklists

VALID = "../data/processed/valid.parquet"
K_CANDIDATES = 50


def main():
    corpus = load_article_corpus()
    bm25 = BM25Retriever(corpus)

    df = pd.read_parquet(VALID)
    df["relevant_articles"] = df["relevant_articles"].apply(json.loads)

    t0 = time.time()
    rank_lists, relevant_sets = [], []
    for _, row in df.iterrows():
        ranked = [aid for aid, _ in bm25.topk(row["fact"], K_CANDIDATES)]
        rank_lists.append(ranked)
        relevant_sets.append(set(row["relevant_articles"]))
    elapsed = time.time() - t0

    metrics = evaluate_ranklists(rank_lists, relevant_sets, ks=(1, 3, 5, 10, 20, 50))
    print(f"BM25-only L1 retrieval on {len(df)} valid queries ({elapsed:.1f}s, "
          f"{len(df)/elapsed:.0f} q/s):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
