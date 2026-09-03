"""L1 dense-retrieval-only evaluation, same protocol as eval_l1_bm25.py, so
the two are directly comparable."""
import json
import time

import pandas as pd

from retrieval import load_article_corpus, DenseRetriever
from metrics import evaluate_ranklists

VALID = "../data/processed/valid.parquet"
K_CANDIDATES = 50


def main():
    corpus = load_article_corpus()
    dense = DenseRetriever(corpus)

    df = pd.read_parquet(VALID)
    df["relevant_articles"] = df["relevant_articles"].apply(json.loads)

    t0 = time.time()
    q_emb = dense.embed_queries(df["fact"].tolist())
    rank_lists, relevant_sets = [], []
    for i in range(len(df)):
        ranked = [aid for aid, _ in dense.topk_from_emb(q_emb[i], K_CANDIDATES)]
        rank_lists.append(ranked)
        relevant_sets.append(set(df.iloc[i]["relevant_articles"]))
    elapsed = time.time() - t0

    metrics = evaluate_ranklists(rank_lists, relevant_sets, ks=(1, 3, 5, 10, 20, 50))
    print(f"Dense-only L1 retrieval on {len(df)} valid queries ({elapsed:.1f}s):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
