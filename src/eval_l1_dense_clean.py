"""Standalone MiniLM dense-retrieval baseline on the CLEAN test set, run as
its own process deliberately -- kept separate from dense_strong.py (which
loads e5-base) because loading two different SentenceTransformer model
families in one process has reproduced a SIGSEGV before (see rank_core.py's
docstring); simplest robust fix is separate processes, not more debugging of
an already-diagnosed native-library fragility."""
import json

import pandas as pd
import numpy as np

from retrieval import load_article_corpus, DenseRetriever
from metrics import evaluate_ranklists


def main():
    corpus = load_article_corpus()
    dense = DenseRetriever(corpus)
    test_df = pd.read_parquet("../data/processed/test_clean.parquet")

    q_embs = dense.model.encode(test_df["fact"].tolist(), normalize_embeddings=True,
                                 batch_size=64, show_progress_bar=True)
    rank_lists, rel = [], []
    for i, (_, row) in enumerate(test_df.iterrows()):
        sims = dense.embeddings @ q_embs[i]
        order = np.argsort(-sims)[:50]
        rank_lists.append([dense.article_ids[j] for j in order])
        rel.append(set(json.loads(row["relevant_articles"])))
    metrics = evaluate_ranklists(rank_lists, rel, ks=(1, 3, 5, 10, 30))

    print(f"MiniLM standalone dense retrieval, clean TEST (n={len(test_df)}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open("../reports/minilm_dense_clean_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nWrote ../reports/minilm_dense_clean_results.json")


if __name__ == "__main__":
    main()
