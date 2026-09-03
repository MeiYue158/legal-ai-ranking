"""Phase 5: apples-to-apples architecture comparison on the held-out TEST
set. Same candidate pool for every architecture (so we're purely measuring
"does the ranking function get better", not "does the candidate set get
bigger") -- then bootstrap-CI + paired significance test the deltas.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from metrics import evaluate_ranklists, ndcg_at_k, recall_at_k
from train_ranker import FEATURE_COLS

PROCESSED = Path(__file__).parent.parent / "data/processed"
MODEL_DIR = Path(__file__).parent.parent / "reports"


def rank_by(df, score_col):
    return df.sort_values(score_col, ascending=False)["article_id"].tolist()


def paired_bootstrap_pvalue(a, b, n_boot=5000, seed=0):
    """One-sided: P(mean(b) - mean(a) <= 0) under the null, via paired
    bootstrap resampling of the per-query metric differences."""
    a, b = np.asarray(a), np.asarray(b)
    diff = b - a
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    p = float((boot_means <= 0).mean())
    return float(diff.mean()), p


def load_true_gold():
    """The candidate-pool `label` column is NOT a valid ground truth for
    recall/NDCG: a truly-relevant article that candidate generation missed
    never appears as a row at all, so `label==1` silently measures against
    only the subset of gold articles the pool happened to retrieve -- that
    tautologically inflates every metric (this is exactly what produced an
    impossible 1.0000 recall ceiling on the first run). Ground truth must
    come from the untouched per-query label set instead."""
    raw = pd.read_parquet(PROCESSED / "test.parquet")
    return {
        row["case_id"]: set(json.loads(row["relevant_articles"]))
        for _, row in raw.iterrows()
    }


def main():
    test_df = pd.read_parquet(PROCESSED / "test_features.parquet")
    test_df["label"] = test_df["label"].astype(int)
    true_gold = load_true_gold()
    lgb_model = joblib.load(MODEL_DIR / "lgb_ranker.pkl")
    logreg_model = joblib.load(MODEL_DIR / "logreg_baseline.pkl")
    test_df["lgb_score"] = lgb_model.predict(test_df[FEATURE_COLS])
    test_df["logreg_score"] = logreg_model.predict_proba(test_df[FEATURE_COLS])[:, 1]

    architectures = {
        "BM25-only (within pool)": "bm25_score",
        "Dense-only (within pool)": "dense_score",
        "Case-precedent-kNN-only (within pool)": "case_vote_score",
        "Logistic regression (pointwise, all features)": "logreg_score",
        "LightGBM LambdaRank L2 (all features)": "lgb_score",
    }

    per_query_ndcg10 = {}
    pool_recall_ceiling = []
    summary_rows = []

    for group_id, g in test_df.groupby("case_id"):
        gold = true_gold.get(group_id, set())
        if not gold:
            continue
        pool_recall_ceiling.append(len(gold & set(g["article_id"])) / len(gold))

    for name, col in architectures.items():
        rank_lists, relevant_sets, ndcg10_list = [], [], []
        for group_id, g in test_df.groupby("case_id"):
            gold = true_gold.get(group_id, set())
            if not gold:
                continue
            ranked = rank_by(g, col)
            rank_lists.append(ranked)
            relevant_sets.append(gold)
            ndcg10_list.append(ndcg_at_k(ranked, gold, 10))
        metrics = evaluate_ranklists(rank_lists, relevant_sets, ks=(1, 3, 5, 10))
        per_query_ndcg10[name] = ndcg10_list
        row = {"architecture": name, **metrics}
        summary_rows.append(row)
        print(f"{name}: NDCG@10={metrics['ndcg@10']:.4f} Recall@10={metrics['recall@10']:.4f} MRR={metrics['mrr']:.4f}")

    print(f"\ncandidate-pool Recall ceiling (upper bound any reranker could hit): "
          f"{np.mean(pool_recall_ceiling):.4f}")

    print("\nPaired bootstrap significance (LightGBM L2 vs. each baseline), NDCG@10:")
    lgb_vals = per_query_ndcg10["LightGBM LambdaRank L2 (all features)"]
    for name, vals in per_query_ndcg10.items():
        if name == "LightGBM LambdaRank L2 (all features)":
            continue
        delta, p = paired_bootstrap_pvalue(vals, lgb_vals)
        print(f"  vs {name}: mean NDCG@10 delta = {delta:+.4f}, one-sided bootstrap p = {p:.4f}")

    out = pd.DataFrame(summary_rows)
    out_path = Path(__file__).parent.parent / "reports/architecture_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
