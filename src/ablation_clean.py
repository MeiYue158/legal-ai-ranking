"""
Phase 5 (continued): repeat the exact same group-ablation as ablation.py,
on the DECONTAMINATED features, to check whether case-precedent retrieval
remains the dominant signal after cleanup -- not assumed, re-verified.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from metrics import ndcg_at_k
from eval_utils import load_true_gold

ALL_FEATURES = [
    "bm25_score", "bm25_rank", "dense_score", "dense_rank",
    "case_vote_score", "case_vote_count", "jaccard",
    "fact_len", "article_len", "article_prior",
]

GROUPS = {
    "bm25": ["bm25_score", "bm25_rank"],
    "dense": ["dense_score", "dense_rank"],
    "case_precedent": ["case_vote_score", "case_vote_count"],
    "jaccard": ["jaccard"],
    "priors_lengths": ["fact_len", "article_len", "article_prior"],
    "article_prior_only": ["article_prior"],  # Phase 6 needs this isolated
}


def load_grouped(name):
    df = pd.read_parquet(f"../data/processed/{name}_clean_features.parquet")
    df = df.sort_values("case_id").reset_index(drop=True)
    groups = df.groupby("case_id", sort=False).size().tolist()
    return df, groups


def train_and_eval(feature_cols, train_df, train_groups, valid_df, valid_groups, test_df, true_gold, seed=42):
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10,
        eval_at=[10], verbose=-1, random_state=seed,
    )
    model.fit(
        train_df[feature_cols], train_df["label"], group=train_groups,
        eval_set=[(valid_df[feature_cols], valid_df["label"])], eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    test_df = test_df.copy()
    test_df["score"] = model.predict(test_df[feature_cols])
    ndcg10 = []
    for case_id, g in test_df.groupby("case_id"):
        gold = true_gold.get(case_id, set())
        if not gold:
            continue
        ranked = g.sort_values("score", ascending=False)["article_id"].tolist()
        ndcg10.append(ndcg_at_k(ranked, gold, 10))
    gains = dict(zip(feature_cols, model.booster_.feature_importance(importance_type="gain")))
    return float(np.mean(ndcg10)), gains, model


def main():
    train_df, train_groups = load_grouped("train")
    valid_df, valid_groups = load_grouped("valid")
    test_df, _ = load_grouped("test")
    true_gold = load_true_gold("_clean")

    print("Training FULL model on CLEAN data (all features, seed=42)...")
    full_ndcg, gains, model = train_and_eval(
        ALL_FEATURES, train_df, train_groups, valid_df, valid_groups, test_df, true_gold)
    print(f"FULL model (clean) NDCG@10 = {full_ndcg:.4f}")
    print("Gain-based feature importance (clean):")
    for name, g in sorted(gains.items(), key=lambda x: -x[1]):
        print(f"  {name}: {g:.1f}")

    print("\nAblations on CLEAN data (remove one feature GROUP at a time):")
    results = {"full": full_ndcg}
    for group_name, cols_to_remove in GROUPS.items():
        remaining = [c for c in ALL_FEATURES if c not in cols_to_remove]
        ndcg, _, _ = train_and_eval(
            remaining, train_df, train_groups, valid_df, valid_groups, test_df, true_gold)
        drop = full_ndcg - ndcg
        results[f"minus_{group_name}"] = ndcg
        print(f"  minus {group_name:20s}: NDCG@10={ndcg:.4f}  (drop vs full: {drop:+.4f})")

    with open("../reports/ablation_clean_results.json", "w") as f:
        json.dump({"gains": gains, "ablations": results}, f, indent=2)
    print("\nWrote ../reports/ablation_clean_results.json")
    print("(Model NOT saved as final yet -- architecture freeze happens after Phase 4's "
          "dense-retrieval conclusion and Phase 6's head/mid/tail analysis, per execution order.)")


if __name__ == "__main__":
    main()
