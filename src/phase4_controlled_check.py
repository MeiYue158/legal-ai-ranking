"""
Phase 4 rigor check: the e5-feature model in dense_strong.py was trained on
an 8K subsample of train while the MiniLM-feature full model (ablation_clean)
used the full 30,214 -- conflating "better embedding model" with "more
data". This isolates JUST the embedding-model variable by training the
ORIGINAL MiniLM-feature set on the SAME 8K subsample (no new embeddings
needed -- reuses already-computed train_clean_features.parquet).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from metrics import ndcg_at_k
from eval_utils import load_true_gold

ALL_FEATURES = ["bm25_score", "bm25_rank", "dense_score", "dense_rank",
                 "case_vote_score", "case_vote_count", "jaccard",
                 "fact_len", "article_len", "article_prior"]


def main():
    train_full = pd.read_parquet("../data/processed/train_clean_features.parquet")
    valid_df = pd.read_parquet("../data/processed/valid_clean_features.parquet")
    test_df = pd.read_parquet("../data/processed/test_clean_features.parquet")

    # SAME subsample as dense_strong.py's Step C (same seed, same N, same source)
    train_clean = pd.read_parquet("../data/processed/train_clean.parquet")
    sub_ids = set(train_clean["case_id"].sample(n=8000, random_state=42))
    train_sub = train_full[train_full["case_id"].isin(sub_ids)]
    print(f"MiniLM-feature model on the SAME 8K subsample (n={train_sub['case_id'].nunique()} queries)")

    def grouped(df):
        df = df.sort_values("case_id").reset_index(drop=True)
        return df, df.groupby("case_id", sort=False).size().tolist()

    train_g, train_groups = grouped(train_sub)
    valid_g, valid_groups = grouped(valid_df)

    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10,
        eval_at=[10], verbose=-1, random_state=42,
    )
    model.fit(
        train_g[ALL_FEATURES], train_g["label"], group=train_groups,
        eval_set=[(valid_g[ALL_FEATURES], valid_g["label"])], eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    true_gold = load_true_gold("_clean")
    test_df = test_df.copy()
    test_df["score"] = model.predict(test_df[ALL_FEATURES])
    ndcg10 = []
    for case_id, g in test_df.groupby("case_id"):
        gold = true_gold.get(case_id, set())
        if not gold:
            continue
        ranked = g.sort_values("score", ascending=False)["article_id"].tolist()
        ndcg10.append(ndcg_at_k(ranked, gold, 10))

    result = float(np.mean(ndcg10))
    print(f"MiniLM-features, 8K-subsample-trained: NDCG@10 = {result:.4f}")
    print(f"(for comparison: e5-features, same 8K subsample: NDCG@10 = 0.7984)")
    print(f"(for comparison: MiniLM-features, FULL 30,214 train: NDCG@10 = 0.7892)")

    with open("../reports/phase4_controlled_check.json", "w") as f:
        json.dump({"minilm_8k_subsample_ndcg10": result,
                    "e5_8k_subsample_ndcg10": 0.7984,
                    "minilm_full_30k_ndcg10": 0.7892}, f, indent=2)


if __name__ == "__main__":
    main()
