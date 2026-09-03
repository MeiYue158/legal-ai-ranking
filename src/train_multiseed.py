"""
Phase 3: reproducibility. Train the SAME architecture (unchanged features,
unchanged hyperparameters) on the CLEAN data across >=3 independent seeds,
report mean +/- SD, not a single cherry-picked run. Also writes a
machine-readable experiment config alongside the results.
"""
import json
import random
from pathlib import Path

import numpy as np

from train_ranker import FEATURE_COLS, load_grouped, train_lambdarank
from eval_utils import load_true_gold, score_model

SEEDS = [42, 123, 2025]
SUFFIX = "_clean"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    train_df, train_groups = load_grouped("train", SUFFIX)
    valid_df, valid_groups = load_grouped("valid", SUFFIX)
    test_df, _ = load_grouped("test", SUFFIX)
    true_gold = load_true_gold(SUFFIX)

    results = []
    for seed in SEEDS:
        set_all_seeds(seed)
        model = train_lambdarank(train_df, train_groups, valid_df, valid_groups, seed=seed)
        metrics, _ = score_model(model, FEATURE_COLS, test_df, true_gold)
        results.append({"seed": seed, **metrics})
        print(f"seed={seed}: NDCG@10={metrics['ndcg@10']:.4f} Recall@10={metrics['recall@10']:.4f} "
              f"MRR={metrics['mrr']:.4f}")

    ndcg10_vals = [r["ndcg@10"] for r in results]
    mean_ndcg10 = float(np.mean(ndcg10_vals))
    sd_ndcg10 = float(np.std(ndcg10_vals, ddof=1))
    print(f"\nNDCG@10 across {len(SEEDS)} seeds: mean={mean_ndcg10:.4f} sd={sd_ndcg10:.4f} "
          f"(individual: {[round(v,4) for v in ndcg10_vals]})")

    config = {
        "feature_cols": FEATURE_COLS,
        "seeds": SEEDS,
        "hyperparameters": {
            "objective": "lambdarank", "n_estimators": 300, "learning_rate": 0.05,
            "num_leaves": 31, "min_child_samples": 10, "eval_at": [1, 3, 5, 10],
            "early_stopping_rounds": 30,
        },
        "candidate_generation": {
            "n_bm25_candidates": 30, "n_dense_candidates": 30,
            "n_case_neighbors": 15, "n_case_vote_candidates": 15,
        },
        "data": {
            "suffix": SUFFIX,
            "train_size": len(train_df["case_id"].unique()),
            "valid_size": len(valid_df["case_id"].unique()),
            "test_size": len(test_df["case_id"].unique()),
        },
        "results": {
            "per_seed": results,
            "ndcg@10_mean": mean_ndcg10,
            "ndcg@10_sd": sd_ndcg10,
        },
    }
    with open("../reports/multiseed_results.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nWrote ../reports/multiseed_results.json")


if __name__ == "__main__":
    main()
