"""Shared "score a trained model against clean/original test set" logic,
factored out of evaluate_full.py so train_multiseed.py and ablation.py don't
each reimplement it."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import ndcg_at_k, evaluate_ranklists

PROCESSED = Path(__file__).parent.parent / "data/processed"


def load_true_gold(suffix=""):
    raw = pd.read_parquet(PROCESSED / f"test{suffix}.parquet")
    return {row["case_id"]: set(json.loads(row["relevant_articles"])) for _, row in raw.iterrows()}


def score_model(model, feature_cols, test_df, true_gold, ks=(1, 3, 5, 10)):
    test_df = test_df.copy()
    test_df["_score"] = model.predict(test_df[feature_cols])
    rank_lists, relevant_sets, ndcg10 = [], [], []
    for case_id, g in test_df.groupby("case_id"):
        gold = true_gold.get(case_id, set())
        if not gold:
            continue
        ranked = g.sort_values("_score", ascending=False)["article_id"].tolist()
        rank_lists.append(ranked)
        relevant_sets.append(gold)
        ndcg10.append(ndcg_at_k(ranked, gold, 10))
    metrics = evaluate_ranklists(rank_lists, relevant_sets, ks=ks)
    return metrics, ndcg10
