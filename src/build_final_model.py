"""
Phase 7: freeze the final architecture and run the one-time final
evaluation. Architecture decisions, each backed by an experiment run
earlier in this audit, not by assumption:

  - dense_score/dense_rank: rebuilt with e5-base (Phase 4: controlled
    8K-subsample check showed +0.0152 NDCG@10 over MiniLM at matched data
    size; now applied at full 30,214-train scale).
  - article_prior: DROPPED (Phase 6: cost only 0.008 NDCG@10 in ablation but
    actively hurt TAIL-article hit@10 by 7.2 points -- a bad trade to keep).
  - Everything else (bm25_score/rank, case_vote_score/count, jaccard,
    fact_len, article_len): unchanged, per the audit's explicit instruction
    not to rebuild what's already validated.

No further tuning happens after this file's test-set numbers are produced.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb  # lightgbm import first (see rank_core.py's note)

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

from retrieval import load_article_corpus
from metrics import ndcg_at_k, evaluate_ranklists

PROCESSED = Path(__file__).parent.parent / "data/processed"
REPORTS = Path(__file__).parent.parent / "reports"
E5_MODEL = "intfloat/multilingual-e5-base"

FINAL_FEATURES = ["bm25_score", "bm25_rank", "dense_score", "dense_rank",
                   "case_vote_score", "case_vote_count", "jaccard",
                   "fact_len", "article_len"]  # article_prior dropped -- see docstring
SEEDS = [42, 123, 2025]


def train_final(train_df, train_groups, valid_df, valid_groups, seed):
    """Same hyperparameters as train_ranker.py's train_lambdarank(), but
    parametrized on FINAL_FEATURES instead of that module's hardcoded
    10-feature FEATURE_COLS (which still includes article_prior, dropped
    here per the Phase 6 finding)."""
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10,
        eval_at=[1, 3, 5, 10], verbose=-1,
        random_state=seed, deterministic=True, force_row_wise=True,
    )
    model.fit(
        train_df[FINAL_FEATURES], train_df["label"], group=train_groups,
        eval_set=[(valid_df[FINAL_FEATURES], valid_df["label"])], eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def rebuild_dense(feat_df, case_to_emb, article_to_emb):
    feat_df = feat_df.copy()
    new_scores, new_ranks = [], []
    for case_id, g in feat_df.groupby("case_id", sort=False):
        q_emb = case_to_emb[case_id]
        sims = {aid: float(article_to_emb[aid] @ q_emb) for aid in g["article_id"]}
        order = sorted(sims, key=lambda a: -sims[a])
        rank_of = {a: r for r, a in enumerate(order)}
        for aid in g["article_id"]:
            new_scores.append(sims[aid])
            new_ranks.append(rank_of[aid])
    feat_df["dense_score"] = new_scores
    feat_df["dense_rank"] = new_ranks
    return feat_df


def main():
    print("Loading e5-base and computing statute-corpus embeddings...")
    e5_model = SentenceTransformer(E5_MODEL, device="cpu")
    corpus = load_article_corpus()
    article_ids = list(corpus.keys())
    article_texts = ["passage: " + corpus[a] for a in article_ids]
    article_embs = e5_model.encode(article_texts, normalize_embeddings=True, batch_size=32)
    article_to_emb = dict(zip(article_ids, article_embs))

    train_e5_embs = np.load(PROCESSED / "emb_cache/train_full_e5_embeddings.npy")
    train_e5_ids = json.load(open(PROCESSED / "emb_cache/train_full_e5_ids.json"))
    train_case_to_emb = dict(zip(train_e5_ids, train_e5_embs))

    print("Rebuilding FULL train dense features with e5-base...")
    train_feat = pd.read_parquet(PROCESSED / "train_clean_features.parquet")
    train_feat_e5 = rebuild_dense(train_feat, train_case_to_emb, article_to_emb)

    # valid/test e5 features already built at full scale in dense_strong.py's Step C
    valid_feat_e5 = pd.read_parquet(PROCESSED / "valid_clean_e5_features.parquet")
    test_feat_e5 = pd.read_parquet(PROCESSED / "test_clean_e5_features.parquet")

    def grouped(df):
        df = df.sort_values("case_id").reset_index(drop=True)
        return df, df.groupby("case_id", sort=False).size().tolist()

    train_g, train_groups = grouped(train_feat_e5)
    valid_g, valid_groups = grouped(valid_feat_e5)

    raw_test = pd.read_parquet(PROCESSED / "test_clean.parquet")
    true_gold = {row["case_id"]: set(json.loads(row["relevant_articles"])) for _, row in raw_test.iterrows()}

    results = []
    models = []
    for seed in SEEDS:
        print(f"\nTraining final architecture, seed={seed}...")
        model = train_final(train_g, train_groups, valid_g, valid_groups, seed=seed)
        test_feat_e5_copy = test_feat_e5.copy()
        test_feat_e5_copy["score"] = model.predict(test_feat_e5_copy[FINAL_FEATURES])
        ndcg10 = []
        for case_id, g in test_feat_e5_copy.groupby("case_id"):
            gold = true_gold.get(case_id, set())
            if not gold:
                continue
            ranked = g.sort_values("score", ascending=False)["article_id"].tolist()
            ndcg10.append(ndcg_at_k(ranked, gold, 10))
        score = float(np.mean(ndcg10))
        print(f"  seed={seed}: NDCG@10={score:.4f}")
        results.append({"seed": seed, "ndcg@10": score})
        models.append(model)

    ndcg_vals = [r["ndcg@10"] for r in results]
    mean_ndcg, sd_ndcg = float(np.mean(ndcg_vals)), float(np.std(ndcg_vals, ddof=1))
    print(f"\nFINAL ARCHITECTURE, NDCG@10 across {len(SEEDS)} seeds: mean={mean_ndcg:.4f} sd={sd_ndcg:.4f}")

    # canonical saved model = seed=42 (first in SEEDS), consistent with the
    # rest of this audit's convention
    canonical = models[0]
    joblib.dump(canonical, REPORTS / "lgb_ranker_final.pkl")
    print(f"Wrote {REPORTS / 'lgb_ranker_final.pkl'} (canonical seed=42 model)")

    # full metric suite for the canonical model, for the final report table
    test_feat_e5["score"] = canonical.predict(test_feat_e5[FINAL_FEATURES])
    rank_lists, rel_sets = [], []
    for case_id, g in test_feat_e5.groupby("case_id"):
        gold = true_gold.get(case_id, set())
        if not gold:
            continue
        rank_lists.append(g.sort_values("score", ascending=False)["article_id"].tolist())
        rel_sets.append(gold)
    full_metrics = evaluate_ranklists(rank_lists, rel_sets, ks=(1, 3, 5, 10))

    out = {
        "final_features": FINAL_FEATURES,
        "seeds": SEEDS,
        "per_seed_ndcg10": results,
        "mean_ndcg10": mean_ndcg,
        "sd_ndcg10": sd_ndcg,
        "canonical_seed": 42,
        "canonical_full_metrics": full_metrics,
    }
    with open(REPORTS / "final_model_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nCanonical model full metrics: {full_metrics}")
    print(f"Wrote {REPORTS / 'final_model_results.json'}")


if __name__ == "__main__":
    main()
