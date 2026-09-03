"""
Phase 6: does the model mainly win on frequent statutes (popularity bias via
article_prior), or does it genuinely help on the long tail too?

Buckets the 451 articles into HEAD/MID/TAIL by TRAIN-set frequency only
(never test), plus a separate UNSEEN bucket for articles with zero train
occurrences. For each (query, gold-article) instance in the clean test set,
reports whether that specific article is hit in the top-10, broken out by
its bucket -- for the full model AND a model with article_prior removed, so
a popularity-bias effect would show up as a HEAD-vs-TAIL gap that shrinks
when article_prior is dropped.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

PROCESSED = "../data/processed"


def bucket_articles(train_df):
    counts = {}
    for arts_json in train_df["relevant_articles"]:
        for a in json.loads(arts_json):
            counts[a] = counts.get(a, 0) + 1
    from retrieval import load_article_corpus
    all_articles = set(load_article_corpus().keys())
    seen = sorted(counts, key=lambda a: -counts[a])
    n = len(seen)
    head = set(seen[: n // 3])
    mid = set(seen[n // 3: 2 * n // 3])
    tail = set(seen[2 * n // 3:])
    unseen = all_articles - set(seen)
    return {"HEAD": head, "MID": mid, "TAIL": tail, "UNSEEN_IN_TRAIN": unseen}, counts


def train_model(train_df, train_groups, valid_df, valid_groups, feature_cols, seed=42):
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
    return model


def analyze(model, feature_cols, test_df, true_gold, buckets, name):
    test_df = test_df.copy()
    test_df["score"] = model.predict(test_df[feature_cols])
    bucket_hits = {b: [] for b in buckets}
    bucket_candidate_recall = {b: [] for b in buckets}

    for case_id, g in test_df.groupby("case_id"):
        gold = true_gold.get(case_id, set())
        if not gold:
            continue
        ranked = g.sort_values("score", ascending=False)["article_id"].tolist()
        top10 = set(ranked[:10])
        candidates = set(g["article_id"])
        for article in gold:
            b = next((bn for bn, members in buckets.items() if article in members), None)
            if b is None:
                continue
            bucket_hits[b].append(1 if article in top10 else 0)
            bucket_candidate_recall[b].append(1 if article in candidates else 0)

    print(f"\n--- {name} ---")
    for b in buckets:
        n = len(bucket_hits[b])
        if n == 0:
            print(f"  {b}: no gold-article instances in test")
            continue
        hit_rate = np.mean(bucket_hits[b])
        cand_recall = np.mean(bucket_candidate_recall[b])
        print(f"  {b}: n={n} gold-article instances, hit@10={hit_rate:.4f}, candidate_recall={cand_recall:.4f}")
    return bucket_hits


def main():
    train_df_raw = pd.read_parquet(f"{PROCESSED}/train_clean.parquet")
    buckets, counts = bucket_articles(train_df_raw)
    print("Article frequency buckets (from TRAIN only):")
    for b, members in buckets.items():
        freqs = [counts.get(a, 0) for a in members]
        print(f"  {b}: {len(members)} articles, train freq range "
              f"[{min(freqs) if freqs else 0}, {max(freqs) if freqs else 0}]")

    def load_grouped(name):
        df = pd.read_parquet(f"{PROCESSED}/{name}_clean_features.parquet")
        df = df.sort_values("case_id").reset_index(drop=True)
        return df, df.groupby("case_id", sort=False).size().tolist()

    train_df, train_groups = load_grouped("train")
    valid_df, valid_groups = load_grouped("valid")
    test_df, _ = load_grouped("test")

    raw_test = pd.read_parquet(f"{PROCESSED}/test_clean.parquet")
    true_gold = {row["case_id"]: set(json.loads(row["relevant_articles"])) for _, row in raw_test.iterrows()}

    ALL_FEATURES = ["bm25_score", "bm25_rank", "dense_score", "dense_rank",
                     "case_vote_score", "case_vote_count", "jaccard",
                     "fact_len", "article_len", "article_prior"]
    NO_PRIOR = [c for c in ALL_FEATURES if c != "article_prior"]

    print("\nTraining FULL model...")
    full_model = train_model(train_df, train_groups, valid_df, valid_groups, ALL_FEATURES)
    full_hits = analyze(full_model, ALL_FEATURES, test_df, true_gold, buckets, "FULL model (with article_prior)")

    print("\nTraining model WITHOUT article_prior...")
    no_prior_model = train_model(train_df, train_groups, valid_df, valid_groups, NO_PRIOR)
    no_prior_hits = analyze(no_prior_model, NO_PRIOR, test_df, true_gold, buckets, "model WITHOUT article_prior")

    out = {
        "bucket_sizes": {b: len(m) for b, m in buckets.items()},
        "full_model_hit_rates": {b: float(np.mean(v)) if v else None for b, v in full_hits.items()},
        "no_prior_hit_rates": {b: float(np.mean(v)) if v else None for b, v in no_prior_hits.items()},
    }
    with open("../reports/head_mid_tail_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote ../reports/head_mid_tail_results.json")


if __name__ == "__main__":
    main()
