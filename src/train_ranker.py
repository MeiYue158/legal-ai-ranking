"""Phase 4: train the L2 re-ranker.

Two models trained on identical features, for comparison:
  (a) LightGBM LambdaRank (GBDT ranker) -- the classic "L2 ranking" workhorse
  (b) Logistic-regression pointwise baseline -- a much weaker ranker, kept
      specifically so we can show the LambdaRank objective (which optimizes
      ranking order directly) beats a pointwise classifier trained on the
      same features, not just "beats no model at all."
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = Path(__file__).parent.parent / "data/processed"
MODEL_DIR = Path(__file__).parent.parent / "reports"

FEATURE_COLS = [
    "bm25_score", "bm25_rank", "dense_score", "dense_rank",
    "case_vote_score", "case_vote_count", "jaccard",
    "fact_len", "article_len", "article_prior",
]


def load_grouped(name, suffix=""):
    df = pd.read_parquet(PROCESSED / f"{name}{suffix}_features.parquet")
    df = df.sort_values("case_id").reset_index(drop=True)
    groups = df.groupby("case_id", sort=False).size().tolist()
    return df, groups


def train_lambdarank(train_df, train_groups, valid_df, valid_groups, seed=None):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        eval_at=[1, 3, 5, 10],
        verbose=-1,
        random_state=seed,        # None preserves original (unseeded) behavior
        deterministic=seed is not None,
        force_row_wise=seed is not None,  # avoids a further nondeterminism source when seeded
    )
    model.fit(
        train_df[FEATURE_COLS], train_df["label"], group=train_groups,
        eval_set=[(valid_df[FEATURE_COLS], valid_df["label"])],
        eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_logreg(train_df):
    # feature scale varies wildly (jaccard in [0,1] vs. fact_len in the
    # hundreds), which is exactly what stalls lbfgs convergence -- scale first.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(train_df[FEATURE_COLS], train_df["label"])
    return model


def main():
    train_df, train_groups = load_grouped("train")
    valid_df, valid_groups = load_grouped("valid")

    print(f"train rows={len(train_df)} groups={len(train_groups)} pos_rate={train_df['label'].mean():.3f}")
    print(f"valid rows={len(valid_df)} groups={len(valid_groups)} pos_rate={valid_df['label'].mean():.3f}")

    lgb_model = train_lambdarank(train_df, train_groups, valid_df, valid_groups)
    logreg_model = train_logreg(train_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgb_model, MODEL_DIR / "lgb_ranker.pkl")
    joblib.dump(logreg_model, MODEL_DIR / "logreg_baseline.pkl")

    importances = sorted(zip(FEATURE_COLS, lgb_model.feature_importances_), key=lambda x: -x[1])
    print("LightGBM feature importances:")
    for name, imp in importances:
        print(f"  {name}: {imp}")

    print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
