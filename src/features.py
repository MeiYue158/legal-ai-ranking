"""
Phase 3: Feature engineering for the L2 re-ranker.

For each query (case fact) we assemble a candidate pool from THREE signals:
  1. BM25 over statute text            (L1 lexical)
  2. Dense embedding over statute text  (L1 semantic)
  3. Case-precedent kNN vote            (retrieve similar TRAIN cases by dense
     similarity, let their gold articles "vote" -- this is the "query
     similar cases via retrieval" module from the original project mindmap,
     repurposed as a ranking *feature* instead of a standalone LLM-facing
     output)

Candidate pool = union of all three, so the L2 model can recover from any
one signal's blind spots. Each row in the output is one (query, candidate
article) pair with engineered features + a binary relevance label, grouped
by case_id for LightGBM's LambdaRank objective.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval import load_article_corpus, BM25Retriever, DenseRetriever, tokenize
from dense_index import DenseIndex

PROCESSED = Path(__file__).parent.parent / "data/processed"
CACHE = Path(__file__).parent.parent / "data/processed/emb_cache"

N_BM25_CAND = 30
N_DENSE_CAND = 30
N_CASE_NEIGHBORS = 15
N_CASE_VOTE_CAND = 15  # top articles by case-vote score also added to pool


def build_case_index(train_df, suffix=""):
    ids = train_df["case_id"].tolist()
    texts = train_df["fact"].tolist()
    idx = DenseIndex(ids, texts, cache_path=CACHE / f"train{suffix}_case_embeddings.npy")
    return idx


def case_id_to_articles(train_df):
    m = {}
    for _, r in train_df.iterrows():
        m[r["case_id"]] = set(json.loads(r["relevant_articles"]) if isinstance(r["relevant_articles"], str) else r["relevant_articles"])
    return m


def build_dataset(split_name, df, statute_corpus, bm25, statute_dense_index,
                   statute_dense_embs, case_index, case_articles_map,
                   train_article_prior, exclude_self=False):
    df = df.copy()
    df["relevant_articles"] = df["relevant_articles"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x)

    # batch-embed all queries once (fast, vectorized)
    query_embs = DenseIndex.embed(df["fact"].tolist())

    case_ids_for_exclude = df["case_id"].tolist() if exclude_self else None
    case_neighbor_lists = case_index.topk_batch(
        query_embs, N_CASE_NEIGHBORS, exclude_self_ids=case_ids_for_exclude)

    article_tokens_cache = {aid: set(tokenize(txt)) for aid, txt in statute_corpus.items()}

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        fact = row["fact"]
        gold = set(row["relevant_articles"])
        q_emb = query_embs[i]

        bm25_top = bm25.topk(fact, N_BM25_CAND)
        dense_scores_all = statute_dense_embs @ q_emb
        dense_order = np.argsort(-dense_scores_all)[:N_DENSE_CAND]
        dense_top = [(statute_dense_index.article_ids[j], float(dense_scores_all[j])) for j in dense_order]

        bm25_map = {aid: (rank, score) for rank, (aid, score) in enumerate(bm25_top)}
        dense_map = {aid: (rank, score) for rank, (aid, score) in enumerate(dense_top)}

        # case-precedent votes
        vote_score, vote_count = {}, {}
        for neighbor_id, sim in case_neighbor_lists[i]:
            for art in case_articles_map.get(neighbor_id, set()):
                vote_score[art] = vote_score.get(art, 0.0) + sim
                vote_count[art] = vote_count.get(art, 0) + 1
        top_vote_articles = sorted(vote_score, key=lambda a: -vote_score[a])[:N_CASE_VOTE_CAND]

        base_candidates = set(bm25_map) | set(dense_map) | set(top_vote_articles)
        candidate_ids = (base_candidates | gold) if split_name == "train" else base_candidates
        # (train split keeps `| gold` so every true positive is guaranteed a
        #  training row even on rare recall misses -- standard LTR practice;
        #  valid/test do NOT get this, so their metrics reflect real recall.)

        fact_tokens = set(tokenize(fact))
        for aid in candidate_ids:
            b_rank, b_score = bm25_map.get(aid, (999, 0.0))
            d_rank, d_score = dense_map.get(aid, (999, 0.0))
            v_score = vote_score.get(aid, 0.0)
            v_count = vote_count.get(aid, 0)
            a_tokens = article_tokens_cache.get(aid, set())
            union = fact_tokens | a_tokens
            jaccard = len(fact_tokens & a_tokens) / len(union) if union else 0.0

            rows.append({
                "case_id": row["case_id"],
                "article_id": aid,
                "bm25_score": b_score,
                "bm25_rank": b_rank,
                "dense_score": d_score,
                "dense_rank": d_rank,
                "case_vote_score": v_score,
                "case_vote_count": v_count,
                "jaccard": jaccard,
                "fact_len": len(fact),
                "article_len": len(statute_corpus.get(aid, "")),
                "article_prior": train_article_prior.get(aid, 0.0),
                "label": int(aid in gold),
            })
        if (i + 1) % 2000 == 0:
            print(f"  [{split_name}] {i+1}/{len(df)} queries featurized "
                  f"({(i+1)/(time.time()-t0):.0f} q/s)")
    return pd.DataFrame(rows)


def compute_article_prior(train_df):
    counts = {}
    total = 0
    for _, r in train_df.iterrows():
        arts = json.loads(r["relevant_articles"]) if isinstance(r["relevant_articles"], str) else r["relevant_articles"]
        for a in arts:
            counts[a] = counts.get(a, 0) + 1
            total += 1
    return {a: c / total for a, c in counts.items()}


def main(suffix=""):
    """suffix="" reproduces the original (pre-cleanup) feature files exactly
    (never overwritten by this change). suffix="_clean" builds features from
    the decontaminated splits (train_clean/valid_clean/test_clean.parquet)
    into separately-named *_clean_features.parquet files."""
    statute_corpus = load_article_corpus()
    bm25 = BM25Retriever(statute_corpus)
    statute_dense = DenseRetriever(statute_corpus)  # also builds/caches article_embeddings.npy

    train_df = pd.read_parquet(PROCESSED / f"train{suffix}.parquet")
    valid_df = pd.read_parquet(PROCESSED / f"valid{suffix}.parquet")
    test_df = pd.read_parquet(PROCESSED / f"test{suffix}.parquet")

    case_index = build_case_index(train_df, suffix=suffix)
    case_articles_map = case_id_to_articles(train_df)
    article_prior = compute_article_prior(train_df)

    for name, df, excl in [("train", train_df, True), ("valid", valid_df, False), ("test", test_df, False)]:
        out_path = PROCESSED / f"{name}{suffix}_features.parquet"
        feat_df = build_dataset(name, df, statute_corpus, bm25, statute_dense,
                                 statute_dense.embeddings, case_index, case_articles_map,
                                 article_prior, exclude_self=excl)
        feat_df.to_parquet(out_path, index=False)
        pos_rate = feat_df["label"].mean()
        print(f"{name}{suffix}: {len(feat_df)} rows, {feat_df['case_id'].nunique()} queries, "
              f"positive rate {pos_rate:.3f} -> {out_path}")


if __name__ == "__main__":
    import sys
    main(suffix="_clean" if len(sys.argv) > 1 and sys.argv[1] == "clean" else "")
