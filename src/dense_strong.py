"""
Phase 4: controlled dense-retrieval experiment.

Swaps ONLY the L1 statute dense retriever's embedding model
(paraphrase-multilingual-MiniLM-L12-v2 -> intfloat/multilingual-e5-base, a
model actually trained with a contrastive retrieval objective, not
paraphrase/STS) while holding everything else fixed: same BM25, same
case-precedent k-NN (still MiniLM -- that component is validated separately
in Phase 5 and is explicitly out of scope for this swap), same candidate
pools, same LambdaRank hyperparameters.

Answers three questions with evidence, not assumption:
  A. Does a retrieval-tuned model actually retrieve better standalone?
  B. Does it raise the L1 candidate-recall ceiling?
  C. Does it add incremental value on top of the LambdaRank ensemble, given
     case-precedent already captures much of what dense retrieval could
     offer?
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# train_ranker (lightgbm) MUST import before anything that touches torch
# (sentence-transformers) -- see rank_core.py's docstring for the reproduced
# SIGSEGV this import order avoids.
from train_ranker import FEATURE_COLS, load_grouped, train_lambdarank

import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

from retrieval import load_article_corpus
from metrics import evaluate_ranklists
from eval_utils import load_true_gold, score_model

PROCESSED = Path(__file__).parent.parent / "data/processed"
E5_MODEL = "intfloat/multilingual-e5-base"


def standalone_eval(model, article_ids, article_embs, queries_df, k=50):
    # IMPORTANT: takes an already-loaded model rather than constructing its
    # own -- a second independent SentenceTransformer instance in the same
    # process reliably SIGSEGVs (reproduced twice now; same class of bug as
    # dense_index.py's shared-model fix). Always reuse the caller's instance.
    rank_lists, relevant_sets = [], []
    q_texts = ["query: " + f for f in queries_df["fact"]]
    q_embs = model.encode(q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    for i, (_, row) in enumerate(queries_df.iterrows()):
        sims = article_embs @ q_embs[i]
        order = np.argsort(-sims)[:k]
        rank_lists.append([article_ids[j] for j in order])
        relevant_sets.append(set(json.loads(row["relevant_articles"])))
    return evaluate_ranklists(rank_lists, relevant_sets, ks=(1, 3, 5, 10, 30)), q_embs


def main():
    corpus = load_article_corpus()
    article_ids = list(corpus.keys())

    print("Embedding statute corpus with e5-base ('passage: ' prefix)...")
    e5_model = SentenceTransformer(E5_MODEL, device="cpu")
    article_texts = ["passage: " + corpus[a] for a in article_ids]
    article_embs = e5_model.encode(article_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    valid_df = pd.read_parquet(PROCESSED / "valid_clean.parquet")
    test_df = pd.read_parquet(PROCESSED / "test_clean.parquet")

    print("\n=== A. Standalone dense retrieval, e5-base, clean valid ===")
    valid_metrics, valid_q_embs = standalone_eval(e5_model, article_ids, article_embs, valid_df)
    for k, v in valid_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== A. Standalone dense retrieval, e5-base, clean TEST ===")
    test_metrics, test_q_embs = standalone_eval(e5_model, article_ids, article_embs, test_df)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # MiniLM comparison deliberately run as a separate process (eval_l1_dense_clean.py)
    # -- see its docstring for why. Loaded and merged into the final report below.
    import subprocess, sys as _sys
    print("\n=== A (comparison). Running MiniLM standalone eval as a separate process ===")
    subprocess.run([_sys.executable, "eval_l1_dense_clean.py"], check=True)
    old_minilm_metrics = json.load(open("../reports/minilm_dense_clean_results.json"))

    print(f"\n=== B. Candidate-recall ceiling @30 (N_DENSE_CAND=30): "
          f"e5-base={test_metrics['recall@30']:.4f} vs MiniLM={old_minilm_metrics['recall@30']:.4f} ===")

    # save A/B now -- these are already expensive and complete; don't lose
    # them if C (the long checkpointed embedding step) gets interrupted again
    partial = {
        "e5_standalone_valid": valid_metrics,
        "e5_standalone_test": test_metrics,
        "minilm_standalone_test_clean": old_minilm_metrics,
    }
    with open("../reports/dense_strong_results.json", "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)
    print("(saved partial A/B results to ../reports/dense_strong_results.json)")

    # === C. incremental contribution when swapped into the LambdaRank feature set ===
    print("\n=== C. Rebuilding dense_score/dense_rank with e5-base for existing candidate pairs ===")
    train_feat_full = pd.read_parquet(PROCESSED / "train_clean_features.parquet")
    valid_feat = pd.read_parquet(PROCESSED / "valid_clean_features.parquet")
    test_feat = pd.read_parquet(PROCESSED / "test_clean_features.parquet")

    # Steps A/B already gave definitive standalone/candidate-recall evidence
    # on the FULL valid+test sets. This step only needs to confirm the
    # DIRECTION and rough MAGNITUDE of e5's incremental contribution inside
    # the LambdaRank ensemble, so the train side is subsampled -- an
    # explicit, documented compute-scoping decision (a full 45K-query
    # re-embed on single-threaded CPU took >60min and was killed, likely by
    # a max-runtime limit; re-embedding is also checkpointed below so a
    # second interruption doesn't lose progress).
    TRAIN_SUBSAMPLE = 8000
    train_df_full = pd.read_parquet(PROCESSED / "train_clean.parquet")
    sub_case_ids = set(train_df_full["case_id"].sample(n=TRAIN_SUBSAMPLE, random_state=42))
    train_df = train_df_full[train_df_full["case_id"].isin(sub_case_ids)]
    train_feat = train_feat_full[train_feat_full["case_id"].isin(sub_case_ids)]
    print(f"(train subsampled to {len(train_df)} of {len(train_df_full)} for this check)")

    all_case_ids = pd.concat([train_df["case_id"], valid_df["case_id"], test_df["case_id"]]).tolist()
    all_facts = pd.concat([train_df["fact"], valid_df["fact"], test_df["fact"]]).tolist()

    ckpt_path = PROCESSED / "emb_cache" / "e5_query_embeddings_ckpt.npz"

    def embed_checkpointed(model, texts, ckpt_path, batch_size=64):
        """Encode in chunks, saving progress after every chunk -- a second
        run after an interruption resumes instead of restarting."""
        n = len(texts)
        if ckpt_path.exists():
            data = np.load(ckpt_path, allow_pickle=True)
            done = int(data["done"])
            embs = data["embs"]
            print(f"  resuming from checkpoint: {done}/{n} already embedded")
        else:
            done = 0
            embs = np.zeros((n, model.get_sentence_embedding_dimension()), dtype="float32")
        chunk = batch_size * 10
        while done < n:
            end = min(done + chunk, n)
            batch = ["query: " + t for t in texts[done:end]]
            embs[done:end] = model.encode(batch, normalize_embeddings=True, batch_size=batch_size)
            done = end
            np.savez(ckpt_path, embs=embs, done=done)
            print(f"  embedded {done}/{n}", flush=True)
        return embs

    print(f"Embedding {len(all_case_ids)} queries with e5-base for feature rebuild (checkpointed)...")
    q_embs_all = embed_checkpointed(e5_model, all_facts, ckpt_path)
    case_to_emb = dict(zip(all_case_ids, q_embs_all))
    article_to_emb = dict(zip(article_ids, article_embs))

    def rebuild(feat_df):
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

    train_feat_e5 = rebuild(train_feat)
    valid_feat_e5 = rebuild(valid_feat)
    test_feat_e5 = rebuild(test_feat)
    train_feat_e5.to_parquet(PROCESSED / "train_clean_e5_features.parquet", index=False)
    valid_feat_e5.to_parquet(PROCESSED / "valid_clean_e5_features.parquet", index=False)
    test_feat_e5.to_parquet(PROCESSED / "test_clean_e5_features.parquet", index=False)

    def grouped(df):
        df = df.sort_values("case_id").reset_index(drop=True)
        return df, df.groupby("case_id", sort=False).size().tolist()

    train_g, train_groups = grouped(train_feat_e5)
    valid_g, valid_groups = grouped(valid_feat_e5)
    true_gold = load_true_gold("_clean")

    print("\nTraining LambdaRank with e5-based dense features (seed=42)...")
    model_e5 = train_lambdarank(train_g, train_groups, valid_g, valid_groups, seed=42)
    metrics_e5, _ = score_model(model_e5, FEATURE_COLS, test_feat_e5, true_gold)
    print(f"Full model w/ e5 dense features: NDCG@10={metrics_e5['ndcg@10']:.4f}")

    out = {
        "e5_standalone_valid": valid_metrics,
        "e5_standalone_test": test_metrics,
        "minilm_standalone_test_clean": old_minilm_metrics,
        "full_model_with_e5_dense_ndcg10": metrics_e5["ndcg@10"],
        "full_model_with_e5_dense_full": metrics_e5,
    }
    with open("../reports/dense_strong_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nWrote ../reports/dense_strong_results.json")


if __name__ == "__main__":
    main()
