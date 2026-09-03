"""
Shared ranking engine, extracted out of serve.py so both the HTTP service
and the LLM agent call the exact same candidate generation + feature
engineering + L2 scoring logic instead of two independently-maintained
copies.

This is the FINAL, audited architecture (see reports/FINAL_TECHNICAL_REPORT.md):
  - dense retrieval upgraded to e5-base for BOTH the statute L1 retriever
    and the case-precedent k-NN index -- unified onto one embedding model
    rather than running MiniLM and e5 side by side. Two DIFFERENT models
    loaded independently in one process is what reproduced the SIGSEGV
    below; ONE model shared via dense_index.get_model()'s per-name cache
    does not have that problem, which is why unifying (rather than keeping
    MiniLM for case-precedent) was the safe choice here.
  - article_prior REMOVED (Phase 6 finding: cost 7.2 points of TAIL-article
    hit@10 for a 0.008 aggregate NDCG@10 gain -- not a good trade).
  - e5 is a retrieval-tuned model and expects "query: "/"passage: " prefixes;
    applied consistently below (query-vs-passage for statutes, asymmetric;
    query-vs-query for case-to-case, which is the correct symmetric-task
    convention for e5).

Import-order note (load-bearing, not decorative): `train_ranker` (lightgbm)
MUST be imported, and its model loaded via joblib, before anything
constructs a SentenceTransformer/torch model in this process. Reproduced
repeatedly: doing it the other way around SIGSEGVs on macOS from LightGBM's
and PyTorch's bundled OpenMP runtimes conflicting.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_final_model import FINAL_FEATURES  # lightgbm import happens inside here, must come first

import torch
torch.set_num_threads(1)

from dense_index import DenseIndex
from retrieval import load_article_corpus, BM25Retriever, DenseRetriever, tokenize

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data/processed"
E5_MODEL = "intfloat/multilingual-e5-base"
STATUTE_E5_CACHE = ROOT / "data/statutes/article_embeddings_e5.npy"
STATUTE_E5_IDS = ROOT / "data/statutes/article_ids_e5.json"
TRAIN_E5_CACHE = PROCESSED / "emb_cache/train_full_e5_embeddings.npy"


class RankingEngine:
    def load(self, model_path=None, data_suffix="_clean"):
        """Loads the final, e5-based, article_prior-free architecture by
        default. Pass model_path explicitly to load a different checkpoint
        (e.g. reports/lgb_ranker.pkl, the pre-audit model) -- note that
        older checkpoints expect the 10-feature set including article_prior
        and will NOT work with this engine's feature computation; use the
        matching git history / a dedicated legacy script for those."""
        if model_path is None:
            model_path = ROOT / "reports/lgb_ranker_final_e5_noprior.pkl"
        self.model = joblib.load(model_path)
        self.model_path = str(model_path)
        self.final_features = FINAL_FEATURES

        self.corpus = load_article_corpus()
        self.bm25 = BM25Retriever(self.corpus)
        self.article_tokens = {aid: set(tokenize(txt)) for aid, txt in self.corpus.items()}

        self.dense = DenseRetriever(
            self.corpus, model_name=E5_MODEL, emb_cache=STATUTE_E5_CACHE,
            doc_prefix="passage: ", query_prefix="query: ",
        )
        # sanity check: the cached embedding order must match this corpus's
        # key order, since DenseRetriever pairs them positionally
        cached_ids = json.load(open(STATUTE_E5_IDS))
        assert cached_ids == self.dense.article_ids, (
            "article_embeddings_e5.npy order does not match load_article_corpus() "
            "-- regenerate the e5 statute cache before serving"
        )
        self.dense.model.encode(["query: warmup"], normalize_embeddings=True)  # see docstring: main-thread warmup

        train_path = PROCESSED / f"train{data_suffix}.parquet"
        if not train_path.exists():
            data_suffix = ""
            train_path = PROCESSED / "train.parquet"
        train_df = pd.read_parquet(train_path)
        self.train_df = train_df
        self.case_index = DenseIndex(
            train_df["case_id"].tolist(), train_df["fact"].tolist(),
            cache_path=TRAIN_E5_CACHE, model_name=E5_MODEL, query_prefix="query: ",
        )
        self.case_articles_map = {}
        self.case_fact_map = {}
        for _, r in train_df.iterrows():
            self.case_articles_map[r["case_id"]] = set(json.loads(r["relevant_articles"]))
            self.case_fact_map[r["case_id"]] = {
                "fact": r["fact"],
                "accusation": json.loads(r["accusation"]),
                "imprisonment": int(r["imprisonment"]),
                "life_imprisonment": bool(r["life_imprisonment"]),
                "death_penalty": bool(r["death_penalty"]),
            }
        return self

    def _embed_query(self, fact: str) -> np.ndarray:
        return DenseIndex.embed([fact], model_name=E5_MODEL, query_prefix="query: ")[0]

    def _candidates(self, fact: str, q_emb: np.ndarray, n_case_neighbors=15):
        bm25_top = self.bm25.topk(fact, 30)
        dense_scores_all = self.dense.embeddings @ q_emb
        dense_order = np.argsort(-dense_scores_all)[:30]
        dense_top = [(self.dense.article_ids[j], float(dense_scores_all[j])) for j in dense_order]

        neighbors = self.case_index.topk_batch(q_emb[None, :], n_case_neighbors)[0]
        vote_score, vote_count = {}, {}
        for cid, sim in neighbors:
            for art in self.case_articles_map.get(cid, set()):
                vote_score[art] = vote_score.get(art, 0.0) + sim
                vote_count[art] = vote_count.get(art, 0) + 1

        bm25_map = {aid: (r, s) for r, (aid, s) in enumerate(bm25_top)}
        dense_map = {aid: (r, s) for r, (aid, s) in enumerate(dense_top)}
        top_vote = sorted(vote_score, key=lambda a: -vote_score[a])[:15]
        candidates = set(bm25_map) | set(dense_map) | set(top_vote)
        return candidates, bm25_map, dense_map, vote_score, vote_count, neighbors

    def rank(self, fact: str, top_k: int = 10):
        """Returns (results, meta) where results is
        [{"article": int, "score": float}, ...] and meta carries the
        candidate count + the case-neighbor list (reused by the agent for
        precedent citation without a second embedding pass)."""
        q_emb = self._embed_query(fact)
        candidates, bm25_map, dense_map, vote_score, vote_count, neighbors = self._candidates(fact, q_emb)

        fact_tokens = set(tokenize(fact))
        rows, ids = [], []
        for aid in candidates:
            b_rank, b_score = bm25_map.get(aid, (999, 0.0))
            d_rank, d_score = dense_map.get(aid, (999, 0.0))
            a_tokens = self.article_tokens.get(aid, set())
            union = fact_tokens | a_tokens
            jaccard = len(fact_tokens & a_tokens) / len(union) if union else 0.0
            rows.append([
                b_score, b_rank, d_score, d_rank,
                vote_score.get(aid, 0.0), vote_count.get(aid, 0), jaccard,
                len(fact), len(self.corpus.get(aid, "")),
            ])
            ids.append(aid)

        X = pd.DataFrame(rows, columns=self.final_features)
        scores = self.model.predict(X)
        order = np.argsort(-scores)[:top_k]
        results = [{"article": int(ids[i]), "score": float(scores[i])} for i in order]
        meta = {"n_candidates": len(candidates), "neighbors": neighbors}
        return results, meta

    def similar_cases(self, fact: str, top_k: int = 3, neighbors=None):
        """Top-k similar precedent cases with their real outcomes, for the
        agent to cite. Reuses the neighbor list from rank() if given."""
        if neighbors is None:
            q_emb = self._embed_query(fact)
            neighbors = self.case_index.topk_batch(q_emb[None, :], top_k)[0]
        out = []
        for cid, sim in neighbors[:top_k]:
            info = self.case_fact_map.get(cid, {})
            out.append({"case_id": cid, "similarity": float(sim), **info})
        return out

    def article_text(self, article_id: int) -> str:
        return self.corpus.get(article_id, "")
