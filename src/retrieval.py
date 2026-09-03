"""
Phase 2: L1 candidate retrieval over the 451-article statute corpus.

Two independent retrievers, matching how real first-stage retrieval is
usually built in production search: a sparse lexical retriever (BM25) and a
dense embedding retriever. Either can serve candidates to the L2 re-ranker;
we keep both so we can quantify how much each contributes (see evaluate.py).
"""
import json
from pathlib import Path
from functools import lru_cache

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from dense_index import get_model, DENSE_MODEL_NAME

ARTICLES_JSON = Path(__file__).parent.parent / "data/statutes/articles.json"
EMB_CACHE = Path(__file__).parent.parent / "data/statutes/article_embeddings.npy"


def load_article_corpus():
    """Collapse the 503 parsed entries (incl. '133_1'-style sub-articles) to
    one document per base article number (451 of them), matching CAIL's
    label granularity."""
    raw = json.load(open(ARTICLES_JSON, encoding="utf-8"))
    corpus = {}
    for v in raw.values():
        n = v["article_no"]
        corpus.setdefault(n, []).append(v["text"])
    return {n: "\n".join(texts) for n, texts in corpus.items()}


@lru_cache(maxsize=1)
def _tokenizer_cache():
    return {}


def tokenize(text: str):
    return list(jieba.cut(text))


class BM25Retriever:
    def __init__(self, corpus: dict):
        self.article_ids = list(corpus.keys())
        self.texts = [corpus[i] for i in self.article_ids]
        tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def score_all(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def topk(self, query: str, k: int):
        scores = self.score_all(query)
        idx = np.argsort(-scores)[:k]
        return [(self.article_ids[i], float(scores[i])) for i in idx]


class DenseRetriever:
    def __init__(self, corpus: dict, model_name: str = DENSE_MODEL_NAME,
                 emb_cache: Path = EMB_CACHE, doc_prefix: str = "", query_prefix: str = ""):
        """Defaults (model_name=MiniLM, no prefixes, EMB_CACHE) are exactly
        the original behavior -- every existing caller is unaffected. The
        final-architecture serving path (rank_core.py) passes e5-base +
        its own cache path + "query:"/"passage:" prefixes explicitly.

        model_name is looked up via dense_index.get_model()'s per-name
        cache, so this NEVER constructs a second independent instance of a
        model already loaded elsewhere in the process -- that's what
        actually reproduced the SIGSEGV before, not "two different models
        coexisting" per se."""
        self.article_ids = list(corpus.keys())
        self.texts = [corpus[i] for i in self.article_ids]
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.model = get_model(model_name)
        if emb_cache.exists():
            self.embeddings = np.load(emb_cache)
        else:
            self.embeddings = self.model.encode(
                [doc_prefix + t for t in self.texts], normalize_embeddings=True, show_progress_bar=True
            )
            emb_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_cache, self.embeddings)

    def embed_queries(self, queries, batch_size=64):
        return self.model.encode(
            [self.query_prefix + q for q in queries], normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=True,
        )

    def score_all(self, query_emb: np.ndarray) -> np.ndarray:
        return self.embeddings @ query_emb

    def topk_from_emb(self, query_emb: np.ndarray, k: int):
        scores = self.score_all(query_emb)
        idx = np.argsort(-scores)[:k]
        return [(self.article_ids[i], float(scores[i])) for i in idx]


if __name__ == "__main__":
    corpus = load_article_corpus()
    print(f"corpus size: {len(corpus)} articles")
    bm25 = BM25Retriever(corpus)
    q = "被告人酒后持刀将被害人捅伤，经鉴定为重伤二级。"
    print("BM25 top5 for sample query:")
    for art_id, score in bm25.topk(q, 5):
        print(f"  art {art_id}  score={score:.2f}  {corpus[art_id][:40]}")
