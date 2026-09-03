"""Generic dense embedding index, reused for both the statute corpus and the
training-case corpus (case-precedent kNN signal)."""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DENSE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # default, backward-compatible
_shared_models = {}


def get_model(model_name: str = DENSE_MODEL_NAME):
    """Cached per model_name -- callers that don't pass one get the original
    MiniLM singleton (all existing pipelines: features.py, retrieval.py's
    default DenseRetriever, etc. are unaffected). The final-architecture
    serving path (rank_core.py) requests e5-base explicitly by name, so both
    live in the cache side by side -- safe, unlike constructing two
    *independent* instances of the same model, which is what actually
    reproduced the SIGSEGV before (see git history / audit notes)."""
    if model_name not in _shared_models:
        # Pinned to CPU: this model reliably SIGSEGVs on PyTorch's MPS
        # backend when loaded inside certain process contexts (reproduced
        # standalone -- fine; inside module-import/FastAPI-startup context
        # -- crashes). CPU adds a few ms per query embed, irrelevant at this
        # request volume, and this is what most inference infra runs on
        # anyway -- not worth chasing the MPS bug for this project.
        _shared_models[model_name] = SentenceTransformer(model_name, device="cpu")
    return _shared_models[model_name]


class DenseIndex:
    def __init__(self, ids, texts, cache_path: Path = None, model_name: str = DENSE_MODEL_NAME,
                 query_prefix: str = ""):
        self.ids = list(ids)
        self.texts = list(texts)
        self.model_name = model_name
        self.query_prefix = query_prefix  # e.g. "query: " for e5-family models
        model = get_model(model_name)
        if cache_path and cache_path.exists():
            self.embeddings = np.load(cache_path)
        else:
            self.embeddings = model.encode(
                [self.query_prefix + t for t in self.texts], normalize_embeddings=True,
                batch_size=64, show_progress_bar=True,
            ).astype("float32")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.embeddings)

    @staticmethod
    def embed(texts, batch_size=64, model_name: str = DENSE_MODEL_NAME, query_prefix: str = ""):
        model = get_model(model_name)
        return model.encode(
            [query_prefix + t for t in texts], normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=True,
        ).astype("float32")

    def topk_batch(self, query_embs: np.ndarray, k: int, exclude_self_ids=None, chunk=1000):
        """Returns list (per query) of [(id, score), ...] top-k.
        exclude_self_ids: optional list same length as query_embs giving an
        id to mask out of its own results (leave-one-out for train queries)."""
        n = query_embs.shape[0]
        results = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims = query_embs[start:end] @ self.embeddings.T  # (chunk, corpus)
            for i in range(end - start):
                row = sims[i]
                if exclude_self_ids is not None:
                    ex = exclude_self_ids[start + i]
                    if ex in self._id_pos_cache():
                        row = row.copy()
                        row[self._id_pos_cache()[ex]] = -1e9
                idx = np.argpartition(-row, min(k, len(row) - 1))[:k]
                idx = idx[np.argsort(-row[idx])]
                results.append([(self.ids[j], float(row[j])) for j in idx])
        return results

    def _id_pos_cache(self):
        if not hasattr(self, "_pos"):
            self._pos = {v: i for i, v in enumerate(self.ids)}
        return self._pos
