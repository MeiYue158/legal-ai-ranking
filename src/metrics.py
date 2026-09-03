"""Ranking metrics: Recall@k (candidate-generation ceiling), NDCG@k, MRR.
Multi-relevant-document aware (a case can have >1 correct article, so this
is genuinely a "graded multi-relevant retrieval" evaluation, not top-1
accuracy)."""
import math
import numpy as np


def recall_at_k(ranked_ids, relevant_set, k):
    topk = set(ranked_ids[:k])
    if not relevant_set:
        return None
    return len(topk & relevant_set) / len(relevant_set)


def ndcg_at_k(ranked_ids, relevant_set, k):
    if not relevant_set:
        return None
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids, relevant_set):
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranklists(rank_lists, relevant_sets, ks=(1, 3, 5, 10)):
    """rank_lists: list of ranked doc-id lists (one per query).
    relevant_sets: list of sets of true-relevant doc ids (same order)."""
    out = {f"recall@{k}": [] for k in ks}
    out.update({f"ndcg@{k}": [] for k in ks})
    out["mrr"] = []
    for ranked, rel in zip(rank_lists, relevant_sets):
        for k in ks:
            r = recall_at_k(ranked, rel, k)
            n = ndcg_at_k(ranked, rel, k)
            if r is not None:
                out[f"recall@{k}"].append(r)
                out[f"ndcg@{k}"].append(n)
        out["mrr"].append(mrr(ranked, rel))
    return {name: float(np.mean(vals)) for name, vals in out.items()}


def bootstrap_ci(per_query_values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_query_values)
    n = len(arr)
    means = [arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)
