"""
Phase 1: data contamination / split cleanup.

Operates on the existing 45,000-case sampled universe (30K/5K/10K train/
valid/test from data_prep.py) rather than re-sampling from the raw 204K CAIL
files -- that sampled pool IS the operating universe for every reported
result so far, and decontaminating it in place is the faithful fix.

Policy (documented, not just implemented):

1A. EXACT duplicates: conservative text normalization (strip + collapse
    whitespace only -- no aggressive rewriting) then exact string match.
    Any duplicate group is assigned to ONE split, never split across.

1B. NEAR duplicates: cosine similarity alone is NOT sufficient (explicitly
    rejected by the audit -- two genuinely different but similarly-phrased
    crimes would be wrongly merged). A pair is only treated as a
    near-duplicate CLUSTER MEMBER if ALL of:
      - cosine similarity >= 0.95
      - character 4-gram Jaccard >= 0.50   (real shared verbatim text, not
        just topical/stylistic similarity)
      - identical gold label set (relevant_articles) -- two independently
        different real crimes essentially never share an identical
        multi-label article set AND near-verbatim text by chance; this is
        the strongest corroborating signal that it's the same underlying
        case (co-defendant records, duplicate court filings) rather than
        two similar-but-distinct crimes.
      - NOT boilerplate/degenerate (see 1C) -- boilerplate similarity is a
        different phenomenon and must not be treated as case duplication.

1C. DEGENERATE/boilerplate facts: flagged (not silently dropped) via a
    regex for "same as first instance" appellate boilerplate, or raw length
    < 60 chars. Reported with metrics both included and excluded.

1D. Rebuild splits: union-find over exact + near-duplicate edges. Any
    cluster touching train is assigned wholly to train (protects eval-set
    purity at ~zero cost, since train isn't evaluated on). Any remaining
    valid+test-only cluster is assigned wholly to test (protects the final
    test set over the validation set). Singleton clusters keep their
    original split. Post-hoc assertion: zero exact-text overlap between
    every split pair.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dense_index import DenseIndex

PROCESSED = Path(__file__).parent.parent / "data/processed"
REPORTS = Path(__file__).parent.parent / "reports"

BOILERPLATE_RE = re.compile(r"与一审相同|同一审|二审审理查明的事实和证据与一审相同")
MIN_SUBSTANTIVE_LEN = 60

COS_CANDIDATE_THRESHOLD = 0.90   # cheap prefilter for pair generation
COS_DUP_THRESHOLD = 0.95         # actual near-duplicate decision threshold
JACCARD_4GRAM_THRESHOLD = 0.50


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def char_ngram_set(s: str, n: int = 4):
    s = s.replace(" ", "")
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_pool():
    frames = []
    for name in ["train", "valid", "test"]:
        df = pd.read_parquet(PROCESSED / f"{name}.parquet")
        df["orig_split"] = name
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool["norm_fact"] = pool["fact"].map(normalize_text)
    pool["is_degenerate"] = pool["fact"].apply(
        lambda t: bool(BOILERPLATE_RE.search(t)) or len(t) < MIN_SUBSTANTIVE_LEN
    )
    return pool


def find_exact_duplicate_groups(pool):
    groups = pool.groupby("norm_fact")["case_id"].apply(list)
    dup_groups = [g for g in groups if len(g) > 1]
    return dup_groups


def get_or_build_embeddings(pool):
    cache = PROCESSED / "emb_cache" / "full_pool_embeddings.npy"
    ids_cache = PROCESSED / "emb_cache" / "full_pool_ids.json"
    if cache.exists() and ids_cache.exists():
        ids = json.load(open(ids_cache))
        if ids == pool["case_id"].tolist():
            return np.load(cache)
    embs = DenseIndex.embed(pool["fact"].tolist())
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embs)
    json.dump(pool["case_id"].tolist(), open(ids_cache, "w"))
    return embs


def find_near_duplicate_pairs(pool, embs, chunk=1500):
    n = len(pool)
    ids = pool["case_id"].tolist()
    facts = pool["fact"].tolist()
    articles = [frozenset(json.loads(a)) for a in pool["relevant_articles"]]
    degenerate = pool["is_degenerate"].tolist()
    ngram_cache = {}

    def get_ngrams(i):
        if i not in ngram_cache:
            ngram_cache[i] = char_ngram_set(facts[i])
        return ngram_cache[i]

    pairs = []
    boundary_examples = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = embs[start:end] @ embs.T  # (chunk, n)
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i]
            candidates = np.where(row >= COS_CANDIDATE_THRESHOLD)[0]
            for j in candidates:
                if j <= i:
                    continue
                cos = float(row[j])
                if degenerate[i] or degenerate[j]:
                    continue  # boilerplate similarity != case duplication
                jac = jaccard(get_ngrams(i), get_ngrams(j))
                same_label = articles[i] == articles[j]
                is_dup = (cos >= COS_DUP_THRESHOLD and jac >= JACCARD_4GRAM_THRESHOLD and same_label)
                if 0.93 <= cos < 0.97 and len(boundary_examples) < 12:
                    boundary_examples.append({
                        "case_a": ids[i], "case_b": ids[j], "cosine": cos,
                        "jaccard_4gram": jac, "same_label": same_label,
                        "decision": "MERGE" if is_dup else "keep separate",
                        "fact_a": facts[i][:80], "fact_b": facts[j][:80],
                    })
                if is_dup:
                    pairs.append((ids[i], ids[j], cos, jac))
    return pairs, boundary_examples


def main():
    pool = load_pool()
    print(f"Pool: {len(pool)} cases ({pool['orig_split'].value_counts().to_dict()})")

    # --- 1A exact duplicates ---
    dup_groups = find_exact_duplicate_groups(pool)
    affected = sum(len(g) for g in dup_groups)
    cross_split_groups = 0
    id_to_split = dict(zip(pool["case_id"], pool["orig_split"]))
    for g in dup_groups:
        if len({id_to_split[c] for c in g}) > 1:
            cross_split_groups += 1
    print(f"\n1A. Exact duplicate groups: {len(dup_groups)}, affected cases: {affected}, "
          f"cross-split groups: {cross_split_groups}")

    # --- 1B near duplicates ---
    print("\n1B. Embedding full pool for near-duplicate search...")
    embs = get_or_build_embeddings(pool)
    print("Searching for near-duplicate pairs (this scans the full pool)...")
    near_pairs, boundary_examples = find_near_duplicate_pairs(pool, embs)
    print(f"Near-duplicate pairs found (cos>={COS_DUP_THRESHOLD}, jaccard4>={JACCARD_4GRAM_THRESHOLD}, "
          f"same label, non-degenerate): {len(near_pairs)}")
    print("\nBoundary examples (cosine in [0.93, 0.97)):")
    for ex in boundary_examples[:8]:
        print(f"  cos={ex['cosine']:.3f} jac4={ex['jaccard_4gram']:.2f} "
              f"same_label={ex['same_label']} -> {ex['decision']}")
        print(f"    A: {ex['fact_a']}")
        print(f"    B: {ex['fact_b']}")

    # --- 1C degenerate ---
    n_degenerate = pool["is_degenerate"].sum()
    print(f"\n1C. Degenerate/boilerplate facts: {n_degenerate} / {len(pool)} "
          f"({n_degenerate/len(pool):.2%})")
    print(pool[pool["is_degenerate"]].groupby("orig_split").size().to_dict())

    # --- 1D rebuild splits ---
    uf = UnionFind(pool["case_id"].tolist())
    for g in dup_groups:
        for c in g[1:]:
            uf.union(g[0], c)
    for a, b, _, _ in near_pairs:
        uf.union(a, b)

    cluster_of = {cid: uf.find(cid) for cid in pool["case_id"]}
    pool["cluster"] = pool["case_id"].map(cluster_of)

    cluster_splits = pool.groupby("cluster")["orig_split"].apply(set)
    new_split = {}
    reassigned = 0
    for cluster_id, splits in cluster_splits.items():
        members = pool.loc[pool["cluster"] == cluster_id, "case_id"].tolist()
        if len(splits) == 1:
            assign = next(iter(splits))
        elif "train" in splits:
            assign = "train"
            reassigned += 1
        else:
            assign = "test"  # valid+test only cluster -> protect test
            reassigned += 1
        for m in members:
            new_split[m] = assign

    pool["new_split"] = pool["case_id"].map(new_split)
    print(f"\n1D. Clusters requiring cross-split reassignment: {reassigned}")
    print("New split sizes:", pool["new_split"].value_counts().to_dict())
    print("Old split sizes:", pool["orig_split"].value_counts().to_dict())

    moved = pool[pool["orig_split"] != pool["new_split"]]
    print(f"Cases moved to a different split than original: {len(moved)}")

    # --- assertions ---
    norm_by_split = {s: set(pool.loc[pool["new_split"] == s, "norm_fact"]) for s in ["train", "valid", "test"]}
    ov_tv = norm_by_split["train"] & norm_by_split["valid"]
    ov_tt = norm_by_split["train"] & norm_by_split["test"]
    ov_vt = norm_by_split["valid"] & norm_by_split["test"]
    print(f"\nPOST-CLEANUP exact overlap check: train∩valid={len(ov_tv)} "
          f"train∩test={len(ov_tt)} valid∩test={len(ov_vt)}")
    assert len(ov_tv) == 0 and len(ov_tt) == 0 and len(ov_vt) == 0, "exact overlap remains -- bug"

    # save clean splits
    for name in ["train", "valid", "test"]:
        sub = pool[pool["new_split"] == name].drop(
            columns=["orig_split", "new_split", "norm_fact", "cluster"])
        out = PROCESSED / f"{name}_clean.parquet"
        sub.to_parquet(out, index=False)
        print(f"wrote {out} ({len(sub)} rows, {sub['is_degenerate'].sum()} degenerate)")

    # contamination audit report
    audit = {
        "pool_size": len(pool),
        "exact_duplicate_groups": len(dup_groups),
        "exact_duplicate_affected_cases": int(affected),
        "exact_duplicate_cross_split_groups": cross_split_groups,
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_examples": [
            {"case_a": a, "case_b": b, "cosine": c, "jaccard_4gram": j} for a, b, c, j in near_pairs[:20]
        ],
        "boundary_examples": boundary_examples,
        "degenerate_count": int(n_degenerate),
        "degenerate_pct": float(n_degenerate / len(pool)),
        "clusters_reassigned_cross_split": reassigned,
        "cases_moved": int(len(moved)),
        "old_split_sizes": pool["orig_split"].value_counts().to_dict(),
        "new_split_sizes": pool["new_split"].value_counts().to_dict(),
        "post_cleanup_overlap": {"train_valid": len(ov_tv), "train_test": len(ov_tt), "valid_test": len(ov_vt)},
        "policy": {
            "cos_candidate_threshold": COS_CANDIDATE_THRESHOLD,
            "cos_dup_threshold": COS_DUP_THRESHOLD,
            "jaccard_4gram_threshold": JACCARD_4GRAM_THRESHOLD,
            "requires_identical_label_set": True,
            "excludes_degenerate_pairs": True,
        },
    }
    with open(REPORTS / "contamination_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {REPORTS / 'contamination_audit.json'}")


if __name__ == "__main__":
    main()
