# Pre-cleanup baseline (legacy reference run)

Frozen snapshot of the pipeline and results **before** the Phase 1 decontamination
work, so every later experiment can be compared against this exact starting
point. Nothing in this directory is overwritten by later phases.

## Split sizes (pre-cleanup)
- train: 30,000 (reservoir-sampled from 154,592, seed=20250902)
- valid: 5,000 (reservoir-sampled from 17,131, seed=20250902)
- test: 10,000 (reservoir-sampled from 32,508, seed=20250902)

## Confirmed contamination (from the audit, reproducible against the snapshots here)
- Exact fact-text overlap: train∩valid=36, train∩test=53, valid∩test=16
- Within-split exact duplicates: train=148, test=32, valid=0
- Near-duplicate (embedding cosine similarity, test→train, n=1000 sample):
  p50=0.882, p90=0.924, p95=0.935, p99=0.967, p99.5=0.982; 0.6% > 0.98, 1.8% > 0.95, 30% > 0.90
- Boilerplate/degenerate facts (regex "与一审相同" family or <60 chars): ~0.2-0.3% per split

## Candidate generation config (pre-cleanup)
- N_BM25_CAND = 30, N_DENSE_CAND = 30, N_CASE_NEIGHBORS = 15, N_CASE_VOTE_CAND = 15
- BM25: `rank_bm25.BM25Okapi` with library defaults (k1=1.5, b=0.75, epsilon=0.25) -- untuned
- Tokenizer: raw `jieba.cut()`, no stopword removal, no punctuation stripping
- Dense model: `paraphrase-multilingual-MiniLM-L12-v2` (general-purpose multilingual paraphrase/STS model, not retrieval-tuned, not legal-domain-adapted)
- Case-precedent index: built from `train.parquet` only; self-exclusion by `case_id` only (no near-duplicate exclusion)
- Training candidate pool: gold union'd into candidates for TRAIN split only (`features.py`), not for valid/test

## Feature list (10 features, unchanged going forward unless a phase says otherwise)
`bm25_score, bm25_rank, dense_score, dense_rank, case_vote_score, case_vote_count, jaccard, fact_len, article_len, article_prior`

## Random-state behavior (pre-cleanup)
- `data_prep.py` sampling: seeded (`SEED = 20250902`) -- reproducible
- `train_ranker.py` LGBMRanker: **no `random_state` set** -- not reproducible run-to-run
- `evaluate_full.py` bootstrap: seeded (`seed=0`) -- reproducible given fixed per-query score arrays

## Headline results (pre-cleanup) -- see architecture_comparison.csv for full table
| Architecture | NDCG@1 | NDCG@10 | Recall@10 | MRR |
|---|---|---|---|---|
| BM25-only | 0.204 | 0.324 | 0.457 | 0.305 |
| Dense-only | 0.109 | 0.202 | 0.316 | 0.188 |
| Case-precedent-kNN-only | 0.584 | 0.727 | 0.867 | 0.688 |
| Logistic regression | 0.557 | 0.706 | 0.861 | 0.665 |
| **LightGBM LambdaRank L2 (original, unseeded)** | **0.669** | **0.7886** | **0.905** | **0.759** |

## Ablation (re-run with `random_state=42` for this specific check -- see ablation_results.json)
Full model (seed=42): NDCG@10 = 0.7903 (vs. 0.7886 unseeded original -- ~0.002 run-to-run variance observed)
- minus case_precedent: 0.6930 (-0.0973) -- by far the largest drop
- minus priors_lengths: 0.7567 (-0.0337)
- minus bm25: 0.7741 (-0.0162)
- minus jaccard: 0.7826 (-0.0078)
- minus dense: 0.7864 (-0.0039)

## Dependencies
See `requirements_freeze.txt` (`pip freeze` at time of this run) and `python_version.txt`.
`requirements.txt` in the repo root has **no version pins** -- this freeze file is the only
record of exact versions actually used for the pre-cleanup numbers.

## Environment
Apple M3 Pro, 12 cores, macOS. PyTorch pinned to CPU device (MPS segfaults in this
process context -- see `dense_index.py`/`serve.py` comments). `OMP_NUM_THREADS=1`,
`KMP_DUPLICATE_LIB_OK=TRUE` required to avoid a LightGBM/PyTorch native OpenMP conflict.
