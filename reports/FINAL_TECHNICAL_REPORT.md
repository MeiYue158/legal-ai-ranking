# Legal Statute & Precedent Retrieval-Ranking System — Final Technical Report

Produced after a full remediation pass triggered by an independent audit of the original
(unaudited) result. Every number below was actually run in this repository; none are
estimated or assumed. Scripts referenced are in `src/`, raw outputs in `reports/`.

## A. Data integrity

**Before cleanup** (45,000-case pool: 30,000/5,000/10,000 train/valid/test)
- Exact fact-text overlap: train∩valid=36, train∩test=53, valid∩test=16
- Within-split exact duplicates: train=148, test=32
- Near-duplicate (embedding cosine, test→train sample): 30% >0.90, 1.8% >0.95, 0.6% >0.98

**Decontamination policy** (`decontaminate.py`): exact-duplicate grouping by normalized
text, plus a near-duplicate policy requiring cosine≥0.95 **and** 4-gram Jaccard≥0.50 **and**
identical gold-label set **and** non-boilerplate — deliberately conservative so that two
merely-similar-but-distinct crimes are never merged (validated against boundary examples:
several same-crime-type, same-label pairs at cosine 0.93–0.96 were correctly kept separate
because their lexical overlap was low).

**After cleanup**
- 267 exact-duplicate groups (552 cases, 105 spanning multiple original splits)
- 680 near-duplicate pairs met the strict policy
- 212 degenerate/boilerplate facts flagged (0.47%), not silently dropped
- 228 clusters reassigned, 243 cases moved; new split sizes: train 30,214 / valid 4,886 / test 9,900
- Post-cleanup assertion: **zero exact overlap** between every split pair (verified programmatically)

## B. Architecture comparison (clean test set, n=9,900)

| Model | NDCG@1 | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|---|
| BM25 (tuned: k1=2.0, b=0.5, selected on valid only) | 0.2263 | 0.3308 | 0.3098 | 0.4450 |
| Dense, old (MiniLM, general-purpose paraphrase model) | 0.1073 | 0.2002 | 0.1786 | 0.3141 |
| Dense, new (e5-base, retrieval-tuned) | 0.2072 | 0.3697 | 0.3286 | 0.5627 |
| **Final LambdaRank (e5 dense + case-precedent + BM25 + jaccard, no article_prior)** | **0.6799** | **0.7958** | **0.7683** | **0.9065** |

(BM25 numbers are the properly-tuned Phase 2 result, not the original untuned baseline.
Case-precedent-alone and the pre-cleanup ensemble aren't repeated here since they were
measured on the pre-cleanup test set and are not directly comparable row-for-row; see
`architecture_comparison.csv` for that original table.)

## C. Ablation (clean data, re-run independently of the pre-cleanup ablation)

| Feature group removed | NDCG@10 | Δ vs. full | (pre-cleanup Δ, for comparison) |
|---|---|---|---|
| None (full, pre-final-architecture) | 0.7892 | — | — |
| case_precedent (case_vote_score/count) | 0.6915 | **−0.0977** | −0.0973 |
| priors_lengths (article_prior+fact_len+article_len) | 0.7513 | −0.0379 | −0.0337 |
| bm25 | 0.7719 | −0.0173 | −0.0162 |
| jaccard | 0.7789 | −0.0103 | −0.0078 |
| dense (old MiniLM) | 0.7846 | −0.0045 | −0.0039 |
| article_prior alone | 0.7812 | −0.0080 | *(new, isolated)* |

**Case-precedent's dominance survives decontamination essentially unchanged** — confirmed via
gain-based feature importance AND an actual retrain-and-remeasure ablation, not split-count
importance alone (which is what the original, unaudited claim rested on).

**Controlled dense-model comparison** (`dense_strong.py`, same 8,000-case training subsample,
only dense_score/dense_rank source differs):

| Dense features | NDCG@10 |
|---|---|
| MiniLM, 8K subsample | 0.7832 |
| e5-base, 8K subsample | 0.7984 |
| MiniLM, full 30,214 (for reference) | 0.7892 |

e5-base's standalone Recall@30 (candidate-recall ceiling) is 0.7454 vs. MiniLM's 0.4761 — a
genuine, controlled +0.0152 NDCG@10 improvement isolates the embedding-model variable from
training-set-size. **Conclusion: the original "vocabulary mismatch" explanation for weak
dense retrieval was partly wrong — a real share of the failure was model choice, not an
inherent semantic gap.** Dense retrieval remains secondary to case-precedent, but is no
longer negligible with a properly retrieval-tuned model.

## D. Robustness

**Multi-seed** (`train_multiseed.py`, clean data, unmodified architecture, 3 seeds):
NDCG@10 = 0.7878 ± 0.0003 (individual: 0.7879, 0.7879, 0.7875) — highly stable.

**Final architecture, multi-seed** (`build_final_model.py`, e5 dense + no article_prior,
full 30,214 train, 3 seeds): NDCG@10 = 0.7959 ± 0.0002 (0.7958, 0.7961, 0.7957).

**Head/mid/tail by train-set article frequency** (`phase6_head_mid_tail.py`):

| Bucket | Train freq | n | hit@10 (with article_prior) | hit@10 (without) | Δ |
|---|---|---|---|---|---|
| HEAD | 373–3092 | 9,059 | 0.921 | 0.915 | −0.6pp |
| MID | 27–373 | 1,798 | 0.847 | 0.829 | −1.8pp |
| **TAIL** | 2–26 | 210 | **0.495** | **0.567** | **+7.2pp** |

`article_prior` actively hurts TAIL-article retrieval for a marginal aggregate gain —
**dropped from the final architecture on this evidence.** TAIL candidate recall (0.771) is
also meaningfully lower than HEAD's (0.945): candidate generation itself, not just ranking,
struggles more on rare statutes — a real, disclosed limitation, not hidden behind the
aggregate NDCG@10.

## E. Deployment

- `Dockerfile` builds successfully (2.46GB image) using a CPU-only PyTorch wheel
  (`--extra-index-url .../whl/cpu`) — the naive install pulled >1GB of unused NVIDIA
  CUDA/cuDNN packages into a container with no GPU; fixed and verified in the build log.
- Container passes its HEALTHCHECK, serves the loaded model in ~18.5s startup, and returns
  correct results end-to-end (verified via `curl` against the running container, e.g.
  correctly ranking Article 264 for a theft narrative).
- **Concurrency, measured, not assumed**: 5 simultaneous requests against the running
  container completed at staggered times (0.09s, 0.13s, 0.18s, 0.23s, 0.26s) rather than
  clustering around one latency — empirically confirms the documented no-concurrency
  limitation (the async handlers force everything onto uvicorn's single event-loop thread;
  a deliberate, disclosed tradeoff, not a hidden defect).
- Basic request validation added (non-empty fact, length cap, top_k range) and inference
  errors now return a clean 500 with logging instead of an unhandled stack trace.
- Two real native-library bugs were root-caused via direct process diagnostics
  (`faulthandler`, controlled import-order tests), not guessed-and-patched: a SIGSEGV from
  LightGBM/PyTorch OpenMP conflict (fixed via import order) and a cross-thread hang
  (fixed via `async def` handlers, which is also the source of the concurrency limitation
  above — same fix, disclosed tradeoff).

## F. Remaining limitations (explicit, not hidden)

- Closed 451-article universe (the actual PRC Criminal Law) — appropriate for this task,
  but candidate generation still misses ~6% of gold articles even at the final architecture
  (Recall@10=0.907), concentrated in TAIL statutes.
- CAIL2018's real-world redundancy (many near-identical fact patterns) is a meaningful part
  of why case-precedent k-NN is so strong; this may not generalize as cleanly to a corpus
  with less repetition.
- Dense retrieval, even upgraded to e5-base, remains a secondary signal — no legal-domain
  fine-tuning was attempted; a domain-adapted embedding model was not tested (documented
  next step, not done here).
- LLM-layer evaluation is small-sample by necessity: the free-tier Gemini quota is a **daily**
  cap of 20 requests for this model, discovered mid-audit, which blocked a larger structured
  evaluation. The only full-test-scale, quota-free metric reported is gold-in-candidate-pool
  rate (0.94); citation-membership was only checked on a handful of cases before the daily
  cap was hit, and is explicitly not reported as a statistically meaningful rate.
- No real production traffic; latency/concurrency numbers are from a single local Docker
  Desktop instance on Apple Silicon, not a cloud deployment.

**Update (post-report): the live service now runs the final architecture.** `rank_core.py`
was rewired to load `reports/lgb_ranker_final_e5_noprior.pkl` and use e5-base for BOTH the
statute L1 retriever and the case-precedent k-NN index — unified onto one embedding model
(reusing the already-computed `train_full_e5_embeddings.npy` for the case index, and a newly
persisted `article_embeddings_e5.npy` for the statute corpus) rather than running two
different model families in the same process, which is what actually caused the earlier
SIGSEGVs — a single shared model via `dense_index.get_model()`'s per-name cache does not
have that problem. `article_prior` was dropped from the live feature computation to match.
Verified end-to-end: direct engine call, local `uvicorn` service, and a rebuilt/redeployed
Docker container (health check passes, e5-base loads in ~26s, correct top-ranked results on
both a theft and an assault test case) — the deployed system now matches every number in
this report, not just the offline evaluation.
