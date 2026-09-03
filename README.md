# Legal Statute & Precedent Retrieval-Ranking System

A two-stage learning-to-rank system that, given a Chinese criminal case's
fact narrative, retrieves and ranks the relevant statute articles out of the
Criminal Law of the PRC. Built on the CAIL2018 dataset. Framed and evaluated
as an information-retrieval / search-relevance problem (query = case facts,
documents = statute articles, graded multi-relevant-document ground truth),
not as a text classifier -- this is deliberately structured to mirror how
production search-ranking systems (candidate generation -> feature
engineering -> learned re-ranking -> offline eval -> deployment) are built.

This project is a rebuild of an earlier prototype (`../Legal_Agent`) that
made a single ungrounded LLM call per document. That prototype's 3-module
plan (Legal Summarization / Appeal Recommendation / Unfair Clause Detection)
is preserved conceptually: the "query similar cases" idea from its Appeal
Recommendation module became the case-precedent kNN feature here (Phase 3),
and its RAG-style output becomes Phase 7 -- an LLM agent grounded in this
ranker's output instead of one raw prompt.

## Data

- **CAIL2018**: 154,592 / 17,131 / 32,508 train/valid/test Chinese criminal
  case narratives, each labeled with charge(s), applicable statute
  article(s), and sentence. Subsampled to 30,000 / 5,000 / 10,000 for
  compute tractability on a laptop (documented, adjustable in
  `src/data_prep.py`).
- **Statute corpus**: full text of the PRC Criminal Law (2024-revised, 451
  articles), parsed from public legal text (`src/statutes.py`). 182/183 of
  the articles CAIL references resolve against this corpus; the 1 miss
  (Article 199) was formally repealed by Amendment VIII (2011) and no
  longer exists in current law -- a documented, expected edge case, not a
  bug.

## Pipeline (`src/`)

| Phase | Script | What it does |
|---|---|---|
| 1a | `statutes.py` | Parse statute corpus from raw HTML into per-article text |
| 1b | `data_prep.py` | CAIL -> clean train/valid/test query sets |
| 2 | `retrieval.py` | L1 candidate generation: BM25 + dense embedding retrieval over statute text |
| 3 | `dense_index.py`, `features.py` | Case-precedent kNN signal (retrieve similar train cases, let their labels vote) + full feature engineering |
| 4 | `train_ranker.py` | L2 re-ranker: LightGBM LambdaRank, vs. a logistic-regression pointwise baseline |
| 5 | `evaluate_full.py`, `metrics.py` | NDCG@k / Recall@k / MRR, architecture comparison with paired bootstrap significance testing |
| 6 | `serve.py`, `rank_core.py` | FastAPI inference service (ranking logic shared via `RankingEngine`) with per-request latency logging |
| 7 | `agent.py` | LLM agent: cites the ranker's actual top-K statutes + top-K precedent cases in a generated appeal recommendation, with a groundedness guardrail on the model's own citations |

## Phase 7: grounded generation, not a raw LLM call

`agent.py` is the original project's "Appeal Recommendation Analysis" module, rebuilt. The
difference from a single ungrounded prompt (what `Legal_Agent/services/llm.py` does): the LLM
is handed the ranker's real top-K statute text and top-K most similar precedent cases (with
their actual outcomes) and instructed to cite only from that set. A regex-based groundedness
check then verifies every article number the model cites in its own output against the
retrieved set and flags (rather than silently drops) anything hallucinated -- same idea as a
PII-detection guardrail, applied to citation grounding instead.

Run: `python3 agent.py "<case fact text>"` (reads `GOOGLE_API_KEY` from `../Legal_Agent/.env`).

## Key finding from L1-only baselines

Pure lexical retrieval (BM25) and pure off-the-shelf dense retrieval both
underperform badly alone, for different reasons -- BM25 misses on
vocabulary mismatch (case narratives use concrete, colloquial language;
statute text is abstract/normative and rarely shares vocabulary with the
facts it governs), and general-purpose multilingual embeddings aren't
domain-adapted to legal text. This directly motivates the L2 model: it
learns to combine BM25 score, dense score, and a case-precedent kNN vote
(which sidesteps vocabulary mismatch entirely by matching case-to-case
instead of case-to-statute) into one ranking function.

See `reports/architecture_comparison.csv` for the full numbers.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/statutes.py
python3 src/data_prep.py
python3 src/features.py     # ~10-15 min on a laptop CPU
python3 src/train_ranker.py
python3 src/evaluate_full.py
```
