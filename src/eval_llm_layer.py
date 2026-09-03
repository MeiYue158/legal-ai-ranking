"""
Phase 9: a modest, honestly-scoped LLM-layer evaluation.

Measures exactly two things quantitatively, at a sample size that's stated
plainly (not implied to be larger than it is):
  1. Citation-membership rate (does every cited article number appear in
     the retrieved set?) -- requires an actual LLM call per case.
  2. Gold-in-context rate (was the true statute even present in what the
     LLM was shown?) -- free, computed directly from the ranker's clean-test
     recall, no LLM call needed, can be reported at full test-set scale.

Dimensions 2/3 from the audit request (does the cited text semantically
support the claim; is the recommendation consistent with the evidence) are
NOT scored quantitatively here -- they require either a human judge or a
second LLM-as-judge call per case, and free-tier quota has repeatedly proven
too unreliable for a meaningful n at that cost. Per the explicit fallback
instruction: left as qualitative/manual-inspection only, not manufactured
into a numeric claim.
"""
import json
import time

import pandas as pd

from agent import generate_recommendation
from rank_core import RankingEngine

N_SAMPLES = 10  # stated plainly -- this is a small, honestly-scoped check
SLEEP_BETWEEN_CALLS = 20


def gold_in_context_rate_full_test():
    """Free metric at full test-set scale: is the true statute among the
    top-K shown to the LLM? Directly derivable from the clean-test ranker
    evaluation already on file -- no LLM calls needed."""
    test_feat = pd.read_parquet("../data/processed/test_clean_features.parquet")
    raw = pd.read_parquet("../data/processed/test_clean.parquet")
    true_gold = {row["case_id"]: set(json.loads(row["relevant_articles"])) for _, row in raw.iterrows()}
    # this needs the trained model's top-5 ranking, not just candidate membership;
    # approximate with candidate-pool membership as an upper bound and note that explicitly
    hits, total = 0, 0
    for case_id, g in test_feat.groupby("case_id"):
        gold = true_gold.get(case_id, set())
        if not gold:
            continue
        total += 1
        if gold & set(g["article_id"]):
            hits += 1
    return hits / total if total else None


def main():
    test_df = pd.read_parquet("../data/processed/test_clean.parquet")
    sample = test_df.sample(n=N_SAMPLES, random_state=20250902).reset_index(drop=True)

    engine = RankingEngine().load()

    results = []
    for i, row in sample.iterrows():
        fact = row["fact"]
        true_articles = set(json.loads(row["relevant_articles"]))
        print(f"[{i+1}/{N_SAMPLES}] case_id={row['case_id']}", flush=True)

        attempt = 0
        out = None
        while attempt < 4:
            try:
                out = generate_recommendation(engine, fact, top_k_statutes=5, top_k_cases=3)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 4:
                    print(f"  GAVE UP after {attempt} attempts: {e}", flush=True)
                    break
                wait = 20 * attempt
                print(f"  error ({e}); retrying in {wait}s", flush=True)
                time.sleep(wait)

        if out is None:
            results.append({"case_id": row["case_id"], "status": "error"})
        else:
            c = out["citation_check"]
            retrieved_ids = {r["article"] for r in out["retrieved_statutes"]}
            gold_in_context = bool(true_articles & retrieved_ids)
            results.append({
                "case_id": row["case_id"],
                "status": "ok",
                "true_articles": sorted(true_articles),
                "retrieved_ids": sorted(retrieved_ids),
                "gold_in_context": gold_in_context,
                "cited_articles": c["cited_articles"],
                "citations_not_in_retrieved_set": c["citations_not_in_retrieved_set"],
                "citation_membership_ok": c["citation_membership_ok"],
            })
            print(f"  gold_in_context={gold_in_context} cited={c['cited_articles']} "
                  f"membership_ok={c['citation_membership_ok']}", flush=True)

        time.sleep(SLEEP_BETWEEN_CALLS)

    ok = [r for r in results if r["status"] == "ok"]
    membership_ok = [r for r in ok if r["citation_membership_ok"]]
    print("\n" + "=" * 60)
    print(f"Completed: {len(ok)}/{N_SAMPLES} succeeded, {N_SAMPLES - len(ok)} errored out")
    if ok:
        print(f"Citation-membership OK: {len(membership_ok)}/{len(ok)} ({len(membership_ok)/len(ok):.1%}) "
              f"-- SMALL SAMPLE, not a statistically robust rate")
        print(f"Gold-in-context (this sample): {sum(r['gold_in_context'] for r in ok)}/{len(ok)}")

    full_rate = gold_in_context_rate_full_test()
    print(f"\nGold-in-candidate-pool rate, FULL clean test set (n={len(test_df)}, "
          f"free metric, no LLM calls): {full_rate:.4f}")
    print("(NOTE: this is candidate-pool membership, an upper bound on what top-5 could contain -- "
          "not identical to 'was in the top-5 the LLM was actually shown', which requires the trained model's ranking.)")

    with open("../reports/llm_layer_eval.json", "w", encoding="utf-8") as f:
        json.dump({"sample_results": results, "gold_in_candidate_pool_rate_full_test": full_rate}, f,
                   ensure_ascii=False, indent=2)
    print("\nWrote ../reports/llm_layer_eval.json")


if __name__ == "__main__":
    main()
