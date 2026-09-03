"""
Actually measure the citation-groundedness rate across a sample of real
test-set cases, instead of asserting it from a single manual run.

Free-tier Gemini quota is 5 requests/minute, so this paces calls and backs
off on 429s rather than hammering the API.
"""
import json
import time

import pandas as pd

from agent import generate_recommendation
from rank_core import RankingEngine

N_SAMPLES = 20
SLEEP_BETWEEN_CALLS = 15  # seconds; keeps us under the 5 req/min free-tier cap


def main():
    test_df = pd.read_parquet("../data/processed/test.parquet")
    sample = test_df.sample(n=N_SAMPLES, random_state=20250902).reset_index(drop=True)

    engine = RankingEngine().load()

    results = []
    for i, row in sample.iterrows():
        fact = row["fact"]
        true_articles = set(json.loads(row["relevant_articles"]))
        print(f"[{i+1}/{N_SAMPLES}] case_id={row['case_id']}", flush=True)

        attempt = 0
        while True:
            try:
                out = generate_recommendation(engine, fact, top_k_statutes=5, top_k_cases=3)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 4:
                    print(f"  GAVE UP after {attempt} attempts: {e}", flush=True)
                    out = None
                    break
                wait = 20 * attempt
                print(f"  error ({e}); retrying in {wait}s", flush=True)
                time.sleep(wait)

        if out is None:
            results.append({"case_id": row["case_id"], "status": "error"})
        else:
            g = out["groundedness"]
            retrieved_ids = {r["article"] for r in out["retrieved_statutes"]}
            results.append({
                "case_id": row["case_id"],
                "status": "ok",
                "true_articles": sorted(true_articles),
                "retrieved_ids": sorted(retrieved_ids),
                "cited_articles": g["cited_articles"],
                "hallucinated_articles": g["hallucinated_articles"],
                "grounded": g["grounded"],
            })
            print(f"  cited={g['cited_articles']} hallucinated={g['hallucinated_articles']}", flush=True)

        time.sleep(SLEEP_BETWEEN_CALLS)

    ok = [r for r in results if r["status"] == "ok"]
    grounded = [r for r in ok if r["grounded"]]
    print("\n" + "=" * 60)
    print(f"Completed: {len(ok)}/{N_SAMPLES} succeeded, {N_SAMPLES - len(ok)} errored out")
    if ok:
        print(f"Grounded (zero hallucinated citations): {len(grounded)}/{len(ok)} "
              f"({len(grounded)/len(ok):.1%})")
        for r in ok:
            if not r["grounded"]:
                print(f"  HALLUCINATION in {r['case_id']}: cited {r['hallucinated_articles']} "
                      f"not in retrieved set {r['retrieved_ids']}")

    with open("../reports/groundedness_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nWrote ../reports/groundedness_eval.json")


if __name__ == "__main__":
    main()
