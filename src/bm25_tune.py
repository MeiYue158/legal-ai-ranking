"""
Phase 2: establish a stronger, fairly-tuned BM25 baseline on the CLEAN
(decontaminated) splits.

All tuning decisions (tokenization variant, k1, b) are made on train+valid
only. The selected configuration is evaluated on the clean test set exactly
ONCE, at the end, for the reported baseline numbers -- test is never looked
at during selection.
"""
import json
import re

import jieba
import pandas as pd
from rank_bm25 import BM25Okapi

from retrieval import load_article_corpus
from metrics import evaluate_ranklists

# a modest, non-exhaustive Chinese function-word list -- not over-engineered,
# just the highest-frequency structural words that carry ~no retrieval signal
STOPWORDS = set("的了是在和与及或也都就而但并又之其该此等对被将从对于以及因为所以如果虽然".split())
PUNCT_RE = re.compile(r"[，。、；：？！""''（）【】《》—…\s]+")


def tokenize_variant(text, strip_punct=True, remove_stopwords=False):
    toks = list(jieba.cut(text))
    if strip_punct:
        toks = [t for t in toks if not PUNCT_RE.fullmatch(t) and t.strip()]
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def evaluate_bm25(corpus, queries_df, k1, b, strip_punct, remove_stopwords, k=10):
    article_ids = list(corpus.keys())
    tokenized_corpus = [tokenize_variant(corpus[a], strip_punct, remove_stopwords) for a in article_ids]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    rank_lists, relevant_sets = [], []
    for _, row in queries_df.iterrows():
        q_tokens = tokenize_variant(row["fact"], strip_punct, remove_stopwords)
        scores = bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:50]
        rank_lists.append([article_ids[i] for i in order])
        relevant_sets.append(set(json.loads(row["relevant_articles"])))
    return evaluate_ranklists(rank_lists, relevant_sets, ks=(1, 3, 5, 10))


def main():
    corpus = load_article_corpus()
    valid_df = pd.read_parquet("../data/processed/valid_clean.parquet")
    test_df = pd.read_parquet("../data/processed/test_clean.parquet")

    print(f"Tuning on clean valid set (n={len(valid_df)}), corpus size={len(corpus)}")

    # Step 1: tokenization variant, fixed k1=1.5/b=0.75 (library defaults) while comparing
    print("\n--- Step 1: tokenization variant (k1=1.5, b=0.75 fixed) ---")
    variants = [
        ("raw (original pipeline)", False, False),
        ("strip punctuation", True, False),
        ("strip punctuation + stopwords", True, True),
    ]
    variant_results = {}
    for name, strip_punct, remove_stop in variants:
        m = evaluate_bm25(corpus, valid_df, 1.5, 0.75, strip_punct, remove_stop)
        variant_results[name] = m
        print(f"  {name}: NDCG@10={m['ndcg@10']:.4f} Recall@10={m['recall@10']:.4f}")
    best_variant = max(variant_results, key=lambda k: variant_results[k]["ndcg@10"])
    best_strip, best_stop = {v[0]: (v[1], v[2]) for v in variants}[best_variant]
    print(f"  Selected tokenization: {best_variant}")

    # Step 2: grid over k1, b with the selected tokenization, validation only
    print(f"\n--- Step 2: k1/b grid (tokenization={best_variant}) ---")
    grid_results = []
    for k1 in [1.0, 1.5, 2.0]:
        for b in [0.5, 0.75, 0.9]:
            m = evaluate_bm25(corpus, valid_df, k1, b, best_strip, best_stop)
            grid_results.append({"k1": k1, "b": b, **m})
            print(f"  k1={k1} b={b}: NDCG@10={m['ndcg@10']:.4f} Recall@10={m['recall@10']:.4f}")

    best = max(grid_results, key=lambda r: r["ndcg@10"])
    print(f"\nSelected config (by validation NDCG@10): k1={best['k1']} b={best['b']}")

    # Step 3: evaluate selected config on clean TEST set ONCE
    print(f"\n--- Step 3: final evaluation on clean TEST set (n={len(test_df)}), ONE TIME ---")
    final = evaluate_bm25(corpus, test_df, best["k1"], best["b"], best_strip, best_stop)
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    out = {
        "tokenization_variant_results": {k: v for k, v in variant_results.items()},
        "selected_tokenization": best_variant,
        "grid_results": grid_results,
        "selected_k1": best["k1"],
        "selected_b": best["b"],
        "final_test_metrics": final,
    }
    with open("../reports/bm25_tuned.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nWrote ../reports/bm25_tuned.json")


if __name__ == "__main__":
    main()
