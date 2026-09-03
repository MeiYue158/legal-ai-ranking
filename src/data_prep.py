"""
Phase 1b: Turn CAIL2018 into a ranking dataset.

Each case's `fact` text is a query; its labeled `relevant_articles` are the
gold-relevant documents out of the fixed statute corpus built by
statutes.py. We keep the dataset's native train/valid/test split (154,592 /
17,131 / 32,508) but SUBSAMPLE for the parts of the pipeline that are
per-query-expensive (dense embedding, cross-encoder scoring) -- this is a
documented, standard scoping decision for compute tractability on a laptop,
not a limitation of the approach. BM25 / LightGBM stages can run on the
full data if desired by raising SAMPLE_* below.
"""
import json
import random
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent.parent / "Legal_Agent" / "exercise_contest"
ARTICLES_JSON = Path(__file__).parent.parent / "data/statutes/articles.json"
OUT_DIR = Path(__file__).parent.parent / "data/processed"

SEED = 20250902
SAMPLE_TRAIN = 30_000
SAMPLE_VALID = 5_000
SAMPLE_TEST = 10_000


def load_corpus_article_numbers():
    articles = json.load(open(ARTICLES_JSON, encoding="utf-8"))
    return {v["article_no"] for v in articles.values()}


def load_split(fname, corpus_articles, sample_n, seed=SEED):
    path = RAW_DIR / fname
    rng = random.Random(seed)
    rows = []
    dropped_no_valid_article = 0
    with open(path, encoding="utf-8") as f:
        # reservoir sample so we don't have to materialize 154K rows to subsample
        reservoir = []
        for i, line in enumerate(f):
            d = json.loads(line)
            arts = [a for a in d["meta"]["relevant_articles"] if a in corpus_articles]
            if not arts:
                dropped_no_valid_article += 1
                continue
            rec = {
                "case_id": f"{fname}:{i}",
                "fact": d["fact"],
                "relevant_articles": arts,
                "accusation": d["meta"]["accusation"],
                "imprisonment": d["meta"]["term_of_imprisonment"]["imprisonment"],
                "life_imprisonment": d["meta"]["term_of_imprisonment"]["life_imprisonment"],
                "death_penalty": d["meta"]["term_of_imprisonment"]["death_penalty"],
            }
            if len(reservoir) < sample_n:
                reservoir.append(rec)
            else:
                j = rng.randint(0, i)
                if j < sample_n:
                    reservoir[j] = rec
    print(f"{fname}: sampled {len(reservoir)} (dropped {dropped_no_valid_article} "
          f"cases whose only labeled article[s] aren't in the parsed corpus, e.g. repealed Art.199)")
    return pd.DataFrame(reservoir)


def main():
    corpus_articles = load_corpus_article_numbers()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = load_split("data_train.json", corpus_articles, SAMPLE_TRAIN)
    df_valid = load_split("data_valid.json", corpus_articles, SAMPLE_VALID)
    df_test = load_split("data_test.json", corpus_articles, SAMPLE_TEST)

    for name, df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        df["relevant_articles"] = df["relevant_articles"].apply(json.dumps)
        df["accusation"] = df["accusation"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        out = OUT_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")

    # avg #relevant articles per case (multi-label!) -- relevant for how we frame NDCG (graded/binary multi-relevant retrieval, not single-answer)
    n_multi = (df_train["relevant_articles"].apply(json.loads).apply(len) > 1).mean()
    print(f"share of train cases with >1 relevant article: {n_multi:.1%}")


if __name__ == "__main__":
    main()
