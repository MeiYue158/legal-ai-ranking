"""
Embed the FULL 30,214-case clean training set with e5-base, checkpointed
(resumable -- background jobs here have been getting killed around the
60-90min mark, likely an external runtime cap, not a code bug). Reuses the
valid+test e5 embeddings already computed and checkpointed by
dense_strong.py's Step C instead of recomputing them.

Output: a permanent, reusable e5 embedding cache for the full train set,
independent of the ad-hoc 8K-subsample checkpoint used for the controlled
check.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from train_ranker import FEATURE_COLS  # import order: lightgbm before torch

import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

PROCESSED = Path(__file__).parent.parent / "data/processed"
E5_MODEL = "intfloat/multilingual-e5-base"
CKPT = PROCESSED / "emb_cache" / "train_full_e5_ckpt.npz"


def main():
    train_df = pd.read_parquet(PROCESSED / "train_clean.parquet")
    texts = train_df["fact"].tolist()
    case_ids = train_df["case_id"].tolist()
    n = len(texts)
    print(f"Embedding {n} full clean-train facts with e5-base (checkpointed)...")

    model = SentenceTransformer(E5_MODEL, device="cpu")

    if CKPT.exists():
        data = np.load(CKPT)
        done = int(data["done"])
        embs = data["embs"]
        print(f"resuming from checkpoint: {done}/{n}")
    else:
        done = 0
        embs = np.zeros((n, model.get_sentence_embedding_dimension()), dtype="float32")

    chunk = 640
    while done < n:
        end = min(done + chunk, n)
        batch = ["query: " + t for t in texts[done:end]]
        embs[done:end] = model.encode(batch, normalize_embeddings=True, batch_size=64)
        done = end
        np.savez(CKPT, embs=embs, done=done)
        print(f"embedded {done}/{n}", flush=True)

    out_path = PROCESSED / "emb_cache" / "train_full_e5_embeddings.npy"
    np.save(out_path, embs)
    ids_path = PROCESSED / "emb_cache" / "train_full_e5_ids.json"
    json.dump(case_ids, open(ids_path, "w"))
    print(f"Wrote {out_path} and {ids_path}")


if __name__ == "__main__":
    main()
