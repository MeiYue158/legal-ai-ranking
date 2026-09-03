"""
Serve the trained L2 ranker as a FastAPI inference service.

POST /rank {"fact": "<case narrative>"} ->
    top-k relevant statute articles with scores, plus per-request latency
    logging.

Run locally:
    uvicorn serve:app --host 0.0.0.0 --port 8000
Run in Docker:
    docker build -t legal-ranker . && docker run -p 8000:8000 legal-ranker
Then:
    curl -X POST localhost:8000/rank -H 'Content-Type: application/json' \
         -d '{"fact": "被告人酒后持刀将被害人捅伤，经鉴定为重伤二级。"}'

Known, disclosed limitation (not silently hidden): the /rank and startup
handlers are `async def` specifically to force everything onto uvicorn's
single event-loop thread (see RankingEngine's docstring for the two
macOS-specific native-library issues this sidesteps). Neither handler
contains a real `await`, so this service has NO request concurrency --
two simultaneous requests fully serialize. That's an acceptable tradeoff at
demo/portfolio request volume; it is not a production-concurrency claim.
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from rank_core import RankingEngine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("legal-ranker")

app = FastAPI(title="Legal Statute Ranking Service")
engine = RankingEngine()

MAX_FACT_LEN = 5000  # generous vs. the ~485-char average training fact length


class RankRequest(BaseModel):
    fact: str
    top_k: int = 10

    @field_validator("fact")
    @classmethod
    def fact_not_blank(cls, v):
        if not v or not v.strip():
            raise ValueError("fact must be a non-empty string")
        if len(v) > MAX_FACT_LEN:
            raise ValueError(f"fact exceeds max length of {MAX_FACT_LEN} characters")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v):
        if not (1 <= v <= 50):
            raise ValueError("top_k must be between 1 and 50")
        return v


@app.on_event("startup")
async def load_everything():
    t0 = time.time()
    try:
        engine.load()
    except Exception:
        log.exception("Fatal error loading ranking engine at startup")
        raise
    log.info(f"Loaded index + model in {time.time()-t0:.1f}s")


@app.post("/rank")
async def rank(req: RankRequest):
    t0 = time.time()
    try:
        results, meta = engine.rank(req.fact, req.top_k)
    except Exception:
        log.exception(f"Inference failed for request (fact len={len(req.fact)})")
        raise HTTPException(status_code=500, detail="Inference failed. See server logs.")
    elapsed_ms = (time.time() - t0) * 1000
    log.info(f"/rank latency={elapsed_ms:.1f}ms candidates={meta['n_candidates']}")
    return {"results": results, "latency_ms": elapsed_ms}


@app.get("/health")
def health():
    ready = hasattr(engine, "model")
    return {"status": "ok" if ready else "loading"}
