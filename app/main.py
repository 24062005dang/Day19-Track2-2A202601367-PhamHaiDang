"""FastAPI service exposing /search?q=...&mode=keyword|semantic|hybrid.

Run: `make api`  (or `uvicorn app.main:app --reload --port 8000`)
Try: curl 'http://localhost:8000/search?q=cloud+computing&mode=hybrid'

The Searcher is built once at startup (lazy). On first request the embedding
model is loaded — subsequent requests reuse it. P99 should be < 50 ms after
warm-up; that's the rubric threshold.

Tail-latency note (this is why INFERENCE_THREADS exists): `/search` is a sync
`def`, so Starlette runs it in the anyio worker threadpool — 40 threads by
default. Each *new* thread that first touches the ONNX session pays a
per-thread warm-up, and onnxruntime's intra-op pool spin-waits across all
cores. Sequential requests therefore scatter over dozens of threads and each
first touch lands in the tail: measured hybrid P99 48.9 ms vs 26.3 ms with the
pool bounded to one worker, same machine, same 500 requests. Bounding the pool
keeps the CPU-bound handler on one thread, which is also what the workload
actually wants — this API is CPU-bound, so extra threads buy contention, not
throughput. Scale out with uvicorn *workers* (processes), not anyio threads.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anyio.to_thread
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.search import Searcher, SearchHit

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "corpus_vn.jsonl"

# How many anyio worker threads may run `/search` concurrently. 1 = no ONNX
# per-thread warm-up in the tail (see module docstring). Raise it only if you
# have measured that your workload is I/O-bound, which this one is not.
INFERENCE_THREADS = int(os.getenv("INFERENCE_THREADS", "1"))

# Module-level state intentionally — Searcher is heavy to build.
_searcher: Searcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Searcher once at startup. Embedding model + indexing the 1000
    docs takes ~30s on first run. Cached on disk in subsequent runs."""
    global _searcher
    if not CORPUS_PATH.exists():
        raise RuntimeError(
            f"{CORPUS_PATH} missing. Run `make seed` first."
        )
    anyio.to_thread.current_default_thread_limiter().total_tokens = INFERENCE_THREADS
    _searcher = Searcher.from_corpus(CORPUS_PATH)
    # Warm the ONNX session on the same thread that will serve requests, so the
    # first real request is not the one that pays for graph/arena init.
    _searcher.search("warm up the onnx session", mode="hybrid", top_k=1)
    yield
    _searcher = None


app = FastAPI(
    title="Day 19 — Search API",
    description="Vector + BM25 + hybrid (RRF) retrieval. Lab 19, Track 2.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchHitOut(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: Literal["keyword", "semantic", "hybrid"]
    top_k: int
    latency_ms: float = Field(description="Server-side latency excluding network.")
    hits: list[SearchHitOut]


@app.get("/")
def root() -> dict:
    return {
        "name": "Day 19 Search API",
        "endpoints": ["/search", "/healthz", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ready": _searcher is not None, "n_docs": _searcher.size if _searcher else 0}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Query string (VN/EN OK)"),
    mode: Literal["keyword", "semantic", "hybrid"] = Query("hybrid"),
    top_k: int = Query(10, ge=1, le=100),
    rrf_k: int = Query(60, ge=1, le=200, description="Reciprocal Rank Fusion constant (hybrid only)"),
) -> SearchResponse:
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not yet ready")

    t0 = time.perf_counter()
    hits = _searcher.search(q, mode=mode, top_k=top_k, rrf_k=rrf_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=q,
        mode=mode,
        top_k=top_k,
        latency_ms=latency_ms,
        hits=[SearchHitOut(**h.dict()) for h in hits],
    )
