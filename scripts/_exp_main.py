"""Throwaway experiment: does pinning the anyio threadpool to 1 worker remove
the ONNX-per-thread-warmup tail? Compare against app.main (40-thread default)."""
from __future__ import annotations

import anyio.to_thread
from app.main import app


@app.on_event("startup")
async def _pin_threadpool() -> None:
    anyio.to_thread.current_default_thread_limiter().total_tokens = 1
