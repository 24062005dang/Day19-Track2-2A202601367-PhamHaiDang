# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

# %% [markdown]
# ## 1. Khởi động API server (background)
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này khởi động uvicorn ở background subprocess và đợi `/healthz` trả ready.

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
# `sys.executable -m uvicorn` thay vì bare "uvicorn": trên Windows, script
# shim nằm ở .venv/Scripts (không phải .venv/bin) nên bare "uvicorn" có thể
# không có trên PATH khi chạy headless. Gọi qua interpreter thì luôn đúng venv.
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000",
     "--log-level", "warning"],
    cwd=str(ROOT),
)

# Đợi server up + warm (Searcher.from_corpus loads embeddings + indexes 1000 docs)
# Ngân sách 15 phút, không phải 60 s: startup phải embed cả 1000 doc, và trên CPU
# chậm (không AVX-512 / bị throttle) riêng bước đó đã mất vài phút. 60 s là con số
# đúng cho máy nhanh và fail giả trên máy chậm — thứ đáng chờ thì chờ cho đủ.
URL = "http://localhost:8000"
DEADLINE_S = 900
t_start = time.perf_counter()
while (waited := time.perf_counter() - t_start) < DEADLINE_S:
    try:
        r = httpx.get(f"{URL}/healthz", timeout=2.0)
        if r.status_code == 200 and r.json().get("ready"):
            print(f"server ready sau {waited:.0f}s")
            break
    except httpx.HTTPError:
        pass
    if proc.poll() is not None:                     # server chết -> đừng chờ tiếp
        raise RuntimeError(f"uvicorn exited early with code {proc.returncode}")
    if int(waited) % 30 == 0 and waited > 1:
        print(f"  … đang đợi server warm ({waited:.0f}s / {DEADLINE_S}s)")
    time.sleep(2)
else:
    proc.terminate()
    raise RuntimeError(f"API didn't become ready within {DEADLINE_S}s")

print(httpx.get(f"{URL}/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = httpx.get(f"{URL}/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. TODO — Latency benchmark (100 queries × 3 modes)
#
# Dùng 50 golden queries × 2 reps = 100 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


def benchmark_mode(mode: str, reps: int = 2) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


# Warm-up: rubric đo P99 "after warm-up", nên 10 query đầu KHÔNG được vào phép
# đo. Lần gọi đầu tiên tới mỗi mode phải trả giá lazy-init (ONNX session cho
# semantic/hybrid, cache BM25 cho keyword) — tính nó vào P99 là đo cold start
# chứ không phải đo steady state.
for warm_mode in ("keyword", "semantic", "hybrid"):
    for q in golden[:10]:
        httpx.get(f"{URL}/search", params={"q": q["query"], "mode": warm_mode})
print("warm-up: 30 query (10 × 3 mode) — không tính vào bảng dưới\n")

print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive")
    print("  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth")

# %% [markdown]
# ### Cái gì thật sự chi phối P99 — và một cái bẫy int8
#
# Chia hybrid ra hai phần đo được: BM25 trên 1000 doc là ~3 ms, còn embed **một**
# câu query là phần còn lại. Nói cách khác P99 của hybrid gần như **bằng** chi phí
# embed một câu — không phải chi phí ANN search (Qdrant trả lời trong micro giây ở
# quy mô 1000 vector). Muốn giảm tail latency thì phải nhắm vào bước embed.
#
# Và đây là chỗ có bẫy thật. `fastembed` mặc định tải bản ONNX **int8 đã lượng tử
# hoá** cho `BAAI/bge-small-en-v1.5`. Trên CPU có AVX-VNNI, kernel `MatMulInteger`
# chạy nhanh. Trên CPU **không** có (ví dụ Intel Kaby Lake / i5-8350U của lab này),
# nó rơi xuống đường chậm:
#
# | ONNX build | model | 1 query (10 token) |
# |---|---|---|
# | int8 quantized | `BAAI/bge-small-en-v1.5` | **~140 ms** |
# | fp32 | `BAAI/bge-small-en` | **~12 ms** |
#
# Cùng họ model, cùng 384 chiều, chênh **~10×** — chỉ khác một chi tiết build.
# Lab dùng bản fp32 (`EMBEDDING_BACKEND=fastembed`); bản int8 vẫn dùng được qua
# `EMBEDDING_BACKEND=fastembed-q` nếu bạn muốn tự đo lại trên CPU của mình.
#
# > Bài học vận hành: "lượng tử hoá thì nhanh hơn" **chỉ đúng khi CPU có
# > instruction tương ứng**. Không benchmark trên đúng máy đích thì một quyết
# > định nghe rất hợp lý lại làm hỏng SLA gấp 10 lần.

# %%
# Tách chi phí: BM25 thuần vs hybrid, để thấy phần embed chiếm bao nhiêu.
kw_p50 = results["keyword"]["p50_server"]
hyb_p50 = results["hybrid"]["p50_server"]
print(f"BM25-only P50          : {kw_p50:6.1f}ms")
print(f"Hybrid   P50           : {hyb_p50:6.1f}ms")
print(f"→ phần embed + ANN      : {hyb_p50 - kw_p50:6.1f}ms  "
      f"({(hyb_p50 - kw_p50) / hyb_p50:.0%} của hybrid P50)")

# %% [markdown]
# ## 5. Cleanup — stop the API server

# %%
proc.terminate()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid.
# 3. Output cell 4: hybrid P99 < 50ms PASS.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.
