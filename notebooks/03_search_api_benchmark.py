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
import math
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
#
# `127.0.0.1`, KHÔNG phải `localhost`: trên Windows, "localhost" resolve ra ::1
# trước, uvicorn chỉ bind IPv4, nên mỗi request trả giá một lần IPv6-timeout rồi
# mới fallback. Đo được: wall-clock median 2827 ms qua "localhost" vs 591 ms qua
# "127.0.0.1" — cùng server, cùng query, chênh nhau chỉ vì cái tên host.
URL = "http://127.0.0.1:8000"

# Một `httpx.Client` dùng lại cho toàn bộ benchmark, thay vì `httpx.get()` mỗi
# lần. `httpx.get()` là hàm tiện lợi: nó dựng client mới → bắt tay TCP mới →
# đóng → cho MỖI request. Với keyword mode, server-side là 3 ms còn wall-clock
# lên 591 ms: 99% thời gian là dựng kết nối, không phải search. Connection pool
# đưa wall-clock keyword về 5 ms. Đây là lý do P99(wall) ở bảng dưới bám sát
# P99(server) chứ không còn lệch 100×.
client = httpx.Client(base_url=URL, timeout=60.0)

DEADLINE_S = 900
t_start = time.perf_counter()
while (waited := time.perf_counter() - t_start) < DEADLINE_S:
    try:
        r = client.get("/healthz", timeout=2.0)
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

print(client.get("/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = client.get("/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. TODO — Latency benchmark (50 golden queries × 10 reps × 3 modes)
#
# Dùng 50 golden queries × 10 reps = 500 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# **Vì sao 10 reps chứ không phải 2?** Với n = 100, "P99" là phần tử thứ 99/100
# — tức là **đúng bằng max**. Một lần OS scheduler hiccup, một lần GC, một lần
# Windows Defender quét file là đủ định nghĩa toàn bộ con số. Đo lại 5 lần trên
# cùng máy này ở n = 100 cho hybrid P99 = 26 / 34 / 75 / 100 / 28 ms: cùng một
# hệ thống, kết luận PASS/FAIL đổi theo lượt chạy. Ở n = 500, P99 là phần tử
# 495 nên 5 điểm chậm nhất không còn tự mình quyết định kết quả — cùng máy cho
# 34,4 / 26,3 / 28,2 ms. Ngưỡng chỉ có nghĩa khi phép đo lặp lại được; n = 100
# không đủ mẫu cho một tail metric.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. `min(int(n*p), n-1)` (cách viết cũ) trả về
    index 99 khi n=100, p=0.99 — tức là max, không phải P99. Định nghĩa
    nearest-rank là ceil(p*n) trên thang 1-based → index ceil(p*n)-1."""
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p * n))
    return ordered[min(rank, n) - 1]


def benchmark_mode(mode: str, reps: int = 10) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = client.get("/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "n":          float(len(server_latencies)),
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "max_server": max(server_latencies),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


# Warm-up: rubric đo P99 "after warm-up", nên 10 query đầu KHÔNG được vào phép
# đo. Lần gọi đầu tiên tới mỗi mode phải trả giá lazy-init (ONNX session cho
# semantic/hybrid, cache BM25 cho keyword) — tính nó vào P99 là đo cold start
# chứ không phải đo steady state.
for warm_mode in ("keyword", "semantic", "hybrid"):
    for q in golden[:10]:
        client.get("/search", params={"q": q["query"], "mode": warm_mode})
print("warm-up: 30 query (10 × 3 mode) — không tính vào bảng dưới\n")

print(f"  {'mode':10} {'n':>4}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'max':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10} {res['n']:>4.0f}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['max_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
n = int(results["hybrid"]["n"])
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms  (n={n}, sau warm-up)")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Ba nguyên nhân đã gặp thật trong lab này, theo thứ tự tác động:")
    print("   1. EMBEDDING_BACKEND=fastembed-q (int8) trên CPU không có AVX-VNNI → ~140ms/query")
    print("   2. INFERENCE_THREADS chưa =1 → mỗi anyio worker thread mới trả giá warm-up ONNX")
    print("   3. n quá nhỏ → 'P99' thực chất là max; tăng reps trước khi kết luận")

# %% [markdown]
# ### Cái gì thật sự chi phối P99 — ba thứ, và chỉ một trong đó là model
#
# Lần chạy đầu của notebook này FAIL ngưỡng: hybrid P99 = **2935,8 ms**, gấp 59×
# ngưỡng 50 ms. Không có dòng nào của `Searcher` chậm. Cả ba nguyên nhân đều nằm
# **ngoài** thuật toán retrieval — và mỗi cái đều đo được riêng:
#
# | # | Nguyên nhân | Bằng chứng | Sửa |
# |---|---|---|---|
# | 1 | `httpx.get()` dựng kết nối TCP mới **mỗi request** | keyword: server 3 ms, wall 591 ms | dùng lại một `httpx.Client` |
# | 2 | `localhost` resolve ra `::1` trước, uvicorn chỉ bind IPv4 | wall median 2827 ms → 591 ms khi đổi sang `127.0.0.1` | `127.0.0.1` |
# | 3 | 40 anyio worker thread → mỗi thread mới trả giá warm-up ONNX | hybrid P99 48,9 ms → 26,3 ms khi giới hạn 1 thread | `INFERENCE_THREADS=1` |
#
# Hai nguyên nhân đầu là lỗi **phép đo** (client), cái thứ ba là lỗi **cấu hình
# server**. Không cái nào sửa được bằng cách "tối ưu search" — và đó chính là
# bài học: một con số P99 tệ chưa nói gì về nơi thời gian thật sự đi.
#
# Sau khi sửa cả ba, phần còn lại của P99 **mới** là chi phí thật của retrieval:
# BM25 trên 1000 doc ~3 ms, embed một câu query ~10–16 ms, ANN trên 1000 vector
# gần như bằng 0. Muốn giảm tiếp thì phải nhắm vào bước embed.
#
# #### Và đây là cái bẫy int8
#
# `fastembed` mặc định tải bản ONNX **int8 đã lượng tử hoá** cho
# `BAAI/bge-small-en-v1.5`. Trên CPU có AVX-VNNI, kernel `MatMulInteger` chạy
# nhanh. Trên CPU **không** có (Intel Kaby Lake / i5-8350U của lab này), nó rơi
# xuống đường chậm:
#
# | ONNX build | model | 1 query (10 token) |
# |---|---|---|
# | int8 quantized | `BAAI/bge-small-en-v1.5` | **~140 ms** |
# | fp32 | `BAAI/bge-small-en` | **~10 ms** |
#
# Cùng họ model, cùng 384 chiều, chênh **~14×** — chỉ khác một chi tiết build.
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
sem_p50 = results["semantic"]["p50_server"]
print(f"BM25-only  P50          : {kw_p50:6.1f}ms")
print(f"Semantic   P50          : {sem_p50:6.1f}ms   (embed + ANN)")
print(f"Hybrid     P50          : {hyb_p50:6.1f}ms   (cả hai + RRF)")
print(f"→ phần embed + ANN       : {hyb_p50 - kw_p50:6.1f}ms  "
      f"({(hyb_p50 - kw_p50) / hyb_p50:.0%} của hybrid P50)")
print(f"→ chi phí RRF + overhead : {hyb_p50 - sem_p50 - kw_p50:6.1f}ms  "
      "(hybrid trừ cả hai retriever — phần fusion gần như miễn phí)")

# %% [markdown]
# ## 5. Cleanup — stop the API server

# %%
client.close()
proc.terminate()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid (n=500/mode).
# 3. Output cell 4: hybrid P99 < 50ms PASS.
# 4. Output cell 5: tách BM25 vs embed vs RRF trong hybrid P50.
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
