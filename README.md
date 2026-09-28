# visual-search-server

[![CI](https://github.com/alichr/visual-search-server/actions/workflows/ci.yml/badge.svg)](https://github.com/alichr/visual-search-server/actions/workflows/ci.yml)

CLIP-based visual search inference server (FastAPI + ONNX Runtime).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check .
```

## Layout

```
app/            model.py (ONNX sessions, preprocessing, embeddings)
                batcher.py (async micro-batching queue)
                main.py (FastAPI app and routes)
tests/          API tests
deploy/         prometheus.yml, k8s.yaml
export.py       export weights to ONNX
bench.py        benchmark
```

Model weights (`models/`), datasets (`data/`) and `*.onnx` files are never committed.

## CI

GitHub Actions (`.github/workflows/ci.yml`) is the pipeline that actually runs, on every push and pull
request: **lint** (ruff check + format), **test** (export the ONNX models, then pytest against INT8) and
**build** (docker build, wait for `/readyz`, one end-to-end `/classify` call). Tags `v*` also push the
image to GitHub Container Registry. `.gitlab-ci.yml` mirrors the same three stages for GitLab CI.

## Kubernetes

`deploy/k8s.yaml` defines a Deployment (2 replicas) and a ClusterIP Service. It was tested on
[kind](https://kind.sigs.k8s.io/):

```bash
kind create cluster --name clip
kind load docker-image clip-serve:latest --name clip
kubectl apply -f deploy/k8s.yaml
kubectl rollout status deployment/clip-serve
kubectl port-forward svc/clip-serve 8000:80   # then open http://localhost:8000/docs
```

- **Probes:** liveness on `/healthz`, readiness on `/readyz`, so a pod only gets traffic after the
  models are loaded and warmed up.
- **Resources:** 1–2 CPUs, with `ORT_THREADS=2` matching the CPU limit, and 512 Mi–1 Gi of memory
  (measured: ~310 MiB idle, ~450 MiB under load).
- **Hardening:** the container runs as non-root with a read-only root filesystem and all
  capabilities dropped. A writable `/tmp` volume holds uploaded files.
- **Rolling updates:** `maxUnavailable: 0` plus a 5 s `preStop` pause. Across two rolling restarts
  under load, 1 of ~1,750 requests failed. Without the pause, 5 of ~740 failed.
- **Metrics:** Prometheus scrape annotations are set on the pod template.

**Limitation:** the image index is held in memory by each replica. An image sent to `/index` is
stored on one pod only, so through the Service, `index_size` came back as 2 or 0 depending on
which pod answered, and a pod restart loses its index. A production deployment would store the
embeddings in a shared vector store (pgvector, Qdrant, Milvus) and keep the pods stateless.

## Benchmark

Run with `python bench.py all`. Each configuration runs in a fresh container limited to
`--cpus=4` with `ORT_THREADS=4`, under Docker Desktop (Linux aarch64 VM) on an Apple M4 laptop.
Accuracy is zero-shot top-1 on 500 Imagenette validation images (50 per class), using the prompt
"a photo of a {class}". Load is 256 `/classify` requests (one real photo, 10 labels) per
concurrency level, sent with `httpx.AsyncClient`. Raw numbers: `docs/bench_results.json`.

| Config | Model size | Imagenette top-1 | c=1 p50 / p95 (ms) | c=1 req/s | c=8 p50 / p95 (ms) | c=8 req/s | c=32 p50 / p95 (ms) | c=32 req/s | c=32 mean batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32, no batching | 606 MB | 98.4% | 31 / 34 | 31.7 | 232 / 242 | 34.3 | 935 / 954 | 34.0 | 1.0 |
| FP32, batching | 606 MB | 98.4% | 61 / 66 | 16.3 | 210 / 219 | 37.8 | 808 / 952 | 38.2 | 15.1 |
| INT8, no batching | 191 MB | 98.4% | 14 / 17 | 66.8 | 99 / 115 | 78.5 | 401 / 421 | 78.8 | 1.0 |
| INT8, batching | 191 MB | 98.4% | 45 / 48 | 22.4 | 88 / 101 | 89.1 | 303 / 336 | **101.9** | 15.1 |

**What the numbers say**

- **INT8 is the biggest win.** The model is 3.2× smaller and 2.1–2.7× faster, with the same
  top-1 accuracy (98.4% in the container; 98.2% vs 98.4% on macOS). The speed-up depends on the
  platform. On macOS the same INT8 files were barely faster than FP32. Getting INT8 accurate
  everywhere took two fixes: UInt8 weights (Int8 weights saturate on x86 AVX2 CPUs without
  VNNI), and keeping the text encoder's `fc2` layers in FP32.
- **Batching raises throughput under load**, by +29% for INT8 and +12% for FP32 at c=32, with
  lower p50 latency. The gain is modest on CPU because a batch of 16 costs almost 16× a single
  image, so what batching saves is per-call overhead. On a GPU the gain would be much larger.
- **Batching costs latency at low load.** At c=1 it adds about 30 ms, more than
  `MAX_WAIT_MS=10`, because a `/classify` request goes through two batchers in turn (image, then
  text), and each waits up to 10 ms for other requests to join. The fix would be to skip the
  wait when nothing else is queued.
- **Sanity check:** at c=32 without batching, p50 ≈ concurrency ÷ throughput (32 / 78.8 ≈
  406 ms, measured 401 ms). The latency is queueing, not slow inference.

**A bottleneck the benchmark found.** The first run gave only 5.1 req/s for INT8 with batching at
c=32, and throughput *fell* as concurrency rose. Profiling showed every `/classify` re-embedding
its 10 labels: 113 ms of text-encoder work against 22 ms for the image. Those text calls also ran
in parallel threads, oversubscribing the 4 CPUs. Two changes fixed it: caching text embeddings,
and giving text its own batcher so only one inference runs at a time. Throughput rose 20×
(5.1 → 101.9 req/s). The benchmark reuses the same 10 labels, so it measures the cached path.
Requests with never-seen label sets would still pay the text cost.

**Batch size in Prometheus during a c=32 run.** The mean batch size sits at the `MAX_BATCH=16`
cap:

![Prometheus batch size during a concurrency-32 run](docs/prometheus_batch_size.png)
