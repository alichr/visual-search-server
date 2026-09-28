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
