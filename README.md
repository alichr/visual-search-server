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
