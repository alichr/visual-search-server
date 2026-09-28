# visual-search-server

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
