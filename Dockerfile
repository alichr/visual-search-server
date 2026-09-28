# syntax=docker/dockerfile:1

# ---- Stage 1: exporter - CPU-only torch, export CLIP to ONNX (FP32 + INT8) ----
FROM python:3.11-slim AS exporter
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /build
COPY requirements.txt requirements-dev.txt ./
# CPU wheel first so the dev requirements don't pull the CUDA build of torch (~2 GB more)
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-dev.txt
COPY export.py .
# INT8_ONLY=1 drops the FP32 models (-606 MB) for a smaller image; PRECISION=fp32 then fails fast
ARG INT8_ONLY=0
RUN python export.py --out /models && if [ "$INT8_ONLY" = 1 ]; then rm /models/*_fp32.onnx; fi

# ---- Stage 2: runtime - no torch, no transformers ----
FROM python:3.11-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models PRECISION=int8 ORT_THREADS=2
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && useradd --create-home --uid 10001 app
COPY --from=exporter /models /models
COPY app/ app/
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"
# One worker: the micro-batcher lives in-process; scale by adding replicas, not workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
