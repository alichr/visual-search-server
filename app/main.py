"""FastAPI app: index images, search them by text, zero-shot classify, health and metrics."""

import asyncio
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from pydantic import BaseModel

from app import model
from app.batcher import Batcher, jlog

image_batcher = Batcher(model.embed_images, "image")
text_batcher = Batcher(lambda texts: model.embed_text(texts.tolist()), "text")  # one run at a time
state = {"emb": np.zeros((0, 512), np.float32), "names": []}
index_lock = asyncio.Lock()
logging.basicConfig(level=logging.INFO, format="%(message)s")
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])
ERRORS = Counter("errors", "Failed requests", ["endpoint", "type"])  # exported as errors_total
Gauge("index_size", "Images in the index").set_function(lambda: len(state["names"]))
Info("model", "Served model").info({"name": "clip-vit-base-patch32", "precision": model.PRECISION})


class Indexed(BaseModel):
    ids: list[int]
    index_size: int


class Hit(BaseModel):
    id: int
    filename: str
    score: float


async def embed(f: UploadFile) -> np.ndarray:
    try:  # open() reads only the header; decode + resize run off the event loop
        x = await asyncio.to_thread(model.preprocess, Image.open(io.BytesIO(await f.read())))
    except (OSError, Image.DecompressionBombError) as e:
        raise HTTPException(415, f"{f.filename!r} is not a readable image") from e
    return await image_batcher.submit(x)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["warmup"] = asyncio.create_task(asyncio.to_thread(model.warmup))  # /readyz watches it
    tasks = [asyncio.create_task(b.run()) for b in (image_batcher, text_batcher)]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="CLIP-Serve", description=__doc__, lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid, t0 = request.headers.get("x-request-id") or uuid.uuid4().hex, time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:  # unhandled -> 500 upstream; count it, then re-raise
        ERRORS.labels(request.scope["route"].path, type(e).__name__).inc()
        raise
    endpoint = getattr(request.scope.get("route"), "path", "unmatched")  # bounded label values
    LATENCY.labels(endpoint).observe(dt := time.perf_counter() - t0)
    if response.status_code >= 400:
        ERRORS.labels(endpoint, str(response.status_code)).inc()
    jlog(request_id=rid, endpoint=endpoint, status=response.status_code, ms=round(dt * 1e3, 1))
    return response


@app.post("/index", response_model=Indexed)
async def index(files: Annotated[list[UploadFile], File()]):
    """Embed one or more images and add them to the in-memory index."""
    embs = await asyncio.gather(*map(embed, files))  # concurrent -> batched together
    async with index_lock:  # serialise writers so ids and rows stay aligned
        start = len(state["names"])
        state["emb"] = np.vstack([state["emb"], *embs])
        state["names"] = state["names"] + [f.filename for f in files]
    return Indexed(ids=list(range(start, start + len(files))), index_size=len(state["names"]))


@app.get("/search", response_model=list[Hit])
async def search(q: Annotated[str, Query(min_length=1)], k: Annotated[int, Query(ge=1, le=50)] = 5):
    """Rank indexed images by cosine similarity to a free-text query."""
    if not state["names"]:
        raise HTTPException(409, "Index is empty: POST images to /index first")
    scores = state["emb"] @ await text_batcher.submit(q)
    return [Hit(id=i, filename=state["names"][i], score=scores[i]) for i in np.argsort(-scores)[:k]]


@app.post("/classify", response_model=dict[str, float])
async def classify(file: Annotated[UploadFile, File()], labels: Annotated[str, Form()]):
    """Zero-shot classify an image against comma-separated labels ("a photo of a {label}")."""
    names = [s.strip() for s in labels.split(",") if s.strip()]
    if not names:
        raise HTTPException(422, "labels must contain at least one non-empty label")
    img = await embed(file)
    txt = np.stack(await asyncio.gather(*(text_batcher.submit(f"a photo of a {n}") for n in names)))
    p = np.exp(100 * (txt @ img - (txt @ img).max()))  # CLIP logit scale 100, stable softmax
    return dict(sorted(zip(names, (p / p.sum()).tolist(), strict=True), key=lambda kv: -kv[1]))


@app.get("/healthz")
def healthz():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz(response: Response):
    """Readiness: models loaded and warm-up done (503 until then)."""
    ready = state["warmup"].done() and state["warmup"].exception() is None  # failed = never ready
    response.status_code = 200 if ready else 503
    return {"ready": ready}


@app.get("/metrics")
def metrics():
    """Prometheus metrics (a route, not a mount: avoids the /metrics -> /metrics/ redirect)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
