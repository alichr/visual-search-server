"""FastAPI app: index images, search them by text, zero-shot classify, health and metrics."""

import asyncio
import io
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app import model
from app.batcher import Batcher

image_batcher = Batcher(model.embed_images, "image")
state = {"ready": False, "emb": np.zeros((0, 512), np.float32), "names": []}
index_lock = asyncio.Lock()


class Indexed(BaseModel):
    ids: list[int]
    index_size: int


class Hit(BaseModel):
    id: int
    filename: str
    score: float


async def embed(f: UploadFile) -> np.ndarray:
    data = await f.read()
    try:  # decode + resize off the event loop; PIL's pixel check is the real validation
        x = await asyncio.to_thread(lambda: model.preprocess(Image.open(io.BytesIO(data))))
    except (OSError, Image.DecompressionBombError) as e:
        raise HTTPException(415, f"{f.filename!r} is not a readable image") from e
    return await image_batcher.submit(x)


async def warmup() -> None:
    await asyncio.to_thread(model.warmup)
    state["ready"] = True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    tasks = [asyncio.create_task(image_batcher.run()), asyncio.create_task(warmup())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="CLIP-Serve", description=__doc__, lifespan=lifespan)


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
    scores = state["emb"] @ (await asyncio.to_thread(model.embed_text, [q]))[0]
    return [Hit(id=i, filename=state["names"][i], score=scores[i]) for i in np.argsort(-scores)[:k]]


@app.post("/classify", response_model=dict[str, float])
async def classify(file: Annotated[UploadFile, File()], labels: Annotated[str, Form()]):
    """Zero-shot classify an image against comma-separated labels ("a photo of a {label}")."""
    names = [s.strip() for s in labels.split(",") if s.strip()]
    if not names:
        raise HTTPException(422, "labels must contain at least one non-empty label")
    img = await embed(file)
    txt = await asyncio.to_thread(model.embed_text, [f"a photo of a {n}" for n in names])
    p = np.exp(100 * (txt @ img - (txt @ img).max()))  # CLIP logit scale 100, stable softmax
    return dict(sorted(zip(names, (p / p.sum()).tolist(), strict=True), key=lambda kv: -kv[1]))


@app.get("/healthz")
def healthz():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz(response: Response):
    """Readiness: models loaded and warm-up done (503 until then)."""
    response.status_code = 200 if state["ready"] else 503
    return {"ready": state["ready"]}


@app.get("/metrics")
def metrics():
    """Prometheus metrics in text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
