"""Smoke tests: the real app (lifespan, batcher, INT8 ONNX models) through FastAPI's TestClient.
Images are generated with Pillow, so no test assets are committed."""

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app  # PRECISION defaults to int8: the fast models, as CI uses


def png(color: str) -> bytes:
    Image.new("RGB", (64, 64), color).save(buf := io.BytesIO(), "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the lifespan: batcher task + background warm-up
        any(c.get("/readyz").status_code == 200 or time.sleep(0.05) for _ in range(600))  # <=30 s
        yield c


def test_ready_after_startup(client):
    assert client.get("/readyz").status_code == 200


def test_classify_solid_red(client):
    r = client.post("/classify", data={"labels": "red, blue"}, files={"file": ("r", png("red"))})
    assert r.status_code == 200 and r.json()["red"] > r.json()["blue"]


def test_index_two_images_then_search(client):
    files = [("files", (f"{c}.png", png(c))) for c in ("green", "yellow")]
    assert client.post("/index", files=files).json()["index_size"] == 2
    hits = client.get("/search", params={"q": "a green square"}).json()  # both, green first
    assert [h["filename"] for h in hits] == ["green.png", "yellow.png"]
