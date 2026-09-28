"""FastAPI app and routes."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import model
from app.batcher import Batcher

image_batcher = Batcher(model.embed_images, "image")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.warmup()
    task = asyncio.create_task(image_batcher.run())  # background batching loop
    yield
    task.cancel()


app = FastAPI(title="CLIP-Serve", lifespan=lifespan)
