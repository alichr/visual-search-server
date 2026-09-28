"""Async micro-batching queue: collect requests for up to MAX_WAIT_MS or MAX_BATCH items, run
one forward pass, and resolve each caller's future with its own row. MAX_BATCH=1 disables it."""

import asyncio
import os

import numpy as np
from prometheus_client import Gauge, Histogram

BATCH_SIZE = Histogram("batch_size", "Items per forward pass", ["batcher"],
                       buckets=[1, 2, 4, 8, 16, 32, 64])
QUEUE_DEPTH = Gauge("queue_depth", "Requests waiting for a batch", ["batcher"])


class Batcher:
    def __init__(self, fn, name: str = "image", max_batch: int | None = None,
                 max_wait_ms: float | None = None):
        self.fn, self.name = fn, name
        self.max_batch = max_batch or int(os.getenv("MAX_BATCH", "16"))
        self.max_wait = (max_wait_ms or float(os.getenv("MAX_WAIT_MS", "10"))) / 1000
        self.q: asyncio.Queue = asyncio.Queue()
        QUEUE_DEPTH.labels(name).set_function(self.q.qsize)  # read live at scrape time

    async def submit(self, x: np.ndarray) -> np.ndarray:
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((x, fut))
        return await fut

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.q.get()]  # block until the first request arrives
            deadline = loop.time() + self.max_wait
            while len(items) < self.max_batch and (t := deadline - loop.time()) > 0:
                try:
                    items.append(await asyncio.wait_for(self.q.get(), t))
                except TimeoutError:
                    break
            xs, futs = zip(*items, strict=True)
            BATCH_SIZE.labels(self.name).observe(len(items))
            try:  # blocking ONNX call runs in a worker thread; the event loop stays free
                out = await loop.run_in_executor(None, self.fn, np.stack(xs))
                for f, row in zip(futs, out, strict=True):
                    if not f.done():  # caller may have disconnected (future cancelled)
                        f.set_result(row)
            except Exception as e:  # one bad batch must not hang its callers or kill the loop
                for f in futs:
                    if not f.done():
                        f.set_exception(e)
