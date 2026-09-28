"""Async micro-batching queue: collect requests for up to MAX_WAIT_MS or MAX_BATCH items, run
one forward pass, and resolve each caller's future with its own row. MAX_BATCH=1 disables it."""

import asyncio
import json
import logging
import os

import numpy as np
from prometheus_client import Gauge, Histogram

MAX_BATCH, MAX_WAIT_MS = int(os.getenv("MAX_BATCH", "16")), float(os.getenv("MAX_WAIT_MS", "10"))
BATCH_SIZE = Histogram("batch_size", "Items per batch", ["batcher"], buckets=(1, 2, 4, 8, 16, 32))
QUEUE_DEPTH = Gauge("queue_depth", "Requests waiting for a batch", ["batcher"])


def jlog(**fields) -> None:  # structured JSON log line
    logging.getLogger("clip-serve").info(json.dumps(fields))


class Batcher:
    def __init__(self, fn, name="image", max_batch=MAX_BATCH, max_wait_ms=MAX_WAIT_MS):
        self.fn, self.name, self.max_batch, self.max_wait = fn, name, max_batch, max_wait_ms / 1000
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
            t0 = loop.time()
            try:  # blocking ONNX call runs in a worker thread; the event loop stays free
                out = await loop.run_in_executor(None, self.fn, np.stack(xs))
            except Exception as e:  # one bad batch must not hang its callers or kill the loop
                out = [e] * len(futs)
            jlog(event="batch", name=self.name, size=len(xs), ms=round((loop.time() - t0) * 1e3, 1))
            for f, r in zip(futs, out, strict=True):
                if not f.done():  # caller may have disconnected (future cancelled)
                    f.set_exception(r) if isinstance(r, Exception) else f.set_result(r)
