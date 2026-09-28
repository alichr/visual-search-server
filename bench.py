"""Benchmark CLIP-Serve: zero-shot accuracy and HTTP latency/throughput.

python bench.py accuracy --data data/imagenette2-320/val --n 500   # uses PRECISION from the env
python bench.py load --url http://localhost:8000 --n 256 --concurrency 1 8 32
python bench.py all      # 4 configs in Docker with --cpus=4; prints the README results table
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CLASSES = {  # Imagenette WordNet ids -> plain class names used as zero-shot labels
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
DATA = "data/imagenette2-320/val"


def sample(data: str, n: int) -> list[tuple[Path, int]]:
    """First n // 10 images of each class, in sorted order: deterministic and balanced."""
    per = n // len(CLASSES)
    return [
        (p, i)
        for i, wnid in enumerate(CLASSES)
        for p in sorted(Path(data, wnid).glob("*.JPEG"))[:per]
    ]


def accuracy(data: str, n: int) -> dict:
    """Top-1 zero-shot accuracy via app.model, one image per forward pass (no batch effects)."""
    from PIL import Image

    from app import model

    labels = model.embed_text([f"a photo of a {c}" for c in CLASSES.values()])
    items = sample(data, n)
    t0 = time.perf_counter()
    hits = 0
    for path, y in items:
        emb = model.embed_images(model.preprocess(Image.open(path))[None])[0]
        hits += int(np.argmax(labels @ emb) == y)
    return {
        "precision": model.PRECISION,
        "n": len(items),
        "top1": hits / len(items),
        "seconds": round(time.perf_counter() - t0, 1),
    }


async def _load(url: str, n: int, concurrency: int, image: bytes) -> dict:
    import httpx

    labels = ", ".join(CLASSES.values())
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:

        async def one() -> float:
            t = time.perf_counter()
            r = await client.post(
                "/classify", data={"labels": labels}, files={"file": ("x.jpg", image, "image/jpeg")}
            )
            r.raise_for_status()
            return time.perf_counter() - t

        for _ in range(8):  # warm up connections and the model
            await one()
        before = await _batch_stats(client)
        sem = asyncio.Semaphore(concurrency)

        async def bounded() -> float:
            async with sem:
                return await one()

        t0 = time.perf_counter()
        lat = np.array(await asyncio.gather(*(bounded() for _ in range(n)))) * 1000
        wall = time.perf_counter() - t0
        after = await _batch_stats(client)
    batches, items = after[0] - before[0], after[1] - before[1]
    return {
        "concurrency": concurrency,
        "n": n,
        "p50": np.percentile(lat, 50),
        "p95": np.percentile(lat, 95),
        "p99": np.percentile(lat, 99),
        "rps": n / wall,
        "mean_batch": items / batches if batches else 0.0,
    }


async def _batch_stats(client) -> tuple[float, float]:
    """(forward passes, items) so far, read from the server's batch_size histogram."""
    text = (await client.get("/metrics")).text

    def get(key: str) -> float:
        return float(next(ln.split()[-1] for ln in text.splitlines() if ln.startswith(key)))

    return get('batch_size_count{batcher="image"}'), get('batch_size_sum{batcher="image"}')


def load(url: str, n: int, levels: list[int]) -> list[dict]:
    image = next(iter(Path(DATA, "n02102040").glob("*.JPEG"))).read_bytes()  # one real photo
    return [asyncio.run(_load(url, n, c, image)) for c in levels]


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout


def run_all(n_acc: int, n_load: int, levels: list[int], cpus: str) -> None:
    """Each config in a fresh container with a fixed CPU limit, so numbers are reproducible."""
    import httpx

    root = Path.cwd()
    info = json.loads(docker("info", "--format", "{{json .}}"))
    hw = {
        "host": platform.platform(),
        "host_cpu": platform.processor() or platform.machine(),
        "docker_os": info["OperatingSystem"],
        "docker_arch": info["Architecture"],
        "docker_cpus": info["NCPU"],
        "container_cpus": cpus,
    }
    results = {"hardware": hw, "accuracy": {}, "load": {}}
    for prec in ("fp32", "int8"):
        out = docker(
            "run",
            "--rm",
            f"--cpus={cpus}",
            "-e",
            f"PRECISION={prec}",
            "-e",
            f"ORT_THREADS={cpus}",
            "-v",
            f"{root}/data:/srv/data:ro",
            "-v",
            f"{root}/bench.py:/srv/bench.py:ro",
            "clip-serve:latest",
            "python",
            "bench.py",
            "accuracy",
            "--n",
            str(n_acc),
        )
        results["accuracy"][prec] = json.loads(out.strip().splitlines()[-1])
        print("accuracy", results["accuracy"][prec], flush=True)
        for mb in (1, 16):
            name = f"bench-{prec}-{mb}"
            docker(
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                f"--cpus={cpus}",
                "-p",
                "8000:8000",
                "-e",
                f"PRECISION={prec}",
                "-e",
                f"ORT_THREADS={cpus}",
                "-e",
                f"MAX_BATCH={mb}",
                "clip-serve:latest",
            )
            try:
                for _ in range(120):
                    try:
                        if httpx.get("http://localhost:8000/readyz").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                rows = load("http://localhost:8000", n_load, levels)
                results["load"][f"{prec}-{mb}"] = rows
                for r in rows:
                    print(f"{prec} MAX_BATCH={mb:2d}", {k: round(v, 1) for k, v in r.items()})
            finally:
                docker("stop", name)
    Path("docs/bench_results.json").write_text(json.dumps(results, indent=2, default=float))
    print_table(results)


def print_table(res: dict) -> None:
    size = {
        p: sum(os.path.getsize(f"models/{k}_{p}.onnx") for k in ("image", "text")) / 1e6
        for p in ("fp32", "int8")
    }
    print(f"\nHardware: {res['hardware']}\n")
    levels = [r["concurrency"] for r in res["load"]["fp32-1"]]
    head = "".join(f" | c={c} p50 / p95 / p99 (ms) | c={c} req/s | c={c} batch" for c in levels)
    print(f"| Config | Model size | Imagenette top-1{head} |")
    print("|" + " --- |" * (3 + 3 * len(levels)))
    for key, label in [
        ("fp32-1", "FP32, no batching"),
        ("fp32-16", "FP32, batching"),
        ("int8-1", "INT8, no batching"),
        ("int8-16", "INT8, batching"),
    ]:
        p = key.split("-")[0]
        cells = "".join(
            f" | {r['p50']:.0f} / {r['p95']:.0f} / {r['p99']:.0f} | {r['rps']:.1f}"
            f" | {r['mean_batch']:.1f}"
            for r in res["load"][key]
        )
        print(f"| {label} | {size[p]:.0f} MB | {res['accuracy'][p]['top1']:.1%}{cells} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["accuracy", "load", "all", "table"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=None, help="images (accuracy) or requests (load)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--cpus", default="4")
    a = ap.parse_args()
    if a.mode == "accuracy":
        print(json.dumps(accuracy(a.data, a.n or 500)))
    elif a.mode == "load":
        for row in load(a.url, a.n or 256, a.concurrency):
            print({k: round(v, 1) for k, v in row.items()})
    elif a.mode == "all":
        run_all(500 if a.n is None else a.n, 256, a.concurrency, a.cpus)
    else:
        print_table(json.loads(Path("docs/bench_results.json").read_text()))
    sys.stdout.flush()
