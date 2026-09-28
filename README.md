# CLIP-Serve: quantised zero-shot visual search

[![CI](https://github.com/alichr/visual-search-server/actions/workflows/ci.yml/badge.svg)](https://github.com/alichr/visual-search-server/actions/workflows/ci.yml)

A small, production-style inference service. It serves OpenAI's CLIP (ViT-B/32), exported to ONNX
and quantised to INT8. You can index a set of images, search them with free text such as "a red car
at night", and classify an image against any labels given in the request, all on a laptop CPU with
no training.

![Searching 20 COCO photos for "a person on a bicycle" in the Swagger UI](docs/swagger_search.png)

*Searching 20 COCO photos for "a person on a bicycle" from `/docs`. Three of the top four results
are cyclists. The third is a small figure riding a donkey, seen from behind.*

## Features

- **Index** images with `POST /index`. Uploads are embedded concurrently and batched together.
- **Text search** with `GET /search`. Cosine similarity over the index is a single matrix multiply.
- **Zero-shot classification** with `POST /classify`, against any comma-separated labels.
- **Async micro-batching.** Requests are grouped for up to `MAX_WAIT_MS` or `MAX_BATCH` items, then
  run as one forward pass.
- **INT8 quantisation.** The models are 3.2× smaller and 2–2.7× faster on Linux, with the same
  Imagenette accuracy as FP32.
- **Torch-free runtime.** The server needs only onnxruntime, NumPy, Pillow and tokenizers. torch
  and transformers are used only for the export.
- **Prometheus metrics and JSON logs:** latency, errors, batch size, queue depth, index size and
  model info, with a request ID on every log line.
- **Docker** (multi-stage, non-root, health-checked), **docker-compose** with Prometheus, and a
  **Kubernetes** manifest tested on kind.
- **CI** on GitHub Actions: lint, tests and a Docker end-to-end check. A GitLab CI mirror is
  included.

## Architecture

```mermaid
flowchart LR
    C([client]) -- "POST /index<br/>POST /classify" --> P["decode + preprocess<br/>(worker thread)"]
    P --> IB["image batcher<br/>≤16 items / ≤10 ms"]
    IB --> IE[["ONNX image encoder<br/>INT8"]]
    IE --> IX[("in-memory index<br/>N × 512, L2-normalised")]
    C -- "GET /search<br/>/classify labels" --> TC{"text cache"}
    TC -- miss --> TB["text batcher"] --> TE[["ONNX text encoder<br/>INT8"]]
    IX -- "index @ query" --> C
    PR[(Prometheus)] -. "scrapes every 5 s" .-> M["/healthz · /readyz · /metrics"]
```

An uploaded image is decoded and preprocessed in a worker thread, so the event loop never blocks.
It then goes to the **image batcher**, which collects requests for up to 10 ms or 16 items and runs
one ONNX forward pass. Each caller receives its own row of the result.

Text (search queries and classification labels) goes through a **cache** first. Only strings it
hasn't seen reach the **text batcher**, which runs one text inference at a time so parallel requests
can't oversubscribe the CPU.

Every embedding is L2-normalised. That makes a search over N images one `(N × 512) @ (512,)` matrix
multiply, and a classification a softmax over 100 × the cosine scores against
`"a photo of a {label}"`.

## Quickstart

```bash
git clone https://github.com/alichr/visual-search-server.git && cd visual-search-server
docker compose up --build        # first build exports the models (~5 min); then open http://localhost:8000/docs
```

In a second terminal, download two sample photos and try the API:

```bash
curl -so cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
curl -so bedroom.jpg http://images.cocodataset.org/val2017/000000000632.jpg

curl -F "files=@cats.jpg" -F "files=@bedroom.jpg" localhost:8000/index
# {"ids":[0,1],"index_size":2}

curl "localhost:8000/search?q=two%20cats%20on%20a%20sofa&k=2"
# [{"id":0,"filename":"cats.jpg","score":0.288},{"id":1,"filename":"bedroom.jpg","score":0.196}]

curl -F "file=@cats.jpg" -F "labels=cat, dog, car" localhost:8000/classify
# {"cat":0.990,"dog":0.008,"car":0.002}
```

Prometheus runs at http://localhost:9090 (try the query `rate(batch_size_count[1m])`). Tagged
releases (`v*`) are also published as `ghcr.io/alichr/visual-search-server:<tag>`.

## API reference

The interactive documentation, with typed schemas, is at **http://localhost:8000/docs**.

| Endpoint | Input | Output | Errors |
| --- | --- | --- | --- |
| `POST /index` | one or more image files (`files`, multipart) | `{"ids": [...], "index_size": n}` | 415 if a file isn't a readable image |
| `GET /search?q=...&k=5` | free-text query; `k` from 1 to 50 | top-k `[{"id", "filename", "score"}]` by cosine | 409 if the index is empty; 422 if `q` is empty |
| `POST /classify` | one image (`file`) and a comma-separated `labels` field | `{label: probability}`, sorted by probability | 415 bad image; 422 no labels |
| `GET /healthz` | – | 200 while the process is alive (liveness) | – |
| `GET /readyz` | – | 200 once the models are loaded and warmed up, 503 before (readiness) | 503 |
| `GET /metrics` | – | Prometheus text format | – |

Uploads are validated by decoding them with Pillow, not by trusting the `Content-Type` header.
Oversized "decompression bomb" images are rejected.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_DIR` | `models` (`/models` in Docker) | Folder holding `{image,text}_{fp32,int8}.onnx` and `tokenizer.json` |
| `PRECISION` | `int8` | `int8` or `fp32`. Any other value stops the server at start-up. |
| `ORT_THREADS` | `2` | ONNX Runtime threads per inference. Set it to the container's CPU limit. |
| `MAX_BATCH` | `16` | Largest micro-batch. `1` turns batching off. |
| `MAX_WAIT_MS` | `10` | Longest a request waits for others to join its batch |

The Docker build argument `INT8_ONLY=1` leaves out the FP32 models. That shrinks the image from
0.67 GB to 0.27 GB compressed; `PRECISION=fp32` then fails immediately with a clear error.

## Design decisions

- **Why ONNX Runtime.** It's a small, CPU-optimised runtime, so the image needs no torch or
  transformers. It fuses operators, lets you set thread counts explicitly, and can switch to CUDA,
  TensorRT or OpenVINO execution providers without code changes. Parity with PyTorch is checked
  on every export: cosine 1.00000 for FP32.
- **Why dynamic INT8 quantisation.** It needs no calibration data. Weights are stored as 8-bit
  integers, which makes them 4× smaller, and activations are quantised on the fly. The details
  matter, and each one below came from measuring:
  - **UInt8 weights, not Int8.** On x86 CPUs without AVX-512 VNNI, the Int8 kernel saturates. On a
    GitHub Actions runner, image-embedding cosine against FP32 dropped to **0.40**. UInt8 weights
    give about 0.99 on macOS arm64, Linux arm64 and x86 AVX2.
  - **The text encoder's MLP `fc2` layers stay FP32.** Their activations have outliers. Without
    this, text cosine is about 0.89 on Linux; with it, 0.999. The text model is 102 MB instead of
    64 MB.
  - **A parity gate.** `export.py` fails the build if FP32 cosine drops below 0.999 or INT8 below
    0.97, so a bad model can't ship silently.
  - **The trade-off.** With dynamic quantisation, an INT8 embedding depends slightly on which
    other images share its batch, because the activation scale is computed per batch. On real
    photos this is typically cosine ≥ 0.99. `PRECISION=fp32` is fully deterministic.
- **How batching trades latency for throughput.** Waiting up to `MAX_WAIT_MS` lets concurrent
  requests share one forward pass. Under load that raises throughput (+29% for INT8 at
  concurrency 32). At low load it only adds latency (about +30 ms at concurrency 1, because a
  `/classify` goes through both batchers). A batcher runs one batch at a time, so ONNX Runtime
  always has all `ORT_THREADS` cores to itself.
- **Explicit thread counts.** ONNX Runtime uses every host core by default, and inside a
  container it can still see all of the host's cores. With 5 workers sharing 10 cores, the default
  gave a p95 of 105 ms, against 30 ms with `ORT_THREADS=2`.
- **Liveness vs readiness.** `/healthz` only says the process is up; if it fails, Kubernetes
  restarts the pod. `/readyz` turns 200 only after warm-up has actually run the models, and it
  stays 503 if warm-up fails. So traffic only reaches pods that can serve it, and a slow start is
  never mistaken for a crash.
- **One worker per container.** The batcher and the index live inside one process. Several
  uvicorn workers would split the batches, load the models several times over, and each hold a
  different index. You scale by adding replicas, each sized with `ORT_THREADS` equal to its CPU
  limit.
- **Matching preprocessing exactly.** Preprocessing mismatches are the most common cause of
  silent accuracy drops in served models. The NumPy/Pillow version here matches the Hugging Face
  processor exactly on 9 test images, including grayscale, RGBA, tiny and odd-sized ones. It also
  follows EXIF rotation, which the Hugging Face processor doesn't, so sideways phone photos come
  out upright.

## Results

Hardware: Apple M4 laptop (4 performance + 6 efficiency cores), Docker Desktop Linux aarch64 VM.
Each configuration ran in a fresh container with `--cpus=4` and `ORT_THREADS=4`.

- **Accuracy:** zero-shot top-1 on 500 Imagenette validation images (50 per class), with the prompt
  `"a photo of a {class}"`.
- **Load:** 256 `/classify` requests at each concurrency level c (one photo, 10 labels), sent with
  `httpx.AsyncClient`.

Reproduce it with `python bench.py all`. The raw numbers are in `docs/bench_results.json`.

| Config | Model size | Imagenette top-1 | c=1 p50 / p95 (ms) | c=1 req/s | c=8 p50 / p95 (ms) | c=8 req/s | c=32 p50 / p95 (ms) | c=32 req/s | c=32 mean batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32, no batching | 606 MB | 98.4% | 31 / 34 | 31.7 | 232 / 242 | 34.3 | 935 / 954 | 34.0 | 1.0 |
| FP32, batching | 606 MB | 98.4% | 61 / 66 | 16.3 | 210 / 219 | 37.8 | 808 / 952 | 38.2 | 15.1 |
| INT8, no batching | 191 MB | 98.4% | 14 / 17 | 66.8 | 99 / 115 | 78.5 | 401 / 421 | 78.8 | 1.0 |
| INT8, batching | 191 MB | 98.4% | 45 / 48 | 22.4 | 88 / 101 | 89.1 | 303 / 336 | **101.9** | 15.1 |

**In two sentences:** INT8 is the clear win, with a 3.2× smaller model and 2–2.7× the throughput at
the same 98.4% accuracy. Micro-batching adds a further 29% under load, but costs about 30 ms per
request at low load, so on CPU it's worth enabling only when requests actually overlap.

More detail:

- **Platform matters.** On macOS the same INT8 files were barely faster than FP32. The Linux
  kernels are where INT8 pays off.
- **Why batching gains are modest on CPU.** A batch of 16 costs almost 16× a single image, so
  batching saves only per-call overhead. On a GPU the gain would be much larger.
- **Sanity check.** At c=32 without batching, p50 ≈ concurrency ÷ throughput (32 / 78.8 ≈ 406 ms;
  measured 401 ms). That latency is queueing, not slow inference.
- **A bottleneck the benchmark found.** The first run gave only 5.1 req/s at c=32, and throughput
  *fell* as concurrency rose. Profiling showed every `/classify` re-embedding its 10 labels: 113 ms
  of text-encoder work against 22 ms for the image. Those calls also ran in parallel threads that
  oversubscribed the CPU. The text cache and the text batcher raised throughput **20×**, to
  101.9 req/s. The benchmark reuses the same labels, so it measures the cached path; never-seen
  labels still pay the text cost.

During a c=32 run, the mean batch size sits at the `MAX_BATCH=16` cap:

![Prometheus: mean batch size and batch-size buckets during a concurrency-32 run](docs/prometheus_batch_size.png)

## Testing and CI

`pytest -q` runs 3 smoke tests in about 1 s. They go through `TestClient`, so the real lifespan
(batcher tasks, warm-up) and the INT8 models are used. Test images are generated with Pillow, so
no test assets are committed.

1. `/readyz` returns 200 after start-up.
2. A solid-red image with the labels "red, blue" is classified as red. This test catches
   channel-order bugs: when BGR is swapped in on purpose, it fails.
3. After indexing a green and a yellow square, "a green square" returns both, green first.

GitHub Actions (`.github/workflows/ci.yml`) is the pipeline that actually runs, on every push and
pull request:

| Stage | What it does |
| --- | --- |
| **lint** | `ruff check .` and `ruff format --check .` |
| **test** | Installs CPU-only torch, runs `export.py` (which fails on bad FP32/INT8 parity), then `pytest` against INT8. pip and Hugging Face downloads are cached. |
| **build** | Builds the image with cached layers, runs it, waits for `/readyz`, and sends one end-to-end `/classify` request. On `v*` tags it also pushes the image to GitHub Container Registry. |

`.gitlab-ci.yml` mirrors the same three stages for GitLab CI, with Docker-in-Docker for the build.
It isn't wired to a GitLab project.

## Deployment

**Docker.** Multi-stage build: the exporter stage installs CPU-only torch and runs `export.py`. The
runtime stage has no torch or transformers, runs as a non-root user (uid 10001), and has a
`HEALTHCHECK` on `/readyz`. Image size is 0.67 GB compressed (1.89 GB in `docker images`), or
0.27 GB with `INT8_ONLY=1`. At runtime it uses about 310 MiB of memory idle and about 450 MiB under
load.

**Kubernetes.** `deploy/k8s.yaml` defines a Deployment (2 replicas) and a ClusterIP Service. It was
tested on [kind](https://kind.sigs.k8s.io/):

```bash
kind create cluster --name clip
kind load docker-image clip-serve:latest --name clip
kubectl apply -f deploy/k8s.yaml
kubectl rollout status deployment/clip-serve
kubectl port-forward svc/clip-serve 8000:80   # then open http://localhost:8000/docs
```

- **Probes:** liveness on `/healthz`, readiness on `/readyz`.
- **Resources:** 1–2 CPUs with `ORT_THREADS=2`, and 512 Mi–1 Gi of memory.
- **Hardening:** non-root, a read-only root filesystem, all capabilities dropped, and a writable
  `/tmp` volume for uploads.
- **Rolling updates:** `maxUnavailable: 0` plus a 5 s `preStop` pause. Under load, 1 of ~1,750
  requests failed across two rolling restarts; without the pause, 5 of ~740 failed.
- **Metrics:** Prometheus scrape annotations on the pod template.

## Limitations and next steps

- **The index is in memory, per replica.** Through a 2-replica Service, `index_size` came back as 2
  or 0 depending on which pod answered, and a restart empties the index. Next step: store the
  embeddings in a shared vector database (pgvector, Qdrant or Milvus), make the pods stateless, and
  add an `INDEX_DIR` option to index a folder at start-up.
- **CPU only.** Next step: GPU, via ONNX Runtime's CUDA or TensorRT execution provider, where
  batching pays off far more. Beyond that, a dedicated serving platform such as NVIDIA Triton or
  Ray Serve, which provides dynamic batching, multi-model serving and GPU scheduling.
- **Autoscaling.** Next step: a HorizontalPodAutoscaler driven by the `queue_depth` metric (through
  the Prometheus adapter), rather than by CPU.
- **Batching at low load.** Next step: dispatch straight away when nothing else is queued, instead
  of always waiting `MAX_WAIT_MS`. That removes the ~30 ms cost at concurrency 1.
- **Quantisation.** Next step: static (calibrated) quantisation. INT8 embeddings would then no
  longer depend on which images share a batch, and it's often faster on x86.
- **Security.** Next step: authentication (API keys or OAuth2), rate limiting, upload size limits
  and TLS termination at an ingress. None of these is implemented yet.
- **Text cache per replica.** It's bounded at 50k entries and then cleared. Next step: an LRU
  policy or a shared cache.

## Development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
# Linux only: install CPU-only torch first, so pip doesn't fetch the ~2 GB CUDA build
# pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python export.py --out models/          # FP32 + INT8 ONNX, with the parity check (~1 min)
pytest -q && ruff check . && ruff format --check .
uvicorn app.main:app --port 8000        # then open http://localhost:8000/docs
```

```
app/model.py      ONNX sessions, preprocessing, tokenisation, cached embeddings
app/batcher.py    async micro-batching queue with batch-size / queue-depth metrics
app/main.py       FastAPI routes, index, middleware (latency, errors, JSON logs), health
tests/test_api.py smoke tests
export.py         export both encoders to ONNX, quantise to INT8, check parity (tooling)
bench.py          Imagenette accuracy + load test, runs every config in Docker (tooling)
deploy/           prometheus.yml, k8s.yaml
```

Model weights (`models/`), datasets (`data/`) and `*.onnx` files are never committed.

## Line count and credits

The service and its tests are under 200 lines of Python. Counted with
[cloc](https://github.com/AlDanial/cloc) (`cloc --include-lang=Python app tests`):

```
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                           4             69             16            199
-------------------------------------------------------------------------------
```

The offline tooling, `export.py` and `bench.py`, adds 263 lines, for 462 in total.

- **Model:** CLIP ViT-B/32 by OpenAI, released under the MIT licence
  ([openai/CLIP](https://github.com/openai/CLIP)) and loaded from Hugging Face as
  `openai/clip-vit-base-patch32`. Radford et al., *Learning Transferable Visual Models From Natural
  Language Supervision*, ICML 2021.
  ```bibtex
  @inproceedings{radford2021clip,
    title     = {Learning Transferable Visual Models From Natural Language Supervision},
    author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and
                 Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and
                 Krueger, Gretchen and Sutskever, Ilya},
    booktitle = {International Conference on Machine Learning (ICML)},
    year      = {2021}
  }
  ```
- **Evaluation data:** [Imagenette](https://github.com/fastai/imagenette) by fast.ai, a 10-class
  subset of ImageNet. It's downloaded for benchmarking and not redistributed. The demo photos are
  from [COCO](https://cocodataset.org/) val2017, fetched at run time and not committed.
- **Licence:** this repository is released under the Apache License 2.0 (see `LICENSE`).
