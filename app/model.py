"""ONNX sessions, preprocessing, embeddings. Runtime deps only: no torch, no transformers."""

import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "models")
PRECISION = os.getenv("PRECISION", "int8")
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))
if PRECISION not in ("fp32", "int8"):
    raise ValueError(f"PRECISION must be 'fp32' or 'int8', got {PRECISION!r}")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _session(name: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS  # explicit: ORT's default = all host cores
    opts.inter_op_num_threads = 1
    path = f"{MODEL_DIR}/{name}_{PRECISION}.onnx"
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


image_session, text_session = _session("image"), _session("text")
tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")  # pads/truncates to 77


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> (3, 224, 224) float32, matching the Hugging Face CLIP processor."""
    img = ImageOps.exif_transpose(img).convert("RGB")  # honour phone-camera rotation
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)  # int(), not round(): HF truncates
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def _normalise(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def embed_images(batch: np.ndarray) -> np.ndarray:
    """(B, 3, 224, 224) float32 -> (B, 512) L2-normalised float32."""
    return _normalise(image_session.run(None, {"pixel_values": batch})[0])


def embed_text(texts: list[str]) -> np.ndarray:
    """B strings -> (B, 512) L2-normalised float32."""
    enc = tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    return _normalise(text_session.run(None, {"input_ids": ids, "attention_mask": mask})[0])


def warmup() -> None:
    """First ONNX run is slow (memory arenas, kernel selection): pay that cost at startup."""
    embed_images(np.zeros((1, 3, 224, 224), dtype=np.float32))
    embed_text(["warmup"])
