"""Export CLIP image/text encoders to ONNX (FP32 + INT8) and check parity against PyTorch.

Usage: python export.py --out models/   (idempotent: re-running overwrites with identical files)
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"


class Encoder(torch.nn.Module):
    """Expose one CLIP tower as a plain tensor -> tensor module so ONNX traces only that tower."""

    def __init__(self, clip: CLIPModel, kind: str):
        super().__init__()
        self.clip, self.kind = clip, kind

    def forward(self, *inputs):
        return getattr(self.clip, f"get_{self.kind}_features")(*inputs).pooler_output


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    hf = AutoTokenizer.from_pretrained(MODEL_ID)
    tok = hf.backend_tokenizer  # bake padding/truncation to 77 into tokenizer.json
    tok.enable_truncation(77)
    tok.enable_padding(length=77, pad_id=hf.pad_token_id, pad_token=hf.pad_token)
    tok.save(f"{out}/tokenizer.json")
    enc = tok.encode_batch(["a photo of a cat", "a car"])
    ids, mask = (torch.tensor([getattr(e, k) for e in enc]) for k in ("ids", "attention_mask"))
    clip = CLIPModel.from_pretrained(MODEL_ID).eval()
    torch.manual_seed(0)
    for kind, feeds in [("image", {"pixel_values": torch.randn(2, 3, 224, 224)}),
                        ("text", {"input_ids": ids, "attention_mask": mask})]:
        path, m, args = f"{out}/{kind}", Encoder(clip, kind).eval(), tuple(feeds.values())
        torch.onnx.export(m, args, f"{path}_fp32.onnx", dynamo=False, opset_version=17,
                          input_names=list(feeds), output_names=["emb"],
                          dynamic_axes={k: {0: "batch"} for k in [*feeds, "emb"]})
        # text MLP fc2 outputs have outliers that Linux/aarch64 INT8 kernels mishandle: keep FP32
        nodes = onnx.load(f"{path}_fp32.onnx").graph.node
        keep = [n.name for n in nodes if kind == "text" and "fc2" in n.name]
        quantize_dynamic(f"{path}_fp32.onnx", f"{path}_int8.onnx", per_channel=True,
                         weight_type=QuantType.QInt8, nodes_to_exclude=keep)
        with torch.no_grad():
            ref = m(*args).numpy()
        np_feeds = {k: v.numpy() for k, v in feeds.items()}
        for p in ("fp32", "int8"):
            emb = ort.InferenceSession(f"{path}_{p}.onnx").run(None, np_feeds)[0]
            cos = (ref * emb).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(emb, axis=1))
            mb = os.path.getsize(f"{path}_{p}.onnx") / 1e6
            print(f"{kind} {p}: cosine vs PyTorch = {cos.min():.5f}, {mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1].rstrip("/") if "--out" in sys.argv else "models")
    os._exit(0)  # skip interpreter teardown: torch + onnxruntime can crash there on macOS
