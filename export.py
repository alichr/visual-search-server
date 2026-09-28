"""Export CLIP image/text encoders to ONNX (FP32 + INT8) and check parity.

Usage: python export.py --out models/   (safe to re-run: overwrites with identical output)
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"


class ImageEnc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m.get_image_features(pixel_values=pixel_values).pooler_output


class TextEnc(ImageEnc):
    def forward(self, input_ids, attention_mask):
        out = self.m.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    hf = AutoTokenizer.from_pretrained(MODEL_ID)
    tok = hf.backend_tokenizer  # bake padding/truncation to 77 into tokenizer.json
    tok.enable_truncation(77)
    tok.enable_padding(length=77, pad_id=hf.pad_token_id, pad_token=hf.pad_token)
    tok.save(f"{out}/tokenizer.json")
    enc = tok.encode_batch(["a photo of a cat", "a car"])
    ids, mask = (torch.tensor([getattr(e, k) for e in enc]) for k in ["ids", "attention_mask"])
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    torch.manual_seed(0)
    for name, wrapper, feeds in [
        ("image", ImageEnc, {"pixel_values": torch.randn(2, 3, 224, 224)}),
        ("text", TextEnc, {"input_ids": ids, "attention_mask": mask}),
    ]:
        path, m, args = f"{out}/{name}", wrapper(model).eval(), tuple(feeds.values())
        torch.onnx.export(m, args, f"{path}_fp32.onnx", dynamo=False, opset_version=17,
                          input_names=list(feeds), output_names=["emb"],
                          dynamic_axes={k: {0: "batch"} for k in [*feeds, "emb"]})
        quantize_dynamic(f"{path}_fp32.onnx", f"{path}_int8.onnx", per_channel=True,
                         weight_type=QuantType.QInt8)
        with torch.no_grad():
            ref = m(*args).numpy()
        for kind in ["fp32", "int8"]:
            emb = ort.InferenceSession(f"{path}_{kind}.onnx").run(
                None, {k: v.numpy() for k, v in feeds.items()})[0]
            cos = (ref * emb).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(emb, axis=1))
            mb = os.path.getsize(f"{path}_{kind}.onnx") / 1e6
            print(f"{name} {kind}: cosine vs PyTorch = {cos.min():.5f}, {mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models", help="output directory")
    main(ap.parse_args().out.rstrip("/"))
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: torch + onnxruntime can crash there on macOS
