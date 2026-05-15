# ResNet adapter for scales_trt.worker
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

H, W, EDGE = 224, 224, 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tag(model: str) -> str:
    return model.replace("/", "--")


def preprocess(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = EDGE / min(w, h)
    im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - W) // 2, (h - H) // 2
    a = np.asarray(im.crop((l, t, l + W, t + H)), dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def export_onnx(model: str, onnx_path: str, labels_path: str) -> None:
    from transformers import ResNetForImageClassification

    hf = ResNetForImageClassification.from_pretrained(model).eval()
    json.dump(hf.config.id2label, open(labels_path, "w"))

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).logits

    w = Wrap(hf).eval()
    dummy = torch.randn(1, 3, H, W)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(w, dummy, onnx_path, opset_version=13, input_names=["input"], output_names=["logits"])


def postprocess(y: np.ndarray, labels: dict) -> str:
    i = int(np.argmax(y, axis=-1))
    return labels.get(str(i), labels.get(i, f"class_{i}"))
