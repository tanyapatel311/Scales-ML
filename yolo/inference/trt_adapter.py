# YOLO adapter for scales_trt.worker — copy to new_model/inference/trt_adapter.py as template
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image

SZ = int(os.environ.get("YOLO_IMGSZ", "640"))
CONF = float(os.environ.get("YOLO_CONF_TH", "0.25"))
IOU = float(os.environ.get("YOLO_IOU_TH", "0.45"))


def tag(weights: str) -> str:
    return f"yolo--{os.path.basename(weights)}--imgsz{SZ}"


def _letterbox(im: Image.Image) -> Image.Image:
    w, h = im.size
    r = min(SZ / w, SZ / h)
    nw, nh = int(w * r), int(h * r)
    c = Image.new("RGB", (SZ, SZ), (114, 114, 114))
    c.paste(im.resize((nw, nh), Image.BILINEAR), ((SZ - nw) // 2, (SZ - nh) // 2))
    return c


def preprocess(path: str) -> np.ndarray:
    im = _letterbox(Image.open(path).convert("RGB"))
    a = np.asarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def export_onnx(weights: str, onnx_path: str, labels_path: str) -> None:
    from scales_trt.jetson_compat import apply_jetson_compat

    apply_jetson_compat()
    from ultralytics import YOLO

    m = YOLO(weights)
    json.dump(getattr(m, "names", {}) or {}, open(labels_path, "w"))
    out = m.export(format="onnx", imgsz=SZ, opset=13, dynamic=False, simplify=False)
    src = str(out) if out else os.path.splitext(weights)[0] + ".onnx"
    if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(onnx_path):
        shutil.copy2(src, onnx_path)


def _nms(boxes, scores):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= IOU]
    return np.array(keep, dtype=int)


def postprocess(y: np.ndarray, labels: dict) -> str:
    if y.ndim == 3:
        y = y[0]
    if y.shape[0] < y.shape[1]:
        y = y.T
    xywh, cls = y[:, :4], y[:, 4:]
    conf, cid = cls.max(1), cls.argmax(1)
    m = conf >= CONF
    if not m.any():
        return "No detections"
    xy, wh = xywh[m, :2], xywh[m, 2:]
    xyxy = np.c_[xy - wh / 2, xy + wh / 2]
    conf, cid = conf[m], cid[m]
    i = _nms(xyxy, conf)[0]
    name = labels.get(str(int(cid[i])), labels.get(int(cid[i]), f"class_{cid[i]}"))
    return f"{name}: {float(conf[i]):.2f}"
