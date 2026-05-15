# F´ (embedded Python) -> JetPack python -> scales_trt.worker
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, Tuple


def ensure_env() -> None:
    # Jetson defaults (no manual export needed); override before F´ if needed
    os.environ.setdefault("SCALES_TRT_PYTHON", "/usr/bin/python3")
    os.environ.setdefault("TRT_FP16", "1")
    os.environ.setdefault("SCALES_TRT_MODEL", "microsoft/resnet-18")
    os.environ.setdefault("SCALES_YOLO_WEIGHTS", "yolov8n.pt")


def jetson_python() -> str:
    ensure_env()
    return os.environ.get("SCALES_TRT_PYTHON") or os.environ.get("TRT_PYTHON") or "/usr/bin/python3"


def worker(argv: List[str]) -> dict:
    ensure_env()
    r = subprocess.run(
        [jetson_python(), "-m", "scales_trt.worker", *argv],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=os.getcwd(),
    )
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    if not line:
        raise RuntimeError((r.stderr or "").strip() or f"worker exit {r.returncode}")
    payload = json.loads(line)
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "worker failed"))
    return payload


def trt_infer(adapter: str, folder: str, model: str, tag: str) -> List[Tuple[str, str]]:
    # folder = absolute path from SET_INFERENCE_PATH
    fp16 = "1" if os.environ.get("TRT_FP16", "1") != "0" else "0"
    p = worker(["infer", "--adapter", adapter, "--folder", folder, "--model", model, "--fp16", fp16])
    stats = p.get("stats") or {}
    fps = stats.get("fps")
    if fps:
        print(
            f"[{tag}] {stats.get('images')} images | {stats.get('seconds', 0):.3f}s | {fps:.2f} FPS",
            file=sys.stderr,
        )
    return [(str(a), str(b)) for a, b in p.get("outputs", [])]
