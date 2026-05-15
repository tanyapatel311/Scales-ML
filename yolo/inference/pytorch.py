# YOLO via Ultralytics — SET_ML_PATH "yolo.inference.pytorch" (PyTorch, not TensorRT)
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import List, Tuple

from scales_trt.fprime_wrap import jetson_python

WEIGHTS = os.environ.get("SCALES_YOLO_WEIGHTS", "yolov8n.pt")
EXT = (".png", ".jpg", ".jpeg")
_model = None


def _yolo():
    global _model
    if _model is None:
        from scales_trt.jetson_compat import apply_jetson_compat

        apply_jetson_compat()
        from ultralytics import YOLO

        _model = YOLO(WEIGHTS)
    return _model


def _run_folder(folder: str) -> List[Tuple[str, str]]:
    # cv2 only in JetPack worker python, not F´ embedded python
    import cv2

    if not os.path.isdir(folder):
        return [("InvalidPath", "Error")]
    names = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(EXT)]
    if not names:
        return [("None", "No images")]
    m = _yolo()
    out = []
    for name in names:
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            out.append((name, "Read error"))
            continue
        best, conf = "No detections", -1.0
        for r in m.predict(img, verbose=False):
            if r.boxes is None:
                continue
            for box in r.boxes:
                c = float(box.conf[0])
                if c > conf:
                    conf = c
                    best = f"{m.names[int(box.cls[0])]}: {c:.2f}"
        out.append((name, best))
    return out


def main(folder: str) -> List[Tuple[str, str]]:
    # F´ embedded python delegates to JetPack python (has ultralytics + CUDA)
    env = os.environ.copy()
    env["SCALES_YOLO_WORKER"] = "1"
    env.setdefault("SCALES_YOLO_WEIGHTS", WEIGHTS)
    r = subprocess.run(
        [jetson_python(), "-m", "yolo.inference.pytorch", folder],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
    )
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    if not line:
        raise RuntimeError(r.stderr or "yolo worker failed")
    p = json.loads(line)
    if not p.get("ok"):
        raise RuntimeError(p.get("error", "yolo failed"))
    s = p.get("stats") or {}
    if s.get("fps"):
        print(f"[YOLO] {s['images']} imgs | {s['seconds']:.3f}s | {s['fps']:.2f} FPS", file=sys.stderr)
    return [(str(a), str(b)) for a, b in p.get("outputs", [])]


def _worker_json(folder: str) -> int:
    # Subprocess entry: one JSON line on stdout for main() above
    try:
        t0 = time.time()
        outputs = _run_folder(folder)
        dt = time.time() - t0
        n = sum(1 for name, _ in outputs if name not in ("None", "InvalidPath"))
        print(
            json.dumps(
                {"ok": True, "outputs": outputs, "stats": {"images": n, "seconds": dt, "fps": n / dt if dt else None}}
            ),
            flush=True,
        )
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m yolo.inference.pytorch <image_folder>")
    folder = os.path.abspath(sys.argv[1])
    if os.environ.get("SCALES_YOLO_WORKER") == "1":
        raise SystemExit(_worker_json(folder))
    for n, c in main(folder):
        print(f"{n}: {c}")
