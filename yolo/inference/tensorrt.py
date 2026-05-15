# YOLO TensorRT for F´ — SET_ML_PATH "yolo.inference.tensorrt"
from __future__ import annotations

import os
import sys
from typing import List, Tuple

from scales_trt.fprime_wrap import trt_infer

WEIGHTS = os.environ.get("SCALES_YOLO_WEIGHTS", "yolov8n.pt")
ADAPTER = "yolo.inference.trt_adapter"


def main(folder: str, weights: str = WEIGHTS) -> List[Tuple[str, str]]:
    return trt_infer(ADAPTER, folder, weights, "YOLO-TRT")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m yolo.inference.tensorrt <image_folder>")
    for n, c in main(os.path.abspath(sys.argv[1])):
        print(f"{n}: {c}")
