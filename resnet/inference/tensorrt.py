# ResNet TensorRT for F´ — SET_ML_PATH "resnet.inference.tensorrt"
from __future__ import annotations

import os
import sys
from typing import List, Tuple

from scales_trt.fprime_wrap import trt_infer

MODEL = os.environ.get("SCALES_TRT_MODEL", "microsoft/resnet-18")
ADAPTER = "resnet.inference.trt_adapter"


def main(folder: str, model: str = MODEL) -> List[Tuple[str, str]]:
    return trt_infer(ADAPTER, folder, model, "ResNet-TRT")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m resnet.inference.tensorrt <image_folder>")
    for n, c in main(os.path.abspath(sys.argv[1])):
        print(f"{n}: {c}")
