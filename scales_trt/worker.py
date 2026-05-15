# Shared TensorRT worker. New model = add trt_adapter.py with: tag, export_onnx, preprocess, postprocess
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

from . import runtime

IMAGE_EXT = (".png", ".jpg", ".jpeg")
CACHE = os.path.join(os.path.expanduser("~"), ".scales_trt_cache")


def _out(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _adapter(path: str):
    mod = importlib.import_module(path)
    for fn in ("tag", "export_onnx", "preprocess", "postprocess"):
        if not hasattr(mod, fn):
            raise RuntimeError(f"{path} missing {fn}()")
    return mod


def _paths(adapter, model: str, cache_dir: str):
    p = runtime.cache_paths(cache_dir, adapter.tag(model))
    eng = p["engine_fp16"] if os.environ.get("TRT_FP16", "1") != "0" else p["engine_fp32"]
    return p, eng


def _prepare(adapter_mod: str, model: str, cache_dir: str) -> dict:
    ad = _adapter(adapter_mod)
    paths, eng = _paths(ad, model, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.isfile(paths["onnx"]):
        ad.export_onnx(model, paths["onnx"], paths["labels"])
    if not os.path.isfile(eng):
        runtime.build_engine(paths["onnx"], eng)
    return {"engine": eng}


def _infer(adapter_mod: str, folder: str, model: str, cache_dir: str):
    ad = _adapter(adapter_mod)
    paths, eng = _paths(ad, model, cache_dir)
    if not os.path.isfile(eng):
        raise RuntimeError("Engine missing. Run: make ml-trt-setup")
    labels = runtime.load_labels(paths["labels"])
    with open(eng, "rb") as f:
        session = runtime.load_session(f.read())
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXT)]
    if not files:
        return [], {"images": 0, "seconds": 0.0, "fps": None}
    t0 = time.time()
    out = []
    for name in files:
        x = ad.preprocess(os.path.join(folder, name))
        y = runtime.run(session, x)
        out.append((name, ad.postprocess(y, labels)))
    dt = time.time() - t0
    n = len(out)
    return out, {"images": n, "seconds": dt, "fps": (n / dt) if dt else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--adapter", required=True)
    base.add_argument("--model", required=True)
    base.add_argument("--fp16", type=int, default=1)
    base.add_argument("--cache-dir", default=CACHE)
    sub.add_parser("prepare", parents=[base])
    inf = sub.add_parser("infer", parents=[base])
    inf.add_argument("--folder", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "prepare":
            _out({"ok": True, "stats": _prepare(args.adapter, args.model, args.cache_dir)})
        else:
            outputs, stats = _infer(args.adapter, args.folder, args.model, args.cache_dir)
            _out({"ok": True, "outputs": outputs, "stats": stats})
        return 0
    except Exception as e:
        _out({"ok": False, "error": str(e)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
