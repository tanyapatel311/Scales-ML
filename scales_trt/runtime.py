# TensorRT build + run (TRT 8/9 and 10)
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorrt as trt
import torch


def cache_paths(cache_dir: str, tag: str) -> dict:
    return {
        "onnx": os.path.join(cache_dir, f"{tag}.onnx"),
        "engine_fp16": os.path.join(cache_dir, f"{tag}_fp16.engine"),
        "engine_fp32": os.path.join(cache_dir, f"{tag}_fp32.engine"),
        "labels": os.path.join(cache_dir, f"{tag}_labels.json"),
    }


def load_labels(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _workspace(config, mb: int) -> None:
    b = mb * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, b)
    else:
        config.max_workspace_size = b


def _to_bytes(blob) -> bytes:
    # TRT 10 returns IHostMemory, not bytes
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if hasattr(blob, "tobytes"):
        return blob.tobytes()
    try:
        return bytes(blob)
    except TypeError:
        return bytes(bytearray(blob))


def build_engine(onnx_path: str, engine_path: str) -> bytes:
    fp16 = os.environ.get("TRT_FP16", "1") != "0"
    print(f"[TRT] building {engine_path} ...", file=sys.stderr, flush=True)
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, log)
    cfg = builder.create_builder_config()
    _workspace(cfg, int(os.environ.get("TRT_WORKSPACE_MB", "256")))
    if fp16 and getattr(builder, "platform_has_fast_fp16", False):
        cfg.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    stop = threading.Event()
    iv = int(os.environ.get("TRT_BUILD_HEARTBEAT_S", "0"))  # 0=off, 60=heartbeat every 60s

    def _tick():
        t0 = time.time()
        while not stop.wait(iv):
            print(f"[TRT] still building ({int(time.time() - t0)}s)", file=sys.stderr, flush=True)

    th = threading.Thread(target=_tick, daemon=True) if iv > 0 else None
    if th:
        th.start()
    try:
        blob = builder.build_serialized_network(net, cfg)
    finally:
        stop.set()
    if blob is None:
        raise RuntimeError("engine build failed")
    data = _to_bytes(blob)
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"[TRT] saved {engine_path}", file=sys.stderr, flush=True)
    return data


@dataclass
class TrtSession:
    ctx: Any
    stream: torch.cuda.Stream
    trt10: bool
    in_name: str | None
    out_name: str | None
    in_buf: torch.Tensor
    out_buf: torch.Tensor
    bindings: list | None
    out_shape: tuple


def load_session(engine_bytes: bytes) -> TrtSession:
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_bytes)
    ctx = eng.create_execution_context()
    stream = torch.cuda.Stream()
    trt10 = hasattr(eng, "num_io_tensors")
    if trt10:
        ins = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        outs = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        inn, outn = ins[0], outs[0]
        ish, osh = tuple(eng.get_tensor_shape(inn)), tuple(eng.get_tensor_shape(outn))
        idt = torch.float16 if np.dtype(trt.nptype(eng.get_tensor_dtype(inn))) == np.float16 else torch.float32
        odt = torch.float16 if np.dtype(trt.nptype(eng.get_tensor_dtype(outn))) == np.float16 else torch.float32
        in_buf = torch.empty(int(np.prod(ish)), dtype=idt, device="cuda")
        out_buf = torch.empty(int(np.prod(osh)), dtype=odt, device="cuda")
        ctx.set_tensor_address(inn, int(in_buf.data_ptr()))
        ctx.set_tensor_address(outn, int(out_buf.data_ptr()))
        return TrtSession(ctx, stream, True, inn, outn, in_buf, out_buf, None, osh)
    binds, in_buf, out_buf, osh = [], None, None, None
    for i in range(eng.num_bindings):
        sh = tuple(eng.get_binding_shape(i))
        dt = trt.nptype(eng.get_binding_dtype(i))
        t = torch.float16 if np.dtype(dt) == np.float16 else torch.float32
        buf = torch.empty(int(np.prod(sh)), dtype=t, device="cuda")
        binds.append(int(buf.data_ptr()))
        if eng.binding_is_input(i):
            in_buf = buf
        else:
            out_buf, osh = buf, sh
    return TrtSession(ctx, stream, False, None, None, in_buf, out_buf, binds, osh)


def run(sess: TrtSession, x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x).cuda().reshape(-1)
    if sess.in_buf.dtype == torch.float16:
        t = t.half()
    sess.in_buf.copy_(t)
    with torch.cuda.stream(sess.stream):
        if sess.trt10:
            sess.ctx.execute_async_v3(sess.stream.cuda_stream)
        else:
            sess.ctx.execute_async_v2(sess.bindings, sess.stream.cuda_stream)
    sess.stream.synchronize()
    out = sess.out_buf.float() if sess.out_buf.dtype != torch.float32 else sess.out_buf
    return out.cpu().numpy().reshape(sess.out_shape)
