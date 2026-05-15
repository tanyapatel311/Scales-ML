# Scales-ML

How to add a **new model** to SCALES F´. To run ResNet/YOLO that already exist, see [fprime-scales-ref](https://github.com/BroncoSpace-Lab/fprime-scales-ref).

## What F´ needs

A Python module with `main(folder)` that returns a list of `(filename, result_string)` pairs — one per image in the folder from `SET_INFERENCE_PATH`.

Example: `SET_ML_PATH "mymodel.inference.tensorrt"`

## 1. Copy a model folder

Duplicate `yolo/` → `mymodel/` (or `resnet/` for classification). You need:

```
mymodel/inference/trt_adapter.py   # export ONNX + preprocess + postprocess
mymodel/inference/tensorrt.py      # thin TRT wrapper
```

Add empty `__init__.py` files in `mymodel/` and `mymodel/inference/` if missing.

## 2. Edit `trt_adapter.py`

Start from `yolo/inference/trt_adapter.py` or `resnet/inference/trt_adapter.py`. Keep these four functions:

- `tag` — unique name for the TRT cache
- `export_onnx` — writes ONNX + labels (runs once during `make ml-trt-setup`)
- `preprocess` — one image → NCHW float32 numpy array
- `postprocess` — TRT output → one short string for GDS

`preprocess` input size must match the ONNX you export.

## 3. Edit `tensorrt.py`

Copy `yolo/inference/tensorrt.py` and set your adapter path and model id:

```python
ADAPTER = "mymodel.inference.trt_adapter"
MODEL = "your-weights-or-model-id"

def main(folder: str, model: str = MODEL):
    return trt_infer(ADAPTER, folder, model, "MyModel-TRT")
```

## 4. Hook up fprime-scales-ref

In the parent repo:

- **`scripts/jetson-python.sh`** — add `"$ML/mymodel"` to the `cp -r` line
- **`scripts/ml_trt_setup.sh`** — add a `mymodel)` case (copy the `yolo)` block, change adapter + model name)

On the Jetson:

```bash
make build-jetson
MODEL=mymodel make ml-trt-setup    # one-time; can take a long time
bash scripts/jetson-python.sh      # after code changes
```

## 5. Test and run

```bash
cd build-python-fprime-aarch64-linux
/usr/bin/python3 -m mymodel.inference.tensorrt /path/to/test-imagery
```

Then in GDS:

```text
jetson_mlManager.SET_ML_PATH "mymodel.inference.tensorrt"
jetson_mlManager.SET_INFERENCE_PATH "../test-imagery"
jetson_mlManager.MULTI_INFERENCE
```

## PyTorch only (no TensorRT)

Copy `yolo/inference/pytorch.py`, implement `main(folder)`, add `mymodel` to `jetson-python.sh`, use `SET_ML_PATH "mymodel.inference.pytorch"`. No `ml-trt-setup`.
