'''
F' drop-in ResNet image classification (Hugging Face Transformers + PyTorch).

Contract:
  main(folder_path, model_name=...) -> list[(filename, label)]

This matches the original baseline approach (HF ResNet), with `use_fast=False`
on the image processor to avoid the default "fast processor" path.
'''
import os
import sys
import time

import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image


def load_model_and_processor(model_name="microsoft/resnet-18"):
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = ResNetForImageClassification.from_pretrained(model_name)
    return model, image_processor


def process_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs


def classify_image(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    return predicted_class_idx


def _to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)


def _print_benchmark(device, n, total_s, per_image_s, filenames):
    if os.environ.get("SCALES_ML_BENCH", "1") == "0":
        return
    if n == 0 or total_s <= 0:
        return

    avg_fps = n / total_s
    ms = [t * 1000.0 for t in per_image_s]
    inst = [1.0 / t for t in per_image_s if t > 0]

    lines = [
        "--- ResNet-PyTorch ---",
        "  device: %s   images: %d" % (device, n),
        "  total: %.3f s   avg FPS: %.1f" % (total_s, avg_fps),
        "  per-image time (ms):  min=%.1f  max=%.1f  mean=%.1f"
        % (min(ms), max(ms), sum(ms) / n),
    ]
    if inst:
        lines.append(
            "  per-image FPS (instant):  min=%.1f  max=%.1f  mean=%.1f"
            % (min(inst), max(inst), sum(inst) / len(inst)))

    if n <= 20:
        lines.append("  per file:")
        for i in range(n):
            name = filenames[i]
            if len(name) > 40:
                name = name[:37] + "..."
            fps_i = 1.0 / per_image_s[i] if per_image_s[i] > 0 else 0.0
            lines.append("    %-42s  %6.1f FPS  (%5.1f ms)" % (name, fps_i, ms[i]))

    lines.append("---")
    print("\n".join(lines), file=sys.stderr, flush=True)


def main(folder_path, model_name="microsoft/resnet-18"):
    filenames = []
    for name in os.listdir(folder_path):
        low = name.lower()
        if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg"):
            filenames.append(name)
    filenames.sort()

    if len(filenames) == 0:
        print("[ResNet-PyTorch] No images in: " + folder_path, file=sys.stderr, flush=True)
        return []

    model, image_processor = load_model_and_processor(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output = []
    per_image_s = []

    for filename in filenames:
        t0 = time.perf_counter()
        image_path = os.path.join(folder_path, filename)
        inputs = process_image(image_path, image_processor)
        _to_device(inputs, device)
        predicted_class_idx = classify_image(model, inputs)
        predicted_class = model.config.id2label[predicted_class_idx]
        output.append((filename, predicted_class))
        per_image_s.append(time.perf_counter() - t0)

    total_s = sum(per_image_s)
    _print_benchmark(device, len(output), total_s, per_image_s, filenames)

    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "../test-imagery"
    main(folder)
