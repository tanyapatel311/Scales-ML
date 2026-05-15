# Jetson: torch.load(weights_only) + torchvision NMS without C++ ops
def apply_jetson_compat() -> None:
    import torch

    if not getattr(torch.load, "_scales", False):
        orig = torch.load

        def load(*a, **k):
            k.pop("weights_only", None)
            return orig(*a, **k)

        load._scales = True
        torch.load = load
    try:
        import torchvision.ops as ops

        ops.nms(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([0.9]),
            0.5,
        )
    except Exception:
        import torchvision.ops as ops

        def _nms(boxes, scores, iou):
            import torch

            if not boxes.numel():
                return torch.empty(0, dtype=torch.int64, device=boxes.device)
            x1, y1, x2, y2 = boxes.unbind(1)
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel():
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
                order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= iou]
            return torch.tensor(keep, device=boxes.device)

        ops.nms = _nms
        ops.batched_nms = lambda b, s, idx, iou: _nms(
            b + idx.to(b).unsqueeze(1) * (b.max() + 1), s, iou
        )
