from ultralytics.utils import ops
import torch

def decode_predictions(raw_output: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45) -> list[dict]:
    """Convert raw YOLO head output into Torchvision-style detection dicts."""
    results = ops.non_max_suppression(raw_output, conf_thres=conf_thres, iou_thres=iou_thres)
    predictions = []
    for detections in results:  # one tensor per image, columns: x1,y1,x2,y2,conf,cls
        predictions.append({
            "boxes": detections[:, :4],
            "scores": detections[:, 4],
            "labels": detections[:, 5].to(torch.int64),
        })
    return predictions

import torch.nn.functional as F

def letterbox(image_CHW: torch.Tensor, size: int = 640) -> torch.Tensor:
    """Resize+pad a CHW image to a size x size square, preserving aspect ratio."""
    _, h, w = image_CHW.shape
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = F.interpolate(image_CHW.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    pad_h, pad_w = size - new_h, size - new_w
    padded = F.pad(resized, (0, pad_w, 0, pad_h), value=0.447)  # YOLO's grey pad value
    return padded