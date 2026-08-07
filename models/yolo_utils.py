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