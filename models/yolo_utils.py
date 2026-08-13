import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def letterbox(image_CHW: torch.Tensor, size: int = 640) -> tuple[torch.Tensor, float, float, float]:
    """Resize+pad a CHW image to a size x size square, preserving aspect ratio.

    Returns the padded image along with the scale factor and the (left, top) pad
    offsets applied, so callers can remap boxes into or out of letterboxed space.
    """
    _, h, w = image_CHW.shape
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = F.interpolate(
        image_CHW.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0]
    pad_h, pad_w = size - new_h, size - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = F.pad(
        resized, (left, pad_w - left, top, pad_h - top), value=0.447
    )  # YOLO's grey pad value
    return padded, scale, left, top


def decode_predictions(
    raw_output: torch.Tensor,
    transforms: list[tuple[float, float, float]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> list[dict]:
    """Convert raw YOLO head output into Torchvision-style detection dicts,
    with boxes remapped from letterboxed space back to original image space."""
    results = ops.non_max_suppression(raw_output, conf_thres=conf_thres, iou_thres=iou_thres)
    predictions = []
    for detections, (scale, left, top) in zip(results, transforms, strict=True):
        boxes = detections[:, :4].clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
        predictions.append(
            {
                "boxes": boxes,
                "scores": detections[:, 4],
                # YOLO's cls ids are 0-indexed (no background class); the dataset's
                # labels are 1-indexed with 0 reserved for background (see
                # LocoDataset.category_labels / _compute_loss's `labels - 1`).
                # Map back so predicted and ground-truth label ids line up in
                # compute_map50 / MeanAveragePrecision.
                "labels": detections[:, 5].to(torch.int64) + 1,
            }
        )
    return predictions
