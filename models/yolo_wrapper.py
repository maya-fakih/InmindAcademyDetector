import torch
from torch import nn
from ultralytics import YOLO

from models.yolo_utils import decode_predictions, letterbox
from ultralytics.nn.tasks import DetectionModel


class YoloWrapper(nn.Module):
    """Adapts an Ultralytics YOLO model to the Torchvision detection interface."""

    def __init__(self, checkpoint: str, num_classes: int) -> None:
        super().__init__()
        pretrained = YOLO(checkpoint).model
        self.model = DetectionModel(cfg=pretrained.yaml, nc=num_classes)
        self.model.load(pretrained)
        self.num_classes = num_classes

    def forward(self, images_L: list, targets_L: list[dict] | None = None):
        batch = torch.stack([letterbox(image_CHW) for image_CHW in images_L])

        if self.training and targets_L is not None:
            return self._compute_loss(batch, targets_L)

        output = self.model(batch)
        raw_output = output[0] if isinstance(output, tuple) else output
        return decode_predictions(raw_output)

    def _compute_loss(self, batch: torch.Tensor, targets_L: list[dict]) -> dict:
        _, _, h, w = batch.shape
        batch_idx, cls, bboxes = [], [], []
        for image_index, target in enumerate(targets_L):
            boxes_xyxy = target["boxes"]
            n = boxes_xyxy.shape[0]
            if n == 0:
                continue
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / w
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / h
            bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / w
            bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / h
            bboxes.append(torch.stack([cx, cy, bw, bh], dim=1))
            cls.append((target["labels"] - 1).float())  # our labels are 1-indexed (0=background)
            batch_idx.append(torch.full((n,), image_index, dtype=torch.float32))

        yolo_batch = {
            "img": batch,
            "batch_idx": torch.cat(batch_idx) if batch_idx else torch.zeros(0),
            "cls": torch.cat(cls) if cls else torch.zeros(0),
            "bboxes": torch.cat(bboxes) if bboxes else torch.zeros(0, 4),
        }
        _, loss_items = self.model.loss(yolo_batch)
        return {"box_loss": loss_items[0], "cls_loss": loss_items[1], "dfl_loss": loss_items[2]}

def create_yolo_model(num_classes: int, checkpoint: str = "yolo26n.pt") -> nn.Module:
    """Create a YOLO model wrapper; ``num_classes`` includes background, YOLO has none."""
    return YoloWrapper(checkpoint, num_classes - 1)