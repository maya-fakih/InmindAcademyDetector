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

    def forward(self, images_L: list) -> list[dict]:
        batch = torch.stack([letterbox(image_CHW) for image_CHW in images_L])
        output = self.model(batch)
        raw_output = output[0] if isinstance(output, tuple) else output
        return decode_predictions(raw_output)

def create_yolo_model(num_classes: int, checkpoint: str = "yolo26n.pt") -> nn.Module:
    return YoloWrapper(checkpoint, num_classes - 1)  # dataset.num_classes includes background; YOLO has nonedef create_yolo_model(num_classes: int, checkpoint: str = "yolo26n.pt") -> nn.Module: