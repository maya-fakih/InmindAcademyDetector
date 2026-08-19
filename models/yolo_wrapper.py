from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG

from models.yolo_utils import decode_predictions, letterbox
from models.yolov4_loss import Yolov4Loss
from vendor.pytorch_yolov4.darknet2pytorch import Darknet
from vendor.pytorch_yolov4.utils import post_processing

YOLOV4_TINY_CFG = Path(__file__).parent.parent / "vendor" / "pytorch_yolov4" / "yolov4-tiny.cfg"


class YoloWrapper(nn.Module):
    """Adapts an Ultralytics YOLO model to the Torchvision detection interface."""

    def __init__(self, checkpoint: str, num_classes: int) -> None:
        super().__init__()
        pretrained = YOLO(checkpoint).model
        self.model = DetectionModel(cfg=pretrained.yaml, nc=num_classes)
        self.model.load(pretrained)
        # DetectionModel.load() only copies the state_dict, not `.args`. Normally
        # Ultralytics' own Trainer sets model.args to a hyperparameter namespace
        # before training; since we drive training ourselves, v8DetectionLoss's
        # first call (self.hyp.box / .cls / .dfl gains) would otherwise raise
        # AttributeError: 'DetectionModel' object has no attribute 'args'.
        self.model.args = DEFAULT_CFG
        self.num_classes = num_classes

    def forward(self, images_L: list, targets_L: list[dict] | None = None):
        letterboxed = [letterbox(image_CHW) for image_CHW in images_L]
        batch = torch.stack([image for image, _, _, _ in letterboxed])
        transforms = [(scale, left, top) for _, scale, left, top in letterboxed]

        if self.training and targets_L is not None:
            return self._compute_loss(batch, targets_L, transforms)

        output = self.model(batch)
        raw_output = output[0] if isinstance(output, tuple) else output
        return decode_predictions(raw_output, transforms)

    def _compute_loss(
        self,
        batch: torch.Tensor,
        targets_L: list[dict],
        transforms: list[tuple[float, float, float]],
    ) -> dict:
        _, _, h, w = batch.shape
        batch_idx, cls, bboxes = [], [], []
        for image_index, (target, (scale, left, top)) in enumerate(
            zip(targets_L, transforms, strict=True)
        ):
            boxes_xyxy = target["boxes"]
            n = boxes_xyxy.shape[0]
            if n == 0:
                continue
            # Map original-image-space boxes into letterboxed space before normalizing.
            x1 = boxes_xyxy[:, 0] * scale + left
            y1 = boxes_xyxy[:, 1] * scale + top
            x2 = boxes_xyxy[:, 2] * scale + left
            y2 = boxes_xyxy[:, 3] * scale + top
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            bboxes.append(torch.stack([cx, cy, bw, bh], dim=1))
            cls.append((target["labels"] - 1).float())  # our labels are 1-indexed (0=background)
            batch_idx.append(torch.full((n,), image_index, dtype=torch.float32))

        yolo_batch = {
            "img": batch,
            "batch_idx": (torch.cat(batch_idx) if batch_idx else torch.zeros(0)).to(batch.device),
            "cls": (torch.cat(cls) if cls else torch.zeros(0)).to(batch.device),
            "bboxes": (torch.cat(bboxes) if bboxes else torch.zeros(0, 4)).to(batch.device),
        }
        # model.loss() returns (grad-enabled per-component loss tensor, a
        # DETACHED dict of the same values meant only for logging). The
        # detached dict can't be backpropagated -- must build the returned
        # dict from the grad-enabled tensor instead.
        total_loss, _ = self.model.loss(yolo_batch)
        loss_names = self.model.criterion.one2many.loss_names
        return dict(zip(loss_names, total_loss, strict=True))


class Yolov4TinyWrapper(nn.Module):
    """Adapts the vendored Darknet YOLOv4-tiny model to the Torchvision detection interface."""

    def __init__(
        self, num_classes: int, cfg_path: Path = YOLOV4_TINY_CFG, weights_path: str | None = None
    ) -> None:
        super().__init__()
        self.model = Darknet(str(cfg_path))
        if weights_path is not None:
            weights_file = Path(weights_path)
            if not weights_file.is_file():
                raise FileNotFoundError(f"Backbone weights not found at {weights_path}")
            size_bytes = weights_file.stat().st_size
            # yolov4-tiny.conv.29 is ~19.6MB; a partial/failed download would be far
            # smaller and would otherwise load silently with no error and no log line.
            if size_bytes < 1_000_000:
                raise ValueError(
                    f"Backbone weights at {weights_path} look truncated "
                    f"({size_bytes:,} bytes, expected ~19.6MB) -- redownload before training"
                )
            self.model.load_weights(weights_path)
            print(
                f"[backbone] loaded pretrained weights from {weights_path} ({size_bytes:,} bytes)"
            )
        else:
            # Expected/harmless in eval.py and predict.py: they intentionally skip
            # backbone-only pretrain loading here since a full trained checkpoint's
            # state_dict is loaded right after construction, overwriting these
            # random values entirely -- see the comment in eval.py. Only a real
            # problem if you're seeing this during train.py on a *fresh* run with
            # model.checkpoint set in config.yaml (means that path didn't resolve).
            print(
                "[backbone] no pretrained backbone weights given -- yolov4-tiny "
                "constructed with random weights (fine if a trained checkpoint is "
                "about to be loaded on top, e.g. in eval.py/predict.py)"
            )
        self.image_size = self.model.width
        assert self.model.width == self.model.height, "only square net inputs are supported"

        yolo_blocks = [block for block in self.model.blocks if block["type"] == "yolo"]
        anchor_values = [float(value) for value in yolo_blocks[0]["anchors"].split(",")]
        self.anchors = list(zip(anchor_values[0::2], anchor_values[1::2], strict=True))
        self.anchor_masks = [
            [int(index) for index in block["mask"].split(",")] for block in yolo_blocks
        ]
        self.num_classes = num_classes
        self.loss_fn = Yolov4Loss(num_classes, self.anchors, self.anchor_masks, self.image_size)

    def forward(self, images_L: list, targets_L: list[dict] | None = None):
        letterboxed = [letterbox(image_CHW, size=self.image_size) for image_CHW in images_L]
        batch = torch.stack([image for image, _, _, _ in letterboxed])
        transforms = [(scale, left, top) for _, scale, left, top in letterboxed]

        if self.training and targets_L is not None:
            labels = self._build_padded_labels(targets_L, transforms, batch.device)
            head_outputs = self.model(batch)
            return self.loss_fn(head_outputs, labels)

        # Darknet.forward already runs get_region_boxes internally when not
        # training (see vendor/pytorch_yolov4/darknet2pytorch.py), returning the
        # concatenated [boxes, confs] pair directly -- do not decode it twice.
        boxes_confs = self.model(batch)
        detections = post_processing(None, conf_thresh=0.25, nms_thresh=0.45, output=boxes_confs)
        return self._to_predictions(detections, transforms, batch.device)

    def _build_padded_labels(
        self,
        targets_L: list[dict],
        transforms: list[tuple[float, float, float]],
        device: torch.device,
    ) -> torch.Tensor:
        """Return ``(batch, max_objects, 5)`` of ``[x1, y1, x2, y2, class_id]`` in
        letterboxed pixel space, zero-padded to the batch's largest object count."""
        remapped = []
        for target, (scale, left, top) in zip(targets_L, transforms, strict=True):
            boxes_xyxy = target["boxes"]
            n = boxes_xyxy.shape[0]
            if n == 0:
                remapped.append(torch.zeros(0, 5, device=device))
                continue
            x1 = boxes_xyxy[:, 0] * scale + left
            y1 = boxes_xyxy[:, 1] * scale + top
            x2 = boxes_xyxy[:, 2] * scale + left
            y2 = boxes_xyxy[:, 3] * scale + top
            class_id = (target["labels"] - 1).float()  # our labels are 1-indexed (0=background)
            remapped.append(torch.stack([x1, y1, x2, y2, class_id], dim=1))

        max_objects = max((rows.shape[0] for rows in remapped), default=0)
        max_objects = max(max_objects, 1)
        padded = torch.zeros(len(remapped), max_objects, 5, device=device)
        for index, rows in enumerate(remapped):
            padded[index, : rows.shape[0]] = rows
        return padded

    def _to_predictions(
        self,
        detections: list[list[list[float]]],
        transforms: list[tuple[float, float, float]],
        device: torch.device,
    ) -> list[dict]:
        predictions = []
        for image_detections, (scale, left, top) in zip(detections, transforms, strict=True):
            if len(image_detections) == 0:
                predictions.append(
                    {
                        "boxes": torch.zeros(0, 4, device=device),
                        "scores": torch.zeros(0, device=device),
                        "labels": torch.zeros(0, dtype=torch.int64, device=device),
                    }
                )
                continue
            rows = torch.tensor(image_detections, device=device)
            boxes = rows[:, :4] * self.image_size
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
            predictions.append(
                {
                    "boxes": boxes,
                    "scores": rows[:, 4],
                    # vendor's post_processing emits 0-indexed class ids; +1 to line up
                    # with LocoDataset's 1-indexed labels (0 reserved for background).
                    "labels": (rows[:, 6] + 1).to(torch.int64),
                }
            )
        return predictions


def create_yolo_model(
    num_classes: int, checkpoint: str = "yolo26n.pt", architecture: str = "ultralytics"
) -> nn.Module:
    """Create a YOLO model wrapper; ``num_classes`` includes background, YOLO has none.

    ``architecture="ultralytics"`` (default) loads any Ultralytics checkpoint (e.g.
    ``yolo26n.pt``) via :class:`YoloWrapper`. ``architecture="yolov4-tiny"`` builds the
    vendored Darknet YOLOv4-tiny model instead, ignoring ``checkpoint`` unless it points
    at a Darknet ``.weights`` file.
    """
    if architecture == "yolov4-tiny":
        weights_path = checkpoint if checkpoint and checkpoint != "yolo26n.pt" else None
        return Yolov4TinyWrapper(num_classes - 1, weights_path=weights_path)
    return YoloWrapper(checkpoint, num_classes - 1)
