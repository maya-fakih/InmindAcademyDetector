"""YOLOv4 training loss for the vendored Darknet ``yolo``-head architecture.

The minimal file set vendored from Tianxiaomo/pytorch-YOLOv4 (see
``vendor/pytorch_yolov4``) only ports the inference-time decode
(``YoloLayer``/``RegionLoss`` in that vendor tree does not implement the loss
used by ``[yolo]`` cfg blocks). This module ports the ``Yolo_loss`` class from
that project's own ``train.py`` (Apache-2.0), generalized from a hardcoded
3-head/608px assumption to the ``num_heads``/``image_size`` used here (2 heads
for yolov4-tiny at 416px), and computing its grids per-batch instead of once
at a fixed batch size so the final (possibly smaller) batch of an epoch works.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def bboxes_ciou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Complete-IoU between every box in ``boxes_a`` and every box in ``boxes_b``.

    Both are ``xywh`` (center-x, center-y, width, height). Returns an
    ``(len(boxes_a), len(boxes_b))`` matrix.
    """
    top_left = torch.max(
        boxes_a[:, None, :2] - boxes_a[:, None, 2:] / 2, boxes_b[:, :2] - boxes_b[:, 2:] / 2
    )
    bottom_right = torch.min(
        boxes_a[:, None, :2] + boxes_a[:, None, 2:] / 2, boxes_b[:, :2] + boxes_b[:, 2:] / 2
    )
    convex_tl = torch.min(
        boxes_a[:, None, :2] - boxes_a[:, None, 2:] / 2, boxes_b[:, :2] - boxes_b[:, 2:] / 2
    )
    convex_br = torch.max(
        boxes_a[:, None, :2] + boxes_a[:, None, 2:] / 2, boxes_b[:, :2] + boxes_b[:, 2:] / 2
    )
    center_distance_sq = ((boxes_a[:, None, :2] - boxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

    width_a, height_a = boxes_a[:, 2], boxes_a[:, 3]
    width_b, height_b = boxes_b[:, 2], boxes_b[:, 3]
    area_a = torch.prod(boxes_a[:, 2:], 1)
    area_b = torch.prod(boxes_b[:, 2:], 1)

    inside = (top_left < bottom_right).type(top_left.type()).prod(dim=2)
    area_intersection = torch.prod(bottom_right - top_left, 2) * inside
    area_union = area_a[:, None] + area_b - area_intersection
    iou = area_intersection / area_union

    diagonal_sq = torch.pow(convex_br - convex_tl, 2).sum(dim=2) + 1e-16
    v = (4 / np.pi**2) * torch.pow(
        torch.atan(width_a / height_a).unsqueeze(1) - torch.atan(width_b / height_b), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v)
    return iou - (center_distance_sq / diagonal_sq + v * alpha)


def bboxes_iou_xyxy(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Plain IoU between ``xyxy`` boxes, used to ignore predictions near an unassigned GT."""
    top_left = torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])
    area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], 1)
    area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], 1)
    inside = (top_left < bottom_right).type(top_left.type()).prod(dim=2)
    area_intersection = torch.prod(bottom_right - top_left, 2) * inside
    area_union = area_a[:, None] + area_b - area_intersection
    return area_intersection / area_union


class Yolov4Loss(nn.Module):
    """Objectness + box + class loss for a Darknet-style multi-head YOLOv4 output.

    ``anchors`` is the full flat anchor list from the cfg (all heads' anchors,
    in pixels at ``image_size``); ``anchor_masks`` selects which of those each
    head is responsible for, in the same order the model emits heads.
    """

    def __init__(
        self,
        num_classes: int,
        anchors: list[tuple[float, float]],
        anchor_masks: list[list[int]],
        image_size: int = 416,
        ignore_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.image_size = image_size
        self.ignore_threshold = ignore_threshold
        self.num_anchors_per_head = len(anchor_masks[0])

    def _grid_anchors(
        self, stride: int, batch_size: int, feature_size: int, head_index: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        anchors_in_grid = [(w / stride, h / stride) for w, h in self.anchors]
        masked = torch.tensor(
            [anchors_in_grid[i] for i in self.anchor_masks[head_index]],
            dtype=torch.float32,
            device=device,
        )
        reference = torch.zeros(len(anchors_in_grid), 4, device=device)
        reference[:, 2:] = torch.tensor(anchors_in_grid, dtype=torch.float32, device=device)

        grid_x = torch.arange(feature_size, dtype=torch.float32, device=device).repeat(
            batch_size, self.num_anchors_per_head, feature_size, 1
        )
        grid_y = grid_x.permute(0, 1, 3, 2)
        anchor_w = (
            masked[:, 0].repeat(batch_size, feature_size, feature_size, 1).permute(0, 3, 1, 2)
        )
        anchor_h = (
            masked[:, 1].repeat(batch_size, feature_size, feature_size, 1).permute(0, 3, 1, 2)
        )
        return masked, reference, grid_x, grid_y, anchor_w, anchor_h

    def _build_target(
        self,
        pred: Tensor,
        labels: Tensor,
        head_index: int,
        stride: int,
        num_channels: int,
        masked_anchors: Tensor,
        reference_anchors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = pred.device
        batch_size, feature_size = pred.shape[0], pred.shape[2]

        obj_mask = torch.ones(batch_size, self.num_anchors_per_head, feature_size, feature_size).to(
            device
        )
        # tgt_mask excludes the objectness channel (index 4 of `target`/`output`) since
        # that channel is masked separately via `obj_mask`; it only marks which of the
        # box (0:4) and class (5:) channels belong to an assigned anchor.
        tgt_mask = torch.zeros(
            batch_size, self.num_anchors_per_head, feature_size, feature_size, num_channels - 1
        ).to(device)
        tgt_scale = torch.zeros(
            batch_size, self.num_anchors_per_head, feature_size, feature_size, 2
        ).to(device)
        target = torch.zeros(
            batch_size, self.num_anchors_per_head, feature_size, feature_size, num_channels
        ).to(device)

        num_labels_per_image = (labels.sum(dim=2) > 0).sum(dim=1)
        truth_x = (labels[:, :, 2] + labels[:, :, 0]) / (stride * 2)
        truth_y = (labels[:, :, 3] + labels[:, :, 1]) / (stride * 2)
        truth_w = (labels[:, :, 2] - labels[:, :, 0]) / stride
        truth_h = (labels[:, :, 3] - labels[:, :, 1]) / stride
        truth_i = truth_x.to(torch.int16).cpu().numpy()
        truth_j = truth_y.to(torch.int16).cpu().numpy()

        for batch_index in range(batch_size):
            n = int(num_labels_per_image[batch_index])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4, device=device)
            truth_box[:, 2] = truth_w[batch_index, :n]
            truth_box[:, 3] = truth_h[batch_index, :n]

            anchor_ious = bboxes_ciou(truth_box.cpu(), reference_anchors.cpu())
            best_anchor_overall = anchor_ious.argmax(dim=1)
            best_anchor = best_anchor_overall % self.num_anchors_per_head
            belongs_to_head = torch.zeros_like(best_anchor_overall, dtype=torch.bool)
            for mask_index in self.anchor_masks[head_index]:
                belongs_to_head |= best_anchor_overall == mask_index
            if not belongs_to_head.any():
                continue

            truth_box[:, 0] = truth_x[batch_index, :n]
            truth_box[:, 1] = truth_y[batch_index, :n]

            pred_ious = bboxes_iou_xyxy(pred[batch_index].reshape(-1, 4), truth_box)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_threshold).view(
                pred[batch_index].shape[:3]
            )
            obj_mask[batch_index] = ~pred_best_iou

            for label_index in range(n):
                if not belongs_to_head[label_index]:
                    continue
                grid_i = truth_i[batch_index, label_index]
                grid_j = truth_j[batch_index, label_index]
                anchor_index = int(best_anchor[label_index])
                obj_mask[batch_index, anchor_index, grid_j, grid_i] = 1
                tgt_mask[batch_index, anchor_index, grid_j, grid_i, :] = 1
                target[batch_index, anchor_index, grid_j, grid_i, 0] = truth_x[
                    batch_index, label_index
                ] - truth_x[batch_index, label_index].to(torch.int16).to(torch.float32)
                target[batch_index, anchor_index, grid_j, grid_i, 1] = truth_y[
                    batch_index, label_index
                ] - truth_y[batch_index, label_index].to(torch.int16).to(torch.float32)
                target[batch_index, anchor_index, grid_j, grid_i, 2] = torch.log(
                    truth_w[batch_index, label_index] / masked_anchors[anchor_index, 0] + 1e-16
                )
                target[batch_index, anchor_index, grid_j, grid_i, 3] = torch.log(
                    truth_h[batch_index, label_index] / masked_anchors[anchor_index, 1] + 1e-16
                )
                target[batch_index, anchor_index, grid_j, grid_i, 4] = 1
                class_id = int(labels[batch_index, label_index, 4])
                target[batch_index, anchor_index, grid_j, grid_i, 5 + class_id] = 1
                tgt_scale[batch_index, anchor_index, grid_j, grid_i, :] = torch.sqrt(
                    2
                    - truth_w[batch_index, label_index]
                    * truth_h[batch_index, label_index]
                    / feature_size
                    / feature_size
                )
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, head_outputs: list[Tensor], labels: Tensor) -> dict[str, Tensor]:
        """``head_outputs``: raw per-head conv output. ``labels``: padded
        ``(batch, max_objects, 5)`` tensor of ``[x1, y1, x2, y2, class_id]`` in
        pixel coordinates at ``self.image_size``, zero-padded rows ignored."""
        num_channels = 5 + self.num_classes
        loss_xy = loss_wh = loss_obj = loss_cls = 0.0
        for head_index, output in enumerate(head_outputs):
            batch_size, feature_size = output.shape[0], output.shape[2]
            device = output.device
            stride = self.image_size // feature_size
            masked_anchors, reference_anchors, grid_x, grid_y, anchor_w, anchor_h = (
                self._grid_anchors(stride, batch_size, feature_size, head_index, device)
            )

            output = output.view(
                batch_size, self.num_anchors_per_head, num_channels, feature_size, feature_size
            )
            output = output.permute(0, 1, 3, 4, 2)
            output = output.clone()
            output[..., np.r_[:2, 4:num_channels]] = torch.sigmoid(
                output[..., np.r_[:2, 4:num_channels]]
            )

            pred = output[..., :4].clone()
            pred[..., 0] += grid_x
            pred[..., 1] += grid_y
            pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
            pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h

            obj_mask, tgt_mask, tgt_scale, target = self._build_target(
                pred, labels, head_index, stride, num_channels, masked_anchors, reference_anchors
            )

            output[..., 4] = output[..., 4] * obj_mask
            output[..., np.r_[0:4, 5:num_channels]] = (
                output[..., np.r_[0:4, 5:num_channels]] * tgt_mask
            )
            output[..., 2:4] = output[..., 2:4] * tgt_scale

            target[..., 4] = target[..., 4] * obj_mask
            target[..., np.r_[0:4, 5:num_channels]] = (
                target[..., np.r_[0:4, 5:num_channels]] * tgt_mask
            )
            target[..., 2:4] = target[..., 2:4] * tgt_scale

            loss_xy += F.binary_cross_entropy(
                output[..., :2], target[..., :2], weight=tgt_scale * tgt_scale, reduction="sum"
            )
            loss_wh += F.mse_loss(output[..., 2:4], target[..., 2:4], reduction="sum") / 2
            loss_obj += F.binary_cross_entropy(output[..., 4], target[..., 4], reduction="sum")
            loss_cls += F.binary_cross_entropy(output[..., 5:], target[..., 5:], reduction="sum")

        return {"xy": loss_xy, "wh": loss_wh, "obj": loss_obj, "cls": loss_cls}
