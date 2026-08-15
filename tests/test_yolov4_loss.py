"""Forward/backward sanity checks for Yolov4Loss against the real vendored Darknet model.

These are the checks the loss module was never actually run through before being
wired into training (see Handoff.md) -- finite loss, real gradients into every
param, and correct handling of a batch with zero labeled objects.
"""

from __future__ import annotations

import torch

from models.yolov4_loss import Yolov4Loss
from vendor.pytorch_yolov4.darknet2pytorch import Darknet

CFG_PATH = "vendor/pytorch_yolov4/yolov4-tiny.cfg"


def _build_model_and_loss() -> tuple[Darknet, Yolov4Loss]:
    model = Darknet(CFG_PATH)
    model.train()
    yolo_blocks = [block for block in model.blocks if block["type"] == "yolo"]
    anchor_values = [float(value) for value in yolo_blocks[0]["anchors"].split(",")]
    anchors = list(zip(anchor_values[0::2], anchor_values[1::2], strict=True))
    anchor_masks = [[int(index) for index in block["mask"].split(",")] for block in yolo_blocks]
    loss_fn = Yolov4Loss(num_classes=5, anchors=anchors, anchor_masks=anchor_masks, image_size=416)
    return model, loss_fn


def test_forward_backward_produces_finite_loss_and_real_gradients() -> None:
    model, loss_fn = _build_model_and_loss()

    images = torch.rand(2, 3, 416, 416)
    labels = torch.zeros(2, 3, 5)
    labels[0, 0] = torch.tensor([50, 60, 150, 200, 1])
    labels[0, 1] = torch.tensor([200, 210, 300, 320, 3])
    labels[1, 0] = torch.tensor([10, 10, 100, 100, 0])

    head_outputs = model(images)
    assert len(head_outputs) == 2, "yolov4-tiny should emit exactly 2 detection heads"
    assert head_outputs[0].shape == (2, 30, 13, 13)
    assert head_outputs[1].shape == (2, 30, 26, 26)

    losses = loss_fn(head_outputs, labels)
    assert set(losses) == {"xy", "wh", "obj", "cls"}
    total = sum(losses.values())
    assert torch.isfinite(total)

    total.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "every param should receive a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.norm() > 0 for g in grads), "gradients should not all be exactly zero"


def test_batch_with_no_labeled_objects_does_not_crash() -> None:
    model, loss_fn = _build_model_and_loss()

    images = torch.rand(1, 3, 416, 416)
    labels = torch.zeros(1, 1, 5)  # no real objects, fully zero-padded

    head_outputs = model(images)
    losses = loss_fn(head_outputs, labels)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
