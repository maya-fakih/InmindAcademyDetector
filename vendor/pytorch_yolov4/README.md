# Vendored: Tianxiaomo/pytorch-YOLOv4

Source: https://github.com/Tianxiaomo/pytorch-YOLOv4 (commit at time of vendoring: master, Aug 2026)
License: Apache-2.0 (see LICENSE in this directory)

Files vendored verbatim (only import paths rewritten to `vendor.pytorch_yolov4.*`):
- `darknet2pytorch.py` — parses a Darknet `.cfg` into an equivalent `nn.Module` graph
  (`Darknet` class), and loads official Darknet `.weights` binaries via `load_weights()`.
- `yolo_layer.py` — the YOLO detection head layer used by the Darknet graph.
- `config.py`, `torch_utils.py`, `utils.py`, `region_loss.py` — supporting utilities
  imported by the two files above (region_loss.py is unused by the tiny variant but
  imported transitively; kept for import-compatibility).
- `yolov4-tiny.cfg` — official AlexeyAB Darknet config, defines the tiny architecture
  (2 YOLO scales, 6 anchors, 416x416 input) that `darknet2pytorch.Darknet` builds from.
  **Edited from upstream**: `classes=80`→`classes=5` and `filters=255`→`filters=30`
  (filters = 3 anchors × (5 + num_classes)) at both YOLO output heads, for LOCO's 5
  classes. Anchor boxes are still COCO's, not re-clustered for LOCO's box-size
  distribution — a real, known gap, not an oversight.

## Why vendor instead of pip-install

This is not a published PyPI package, and we need the exact `Darknet` cfg-parsing
behavior (not a reimplementation) so the model architecture matches the official
Darknet YOLOv4-tiny graph exactly, including the officially-published `.cfg`.

## What's NOT reused from this repo

`train.py`'s `Yolo_loss` class in the upstream repo is hardcoded for full YOLOv4's
3-scale / 608px / 9-anchor setup and does not correctly support the tiny variant's
2-scale / 6-anchor architecture. We do not vendor it. Instead, `models/yolov4tiny_wrapper.py`
implements a loss driven dynamically by the anchors/masks/strides that
`darknet2pytorch.Darknet` actually parses from `yolov4-tiny.cfg`, adapted from the same
IOU-assignment / regression / BCE approach as upstream's `Yolo_loss`, generalized to work
for either scale count instead of hardcoded to 3.

## Fidelity check

Tianxiaomo's port publishes a validated benchmark against official Darknet weights on
COCO val2017: 0.704 AP50 (port) vs 0.710 AP50 (official Darknet) — see upstream README.
This is why the port was chosen over a from-scratch reimplementation: it's a checked,
not assumed, match to the reference implementation.
