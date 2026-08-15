# Handoff (yolov4t-loco branch)

## Status: ready to launch a training run on Colab, one prerequisite step first

The previous Handoff.md in this branch was stale -- it described the branch
as "not yet pushed, start from scratch" when in fact 4 more commits had
already landed (vendoring, train.py wiring, pretrained backbone + anchor
tooling, colab script). This doc replaces it with what's actually true as of
commit `c456fea`.

## What's proven, with real forward/backward runs (not guessed)
- `Darknet('vendor/pytorch_yolov4/yolov4-tiny.cfg')` builds correctly:
  5,883,356 params (matches expected size once heads are resized from
  COCO's 80 classes to LOCO's 5 -- filters=30 not 255, confirmed in the
  cfg itself).
- `models/yolov4_loss.py`'s `Yolov4Loss` -- previously never run at all,
  despite already being wired into `Yolov4TinyWrapper` -- now has a real
  test (`tests/test_yolov4_loss.py`) that: runs a full forward pass on
  synthetic images/labels, confirms both head shapes ((2,30,13,13) and
  (2,30,26,26)), confirms the total loss is finite, calls `.backward()`
  and confirms every one of the model's 61 param tensors receives a
  gradient (not just some), and confirms a batch with zero labeled
  objects doesn't crash. All passing.
- `Yolov4TinyWrapper` (in `models/yolo_wrapper.py`) matches `YoloWrapper`'s
  interface: `forward(images, targets) -> loss dict` in training mode,
  `forward(images) -> list[{boxes, scores, labels}]` in eval mode.
  `train.py` dispatches on `model.architecture` in config.yaml, already
  wired (`create_yolo_model(..., architecture=...)`).

## Two real bugs found and fixed this session
1. **`scripts/colab_runner.sh`** accepted a branch name as `$2` but then
   immediately overwrote it with a hardcoded `BRANCH="yolo26s-coco"` two
   lines later -- passing any other branch name silently did nothing. This
   is why the training log you had showed it running yolo26s-coco even
   after intending to run this branch. Fixed: the hardcoded line is gone,
   `${2:-yolo26s-coco}` is respected.
2. **`eval.py` / `predict.py`** called `create_yolo_model(dataset.num_classes)`
   with no `architecture` argument, so they always defaulted to
   `"ultralytics"` regardless of `config.yaml` -- would have tried to
   build/fetch a yolo26n checkpoint and then failed (or silently
   mismatched) on `load_state_dict()` instead of loading real trained
   yolov4-tiny weights. Fixed: both now read `model.architecture` from
   config, same as train.py.

## One prerequisite left before a real training run
`vendor/pytorch_yolov4/yolov4-tiny.cfg` still has the **stock COCO-pretrained
anchors** (`10,14, 23,27, 37,58, 81,82, 135,169, 344,319`), not LOCO-specific
ones. `scripts/cluster_anchors.py` exists and is written to re-cluster them
on LOCO's actual box shapes (subsets 2/3/5 only, per the project's own
holdout rule), but it was never run against the real dataset -- the sandbox
that wrote it had no network access to LOCO. This is stated honestly in that
commit's message, not hidden.

**Run this on Colab before training, and commit the result:**
```
python scripts/cluster_anchors.py --raw-dir <path-to-loco> --write
git add vendor/pytorch_yolov4/yolov4-tiny.cfg
git commit -m "chore: re-cluster yolov4-tiny anchors on real LOCO box shapes"
git push origin yolov4t-loco
```
Training on stock COCO anchors would still run and probably still learn
something, but mAP would likely be measurably worse than with anchors sized
for LOCO's actual object shapes (small warehouse objects, different aspect
ratios than COCO). Not a hard blocker, but don't skip it if you have a
minute -- it's cheap and the whole point of doing this branch is a fair,
deliberate comparison.

## Launch command (once anchors are re-clustered)
```
!bash scripts/colab_runner.sh fresh yolov4t-loco
```
Note the explicit `yolov4t-loco` -- this is now required to hit the right
branch (see bug #1 above); omitting it defaults to `yolo26s-coco`.

`config.yaml` on this branch: `epochs: 100` (matches the yolo26n run for a
fair comparison), `augment.enabled: true` (background-swap + hflip, same
augmentation pipeline as yolo26s-coco once that branch's run finishes and
its result is known -- if background-swap turns out to hurt there, consider
whether to also disable it here before committing to a full 100-epoch run).

## Left to do after this run finishes
- `eval.py` on subsets 1/4 (colab_runner.sh does this automatically at the
  end of the run).
- Write `BRANCH.md`: papers (Savas & Hinckeldeyn arXiv 2209.13499 -- 49.98%
  YOLOv4-tiny under an easier split than ours; original LOCO paper 22.1%
  YOLOv4-tiny under a split closer to ours), GFLOPs/params comparison table
  (YOLOv4-tiny: 6.9 GFLOPs / 6.06M full-COCO-class params vs baseline
  Faster R-CNN: 23.825 GFLOPs / 18.95M params -- YOLOv4-tiny is lighter on
  both axes even before considering our 5-class head is smaller still).
- Honest caveat to keep stating in BRANCH.md: this is architecturally
  equivalent to YOLOv4-tiny (same .cfg, same official pretrained backbone
  loadable), not literally Darknet-C's codebase. The loss was adapted from
  Tianxiaomo/pytorch-YOLOv4's `Yolo_loss` (generalized from its hardcoded
  3-head/608px/9-anchor assumption to this project's 2-head/416px/6-anchor
  one) -- now numerically verified via `tests/test_yolov4_loss.py`, not
  just copy-pasted and trusted.
