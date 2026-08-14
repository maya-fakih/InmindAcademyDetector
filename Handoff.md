# Handoff (yolo26-roboflow branch)

## Where we are
Base fixes ported from yolo26s-coco: `num_workers: 0` (2 caused OOM-kills
on Colab free tier), config-driven checkpoint (`checkpoint: yolo26n.pt`),
per-epoch loss breakdown logging, freeze/warmup/cosine-LR (`config.yaml`
already has `warmup_epochs`/`freeze_backbone_epochs`/`freeze_backbone_layers`/
`lr_final_fraction` set to yolo26s-coco's tuned values), explicit-resume-flag
safety in `colab_runner.sh`, and the background-swap/hflip augmentation
machinery (`augmentation.py`, wired into `LocoDataset` and `train.py` --
same train-only/validation-forced-off safety as yolo26s-coco). `augment`
is not enabled in this branch's `config.yaml` yet (no `augment:` section) --
add one when actually training LOCO fine-tuning here; it's dormant until
then. Ported via cherry-pick, one manual merge conflict (checkpoint-selection
vs. resume/freeze logic in `train.py`, both kept) -- verified: nano
checkpoint still builds correctly (2.5M params) through config, `ruff
format`/`ruff check` both pass.

No Roboflow dataset ingestion or training run started on this branch yet --
Roboflow class distribution has been checked (real counts, screenshot):
forklift 11,295, pallet 83,407, pallet_truck 9,264, small_load_carrier
24,853, stillage 7,022 -- ~12:1 max:min ratio, much better than LOCO alone's
144:1 forklift problem.

**Next real gap**: `train.py`/`LocoDataset` only understand LOCO's COCO-JSON
format (`loco-sub*-v1-*.json`) -- a Roboflow YOLO-format export (images +
per-image `.txt` label files) needs either a new dataset loader class or a
conversion step before any of this branch's training machinery can touch
it. Not built yet. See `ROBOFLOW.md` step 5 for the originally-suggested
plain-Ultralytics-CLI alternative, which sidesteps this but doesn't get the
freeze/warmup/logging benefits just ported here -- worth deciding which
path before writing more code.

See `ROBOFLOW.md` for the full dataset-merge plan and steps in order.

## What's new on this branch
- `scripts/dedup_check.py` -- perceptual-hash leak check between the
  subset-1/4 holdout and any external pretraining candidates. Run before
  merging any Roboflow export. Best-effort: catches near-identical images,
  not distinct photos of the same scene -- residual leak risk is real and
  should be stated as a limitation in the final report, not claimed solved.
- `models/class_weight_transfer.py` -- row-level classification-head
  weight transfer for classes present in both a pretrained checkpoint and
  LOCO's 5 classes. Not wired into `yolo_wrapper.py` yet. Heuristic-based
  (matches tensors by `cv3` in the key name + leading dim == class count)
  since there's no live model instance here to verify exact YOLO26 module
  paths -- **must** be verified with `list_candidate_tensors()` against a
  real checkpoint before trusting it in an actual training run.

## Why full-tensor `load()` skips the head on nc mismatch
`DetectionModel.load()` copies whole tensors only when shapes match
exactly. Backbone/neck shapes don't depend on class count so they always
transfer; head tensors do (`nc`-dependent), so on any class-count mismatch
the entire head -- including rows for classes that exist in both models --
gets randomly reinitialized. That's the gap `class_weight_transfer.py`
is meant to close, but only where a real semantic class match exists.

## LOCO's 5 classes and the papers behind row transfer
`forklift`, `pallet`, `pallet_truck`, `small_load_carrier`, `stillage`.
None of these match COCO's 80 classes, so row transfer is irrelevant for
the yolo26s-coco branch -- COCO pretraining there stays whole-head-reinit,
correctly. Row transfer only makes sense against a logistics-domain
pretrain source where real class-name overlap exists, i.e. the Roboflow
merge described in ROBOFLOW.md.

Papers:
1. Kuen et al., "Scaling Object Detection by Transferring Classification
   Weights," ICCV 2019. https://openaccess.thecvf.com/content_ICCV_2019/papers/Kuen_Scaling_Object_Detection_by_Transferring_Classification_Weights_ICCV_2019_paper.pdf
2. "Learning Effective Visual Relationship Detector on 1 GPU," arXiv
   1912.06185. https://arxiv.org/pdf/1912.06185

## Rules we're respecting
- Train/fine-tune only on LOCO subsets 2, 3, 5. Subsets 1+4 reserved for
  final eval only -- this is the whole reason dedup_check.py exists.
- Pretrained models allowed only when fine-tuned (per README) -- the
  Roboflow pretrain-then-LOCO-finetune pipeline satisfies this.
- Checkpoints/logs stay out of git, Drive only.

## Left to do
- Finish the Roboflow merge + relabel (manual, in progress by Maya).
- Find a real source for the `stillage` class (flagged in ROBOFLOW.md).
- Verify `class_weight_transfer.py` against a real pretrained checkpoint,
  then wire it into `yolo_wrapper.py` if it checks out.
- Run `dedup_check.py` on the final merged export before any training.
- Actually train yolo26n on the merged set, then fine-tune on LOCO.
- Port the freeze/warmup/cosine-LR fix from `yolo26s-coco` (commits
  cc15ea7, cb63f31, 1246b08, 3b6c641) once that branch's run confirms
  the recipe works -- same diagnosis applies here: an unfrozen head at
  flat LR from epoch 0 drags pretrained weights around regardless of
  which checkpoint they came from.

## Update: paused on GPU-hour cap
Hit a Colab GPU quota while iterating on `yolo26s-coco`. Resuming on a
fresh Colab account -- nothing server-side changes here, this branch's
5 commits are already pushed and waiting. Note for whoever continues:
CI (`ruff format --check .`) also validates Python code fences inside
markdown files, not just `.py` -- caught and fixed once already in
ROBOFLOW.md (commit f435ca2); worth double-checking any new `.md` edits
the same way before pushing.
