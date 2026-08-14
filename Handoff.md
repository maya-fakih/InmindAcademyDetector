# Handoff (yolo26s-coco branch)

## Status
Not yet run under the new regime — a GPU-hour cap hit mid-transition,
resuming on a fresh Colab account with the same repo/branches (nothing
server-side changes; just remount Drive and reset the GH_TOKEN secret
on the new account).

## What changed and why
Original 20-epoch run (COCO-pretrained yolo26s -> LOCO, whole model
unfrozen from epoch 0, flat LR) converged to mAP@0.5 = 0.0274 -- behaving
like retraining from scratch, not fine-tuning. Diagnosed cause: a
randomly reinitialized classification head (only `cv3`/class-count-
dependent tensors fail to transfer -- confirmed from the Colab log,
"Transferred 696/708 items from pretrained weights") sends large, noisy
gradients back through the pretrained backbone before the head has
calibrated, dragging good weights off their pretrained initialization.

Correction to an earlier (wrong) claim in this file: previously said
the *entire* head reinitializes on class-count mismatch. It doesn't --
only the classification branch does; box-regression layers transfer
fine since their shape doesn't depend on class count.

Fix, following Ultralytics' own fine-tuning guidance:
- `freeze_backbone_epochs: 5`, `freeze_backbone_layers: 11` -- freeze
  layers 0-10 (backbone through the C2PSA block, per the printed model
  summary) for the first 5 epochs so only the head adapts initially.
- `warmup_epochs: 3` -- linear LR ramp 0 -> base LR.
- `lr_final_fraction: 0.01` -- cosine decay to 1% of base LR by the
  final epoch.
- `epochs: 50` for this run (up from 20).

Implemented in `train.py`: `compute_lr()` and `set_backbone_frozen()`.
Freeze mechanism verified live on Colab before trusting it for a full
run: `120/366 params frozen` when called on a real model instance --
confirms the `model.model.model` layer-indexing assumption was correct
(this couldn't be checked from the sandbox alone, no live model there).

## colab_runner.sh change
Previously auto-resumed from any `weights/last.ckpt` found on Drive.
That would've silently skipped the freeze/warmup window (keyed to
epochs 0-5) and built on backbone weights already perturbed by the old
run. Now requires an explicit argument:
- `bash scripts/colab_runner.sh` -- fresh run (default). Backs up any
  existing `weights/` dir with a timestamp instead of overwriting it.
- `bash scripts/colab_runner.sh resume` -- resumes from `last.ckpt`,
  errors if none exists.

**Run this one fresh, no resume argument** -- the old checkpoint was
trained under the flawed regime and shouldn't be built on.

## CI note
Caught late (should've checked from the first commit): `ruff format`
also validates Python fences inside markdown files, not just `.py`
files. Costed one extra fix commit on the roboflow branch. Now running
`ruff format --check .` and `ruff check .` on every branch before every
push, no exceptions. Can't run `pytest` from this sandbox (no torch
available) -- that layer of CI stays unverified by me locally; said so
explicitly rather than assuming it's fine.

## Papers / row-transfer note
Row-level class weight transfer (Kuen et al. ICCV 2019; arXiv
1912.06185) does NOT apply to this branch -- LOCO's 5 classes
(`forklift`, `pallet`, `pallet_truck`, `small_load_carrier`, `stillage`)
don't match any of COCO's 80 classes by name, so there's no valid row
mapping to transfer. That work is scoped to `yolo26-roboflow` instead,
where a real logistics-domain pretrain source exists. See that branch's
Handoff.md and ROBOFLOW.md.

## Left to do
- Run the 50-epoch fresh fine-tune (see command above) once GPU access
  is back.
- Target: mAP@0.5 >= 0.5. If freeze/warmup alone doesn't get there,
  next levers to check: batch_size (currently 4, very small/noisy --
  bump if a bigger Colab GPU is available), and whether `num_workers: 0`
  is still limiting throughput on the new account's GPU tier.
