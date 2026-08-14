# Handoff (yolo26s-coco branch)

## Latest: background-swap augmentation + 100-epoch convergence run

### Why
The freeze/warmup 50-epoch run (see below) hit val_mAP50 = 0.3048 but only
0.1120 on the real test subsets (1/4) -- a bad val/test gap. The LOCO
dataset's own paper explains the likely mechanism directly: train (subsets
2/3/5) and test (subsets 1/4) are **different physical warehouses** by
design --

> "we use subsets to perform the training and evaluation split, since each
> of them corresponds to one particular logistics environment. This
> guarantees that the training and evaluation sets are disjoint from each
> other, which implicitly shifts the focus in machine-learning applications
> towards generalizable models, since certain conditions (i.e. scene,
> lighting, color) may have not been encountered in the training set."
> -- Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J. "LOCO: Logistics
> Objects in Context." IEEE ICMLA 2020.
> https://mediatum.ub.tum.de/doc/1578845/ndh526owvo3yhfqcaw2qypl5g.201007-loco-ieee-compressed.pdf

That's the dataset's own stated design goal, not a separate paper's
critique -- if the model is keying off background/scene context (warehouse
lighting, floor color, rack layout) rather than the objects themselves,
performance should drop exactly this way when the test warehouses are ones
it's never seen. I did not find a paper making this exact background-bias
claim about LOCO specifically beyond the original paper's own framing above;
said so rather than inventing a stronger citation.

### What changed
- `augmentation.py` (new) -- `random_horizontal_flip` and
  `random_background_swap`. The latter keeps an image's labeled objects
  (exact pixels, so boxes stay valid) but replaces everything outside those
  boxes with a different training image's background, directly attacking
  the background-context shortcut. Verified with synthetic array tests:
  box-remap math for hflip, and that the object region survives a swap
  pixel-for-pixel while the rest becomes the donor's content.
  **Known limitation, stated not hidden**: the donor's own labeled objects
  aren't masked out of the donor background, so a donor object can bleed
  into the composited image. Worth revisiting if this augmentation helps
  overall but seems to add label noise.
- `dataset.py` -- `LocoDataset` takes `background_swap_prob`/`hflip_prob`,
  applied in `__getitem__` before tensor conversion. **Validation and test
  splits force these to 0 regardless of what's passed in** (checked in
  `__init__`) -- `best.pt` selection must stay on clean, real val images,
  not augmented ones, or "best" stops meaning what it should.
- `train.py` -- reads `augment.enabled`/`background_swap_prob`/`hflip_prob`
  from config, passed only to the train split.
- `config.yaml` -- `augment.enabled: true`, both probs 0.5; `epochs: 100`
  (up from 50) to see whether mAP keeps climbing or plateaus/overfits by
  epoch 100 -- the freeze/warmup run was still rising at epoch 50, hadn't
  converged.

### Verification performed here (no live Colab/GPU in this sandbox)
- `random_horizontal_flip`/`random_background_swap` unit-tested on
  synthetic numpy arrays (box remap correctness, object-region pixel
  preservation under swap).
- `LocoDataset` end-to-end tested against a synthetic mini COCO-format
  dataset: augmented `__getitem__` returns correct shapes, and validation
  split's aug probs are confirmed forced to 0 even when 1.0 is passed in.
- `ruff format --check .` and `ruff check .` both pass.
- **Not verified here**: an actual multi-epoch training run with
  augmentation on, since that needs the real LOCO images + a GPU. First
  real signal is whatever epoch 1 looks like on Colab -- if loss/mAP
  behave wildly differently from the un-augmented runs, stop and diagnose
  before trusting a full 100-epoch run to it.

### Left to do
- Run this on Colab: `bash scripts/colab_runner.sh` (fresh run, no
  `resume` arg -- old checkpoints were trained without augmentation).
- Watch val_mAP50 through ~epoch 50 vs the un-augmented run's 0.3048 at the
  same point -- if augmentation is actively hurting (worse, not better),
  turn off `augment.enabled` rather than pushing through 100 epochs on a
  regression.
- At 100 epochs: check whether mAP plateaus (real convergence signal) vs.
  still climbing (would suggest even more epochs might help) vs. dropping
  late (overfitting despite augmentation).
- If the val/test gap doesn't close, background bias may not be the whole
  story -- also worth checking image resolution/scale mismatch between
  subsets, and whether class distribution genuinely differs across
  warehouses beyond what `compute_balanced_split` already corrects for
  (that function only balances within 2/3/5, not against 1/4's actual
  distribution, which it can't see by design).

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
