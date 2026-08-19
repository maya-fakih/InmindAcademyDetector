# Results

Full reports (curves, images, notes) live per-branch under `results/`.
This is just the running scoreboard.

| run | branch | mAP@0.5 (eval) | mAP@0.5 (best train-val) | params | GFLOPs | notes |
|---|---|---|---|---|---|---|
| yolo26n-coco-50-epochs | yolo26s-coco | 0.112 | 0.305 (epoch 33/50) | 2.51 M | 5.75 | COCO-pretrained, no head transfer, no leak check yet |
| yolo26n-coco-100-epochs | yolo26s-coco | 0.156 | 0.349 (best.ckpt tracking) | 2.51 M | 5.75 | Large val (subsets 2/3/5 held-out)/test (subsets 1/4) gap identified -- warehouse-shortcut overfitting, not a bug (see Handoff.md). Augmentation + warm-restart fix in progress. |

Target: 0.70 mAP@0.5.

## frcnn-amir-recipe: Amir's actual recipe + a properly-fitted validation split

Disclosed collaboration: Amir is a teammate on this project. This branch
reproduces his winning architecture exactly (verified against his real
`exp/03-recipe` code and `EXPERIMENTS.md`, not assumed), but deliberately
does **not** copy his validation split -- see below for why, and why it
also doesn't reuse `frcnn-mobilenetv3-augment`'s demo split either.

**Model** (`models/baseline.py`, `config.yaml` `model:`): `slim_mobilenet_fpn`,
stages (12, 16), fpn_channels 128, representation_size 512, 800px input --
Amir's "slimB", 7,051,673 params / 19.238 GFLOPs, locally verified against
his exact number. Not `fasterrcnn_mobilenet_v3_large_320_fpn` (an earlier,
wrong guess in this file, still visible in git history) and not any
ThunderNet/CEM/SAM head -- his repo has none; grepped his full branch to confirm.

**Validation split** (`dataset.py`): proportional-to-test, not whole-subset.
Amir's own split (subsets 2+5 train, whole subset 3 val) validates on a
different warehouse *and* a different class mix than test (subsets 1/4) --
he's flagged this himself as making his own number optimistic. Copying it
would inherit that bias. `frcnn-mobilenetv3-augment`'s split fixes the class-mix
problem but does it by reading subsets 1/4 at runtime on every dataset init --
a real coupling to test we'd rather not have, even with zero pixels/labels
crossing over. This branch instead: pools subsets 2/3/5 for both train and
validation (never opens 1/4), and picks which pool images go to validation
via a greedy sampler targeting a **fixed constant** -- test's real per-category
distribution, computed once offline from the actual annotation JSONs and
hardcoded in `dataset.py`, not read at runtime. Achieved vs. target (test):

| category | test (target) | old whole-subset-3 | this split |
|---|---|---|---|
| pallet | 81.09% | 88.06% | 80.88% |
| small_load_carrier | 13.82% | 8.57% | 12.93% |
| stillage | 3.11% | 1.51% | 3.46% |
| pallet_truck | 1.79% | 1.16% | 2.03% |
| forklift | 0.19% | 0.71% | 0.70% |

**Augmentation / LR schedule**: same as Amir's (horizontal flip + photometric
jitter, warmup + cosine decay), already correct on this branch.

**Target**: Amir's own repeated runs of this exact recipe landed 0.2849-0.2929
test mAP@0.5 (run-to-run variance, not a single number -- see his
`EXPERIMENTS.md`). This branch's 30-epoch run should land in the same range;
its validation-based checkpoint selection may track test slightly better
since validation now resembles test's class mix, but that's not guaranteed
to move the final number much on its own.



## ⚠️ frcnn-mobilenetv3-augment: validation split is intentionally biased (demo)

This branch's `dataset.py` selects validation images by similarity to the
TEST subsets' (1/4) category distribution, instead of a distribution-neutral
split — see the comment block at the top of `dataset.py`. This is a
deliberate demonstration that **split composition alone inflates reported
val mAP**, without any pixels/labels from subsets 1/4 touching training.
`best.ckpt` on this branch is selected against that biased validation set, so
its val mAP is not comparable to the other branches' val mAP, and should not
be read as this architecture's real performance. Compare against `eval.py`
on subsets 1/4 (test) for the honest number, and expect a visible gap between
the two — that gap *is* the point being demonstrated.
