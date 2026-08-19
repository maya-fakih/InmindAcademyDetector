# Results

Full reports (curves, images, notes) live per-branch under `results/`.
This is just the running scoreboard.

| run | branch | mAP@0.5 (eval) | mAP@0.5 (best train-val) | params | GFLOPs | notes |
|---|---|---|---|---|---|---|
| yolo26n-coco-50-epochs | yolo26s-coco | 0.112 | 0.305 (epoch 33/50) | 2.51 M | 5.75 | COCO-pretrained, no head transfer, no leak check yet |
| yolo26n-coco-100-epochs | yolo26s-coco | 0.156 | 0.349 (best.ckpt tracking) | 2.51 M | 5.75 | Large val (subsets 2/3/5 held-out)/test (subsets 1/4) gap identified -- warehouse-shortcut overfitting, not a bug (see Handoff.md). Augmentation + warm-restart fix in progress. |

Target: 0.70 mAP@0.5.

## frcnn-amir-recipe: mirrors Amir's exp/03-recipe branch (teammate collaboration)

Disclosed collaboration: Amir is a teammate on this project. We agreed I'd
lead with yolo26s-small-coco (best results on my side so far) and he'd try
his own recipe on Faster R-CNN; this branch reproduces his approach for a
side-by-side comparison, to be discussed openly in the presentation. Not an
independent methodology -- credited throughout as ported from
[amiroo-star/inmind-detector, branch exp/03-recipe](https://github.com/amiroo-star/inmind-detector/tree/exp/03-recipe).

Mirrored end to end:
- **Split** (`dataset.py`): train on whole subsets 2+5, validate on the
  whole of subset 3 as an unseen-warehouse holdout, test on 1+4 unchanged.
- **Model** (`models/baseline.py`, `config.yaml` `model.name`):
  `fasterrcnn_mobilenet_v3_large_320_fpn` (320px input, 6.8 GFLOPs) instead
  of this repo's stock 42.3-GFLOPs baseline.
- **Augmentation** (`dataset.py`'s `_augment`): horizontal flip + brightness/
  contrast/saturation jitter, train split only -- his exact function, not
  this repo's separate background-swap/scale-jitter pipeline.
- **LR schedule** (`train.py`): warmup + cosine decay stepped per batch
  (`schedule: cosine`, `warmup_fraction: 0.05`), matching his schedule
  shape rather than this repo's old per-epoch schedule.

Added on top of his pipeline, not present on his branch:
- `test_eval_every: 10` -- periodic held-out subsets-1/4 monitoring during
  training, logged as `[TEST subsets 1/4]`, never used for `best.pt`
  selection.
- `history.json` -- per-epoch train_loss/val_mAP50/lr/test_mAP50, written
  every epoch (survives a Colab disconnect), which is what
  `results/generate_report.py`'s LR-vs-epoch panel plots. Amir's branch
  also writes `lr` into its own `history.json` every epoch, but has no
  report generator to plot it -- this branch reuses this repo's existing
  `results/generate_report.py`, so the same numbers actually get charted.

**Comparison point -- reported by Amir, not reproduced here:** his README/
commits report ~26% test-set accuracy after a 20-epoch run. His current
`config.yaml` is set to 40 epochs, and his repo has no committed
`history.json` or run log for the 20-epoch run the 26% figure came from, so
it's recorded here as a reference target pending an actual comparable run
on this branch, not a validated result.

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
