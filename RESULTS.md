# Results

Full reports (curves, images, notes) live per-branch under `results/`.
This is just the running scoreboard.

| run | branch | mAP@0.5 (eval) | mAP@0.5 (best train-val) | params | GFLOPs | notes |
|---|---|---|---|---|---|---|
| yolo26n-coco-50-epochs | yolo26s-coco | 0.112 | 0.305 (epoch 33/50) | 2.51 M | 5.75 | COCO-pretrained, no head transfer, no leak check yet |
| yolo26n-coco-100-epochs | yolo26s-coco | 0.156 | 0.349 (best.ckpt tracking) | 2.51 M | 5.75 | Large val (subsets 2/3/5 held-out)/test (subsets 1/4) gap identified -- warehouse-shortcut overfitting, not a bug (see Handoff.md). Augmentation + warm-restart fix in progress. |

Target: 0.70 mAP@0.5.

## frcnn-amir-recipe: unseen-warehouse validation split

This branch reproduces (from scratch, own code) the split methodology used
on Amir's `exp/03-recipe` branch
([amiroo-star/inmind-detector](https://github.com/amiroo-star/inmind-detector)):
train on whole subsets 2 and 5, validate on the whole of subset 3 as an
unseen warehouse the model never trains on, test on subsets 1/4 unchanged.
This is different from both the assignment template's split (1-in-5 images
per development subset) and this repo's own `frcnn-mobilenetv3-augment`
demo split (biased-but-in-domain validation). See `dataset.py` on this
branch for the implementation and full rationale.

**Comparison point -- reported by Amir, not reproduced here:** Amir's README
and commit history report ~26% test-set accuracy after 20 epochs on his
`exp/03-recipe` branch. His repo has no committed `history.json` or run logs
to verify that number against, so it's recorded here purely as a reference
target, not a validated result. This branch's own `history.json`
(train_loss, val_mAP50, lr, periodic test_mAP50) is written every epoch to
`output_dir/history.json` once a run completes, so future entries in the
table above from this branch will be reproducible from a committed log,
unlike Amir's figure.

Also notable: Amir's branch has no learning-rate curve or schedule logging
at all (flat LR, no warmup/decay, nothing recorded per epoch). This branch
uses this repo's existing cosine warmup/decay schedule and logs `lr` into
`history.json` every epoch, so `results/generate_report.py`'s LR-vs-epoch
panel is populated here even though it wouldn't be if the run matched
Amir's setup literally.

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
