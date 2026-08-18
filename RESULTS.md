# Results

Full reports (curves, images, notes) live per-branch under `results/`.
This is just the running scoreboard.

| run | branch | mAP@0.5 (eval) | mAP@0.5 (best train-val) | params | GFLOPs | notes |
|---|---|---|---|---|---|---|
| yolo26n-coco-50-epochs | yolo26s-coco | 0.112 | 0.305 (epoch 33/50) | 2.51 M | 5.75 | COCO-pretrained, no head transfer, no leak check yet |
| yolo26n-coco-100-epochs | yolo26s-coco | 0.156 | 0.349 (best.ckpt tracking) | 2.51 M | 5.75 | Large val (subsets 2/3/5 held-out)/test (subsets 1/4) gap identified -- warehouse-shortcut overfitting, not a bug (see Handoff.md). Augmentation + warm-restart fix in progress. |

Target: 0.70 mAP@0.5.

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
