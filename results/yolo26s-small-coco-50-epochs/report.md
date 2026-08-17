# yolo26s-small-coco-50-epochs

## Eval metrics (via eval.py, subsets 1/4 -- unseen warehouses)

- **mAP@0.5:** 0.2101
- **parameters:** 9,951,734 (9.95 M)
- **of which trainable:** 9,951,734
- **GFLOPs per image:** 22.453
- **median batch-1 latency:** 19.64 ms (cuda) -- machine-dependent, not graded

For reference, budget this must stay under (Faster R-CNN baseline):
18,950,729 params / 23.825 GFLOPs. Comfortably inside on both counts.

## Training curves

Not regenerated here -- `curves.png` requires running `results/generate_report.py`
against the actual `weights/` + full `train.log` in the Colab environment
where the run happened. This report was assembled from the `train.log` /
`eval.log` text pasted into chat, same as `yolo26n-coco-100-epochs`'s report.
Regenerate properly with `generate_report.py` next time a report is needed
for this run's `weights/`, which are still on Drive
(`InmindAcademyDetector-yolo26s-small-coco/runs/baseline/weights/`).

## Full run (val_mAP50 = held-out slice of train subsets 2/3/5, NOT the eval number above)

Architecture confirmed as YOLO26s (9,951,734 params, 22.8 GFLOPs) from the
actual training log, not assumed -- this run genuinely trained yolo26s.pt,
unlike yolo26s-coco's history of silently training nano (see that branch's
config.yaml comment).

| epoch | train loss | val mAP@0.5 (train-side) | lr | time (s) | note |
|---|---|---|---|---|---|
| 1 | 17.7329 | 0.0423 | 0.001667 | 260 | |
| 5 | 12.3413 | 0.1199 | 0.000900 | 284 | |
| 6 | 14.9526 | 0.0383 | 0.001200 | 314 | backbone unfrozen, val mAP dips as expected |
| 11 | 11.6037 | 0.1442 | 0.001482 | 318 | |
| 16 | 10.5783 | 0.2005 | 0.001377 | 309 | |
| 21 | 9.5891 | 0.2369 | 0.001190 | 315 | |
| 27 | 8.9638 | 0.3118 | 0.000892 | 312 | |
| 33 | 8.4420 | 0.3424 | 0.000570 | 307 | |
| 39 | 8.2353 | 0.3980 | 0.000283 | 317 | |
| 44 | 7.9348 | 0.3986 | 0.000110 | 309 | |
| 47 | 7.9049 | 0.4075 | 0.000046 | 309 | |
| 50 | 7.7571 | 0.4093 | 0.000017 | 318 | final, best checkpoint |

Val-side mAP (0.4093) vs. held-out eval mAP (0.2101): same train/val-vs-test
distribution gap documented for nano (val drawn from the same subsets as
train -- sub2/3/5 -- test is entirely different subsets sub1/4). Not a bug;
expected given the split. See `dataset.py`'s `DEVELOPMENT_FILES`/`TEST_FILES`.

## Comparison to nano (yolo26s-coco, 100 epochs)

| run | params | GFLOPs | test mAP@0.5 |
|---|---|---|---|
| yolo26n-coco-100-epochs | 2.51 M | 5.75 | 0.1559 |
| yolo26s-small-coco-50-epochs | 9.95 M | 22.45 | 0.2101 |

Real improvement from the larger architecture, though not yet at the
30-40% target. Extending to 100 epochs (this branch's `config.yaml` now
set to `epochs: 100`) to see how much of that gap closes with more training
before trying further architecture/augmentation changes.
