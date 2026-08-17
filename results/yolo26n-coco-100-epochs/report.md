# yolo26n-coco-100-epochs

## Eval metrics (via eval.py, subsets 1/4 -- unseen warehouses)

- **mAP@0.5:** 0.1559
- **parameters:** 2,505,750 (2.51 M)
- **of which trainable:** 2,505,750
- **GFLOPs per image:** 5.749
- **median batch-1 latency:** 16.26 ms (cuda) -- machine-dependent, not graded

## Training curves

Not regenerated here -- `curves.png` requires running `generate_report.py`
against the actual `weights/` + full `train.log` in the Colab environment
where the run happened. This report was assembled from the `train.log` /
`eval.log` text the run produced, uploaded after the fact; only the tail of
`train.log` (epochs 94-100) survived on the uploading side, so a full
100-epoch loss/mAP curve can't be reconstructed here. Regenerate properly
with `results/generate_report.py` next time a report is needed for a run
whose `weights/` are still on Drive.

## Last 7 epochs (val_mAP50 = held-out slice of train subsets 2/3/5, NOT the eval number above)

| epoch | train loss | val mAP@0.5 (train-side) | lr | time (s) |
|---|---|---|---|---|
| 94 | 8.1887 | 0.3472 | 0.000113 | 1872 |
| 95 | 8.0743 | 0.3466 | 0.000097 | 226 |
| 96 | 8.2421 | 0.3437 | 0.000082 | 226 |
| 97 | 8.0754 | 0.3444 | 0.000071 | 221 |
| 98 | 8.3031 | 0.3410 | 0.000062 | 210 |
| 99 | 8.1363 | 0.3464 | 0.000055 | 208 |
| 100 | 8.2338 | 0.3426 | 0.000051 | 211 |

## Notes

- **val/test gap:** train-side val_mAP50 plateaus ~0.34-0.35; the real
  eval (subsets 1/4, disjoint warehouses) is 0.1559 -- see `Handoff.md`
  for the diagnosis (warehouse-specific visual shortcuts, not a bug).
  This is the number the augmentation + warm-restart changes on this
  branch are meant to move, not the train-side val_mAP50.
- **Loss plateau:** loss is flat (~8.0-8.3) across epochs 94-100 with lr
  already decayed to ~5-11e-5 -- consistent with a cosine schedule near
  its end, not with active learning. This run is a base to resume from,
  not a converged final result.
- Epoch 94 took 1872s vs ~210-226s for neighboring epochs -- almost
  certainly a Colab disconnect/reconnect or Drive I/O stall mid-epoch,
  not a real compute regression.
