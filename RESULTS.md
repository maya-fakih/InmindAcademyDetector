# Results

Full reports (curves, images, notes) live per-branch under `results/`.
This is just the running scoreboard.

| run | branch | mAP@0.5 (eval) | mAP@0.5 (best train-val) | params | GFLOPs | notes |
|---|---|---|---|---|---|---|
| yolo26n-coco-50-epochs | yolo26s-coco | 0.112 | 0.305 (epoch 33/50) | 2.51 M | 5.75 | COCO-pretrained, no head transfer, no leak check yet |
| yolo26n-coco-100-epochs | yolo26s-coco | 0.156 | 0.349 (best.ckpt tracking) | 2.51 M | 5.75 | Large val (subsets 2/3/5 held-out)/test (subsets 1/4) gap identified -- warehouse-shortcut overfitting, not a bug (see Handoff.md). Augmentation + warm-restart fix in progress. |
| yolo26s-small-coco-50-epochs | yolo26s-small-coco | 0.210 | 0.409 (epoch 50/50) | 9.95 M | 22.45 | Real improvement over nano from larger architecture; not yet at target. Extending to 100 epochs. |
| frcnn-mobilenetv3-augment-epoch20 | frcnn-mobilenetv3-augment | 0.2280 | 0.2562 (epoch 20/50) | 18.95 M | not measured (periodic in-training check, not a full eval.py run) | |
| frcnn-mobilenetv3-augment-epoch30 | frcnn-mobilenetv3-augment | 0.2271 | 0.2647 (epoch 30/50) | 18.95 M | not measured (periodic in-training check, not a full eval.py run) | val still climbing, test starting to slip |
| frcnn-mobilenetv3-augment-epoch40 | frcnn-mobilenetv3-augment | 0.2125 | 0.2760 (best so far, epoch 34) | 18.95 M | not measured (periodic in-training check, not a full eval.py run) | val/test gap widened further -- the decline (0.2280 -> 0.2271 -> 0.2125) is the val/test divergence this branch's biased split is built to demonstrate; final full eval.py on best.pt (epoch 34) was 0.2306 -- see row below |
| frcnn-mobilenetv3-augment-50-epochs | frcnn-mobilenetv3-augment | 0.2306 | 0.2760 (epoch 34/50) | 18.95 M | 42.316 | full eval.py run, end of training, on epoch-34's best.pt |
| yolov4t-loco-50-of-100-epochs | yolov4t-loco | 0.000 | 0.000 (epoch 50/100) | not measured | not measured | Loss collapsed to exactly 0.0000 -- broken, not a legitimate low score. Abandoned per earlier decision (unmaintained recipe, anchor/backbone setup never worked); not investigated further. |

Target: 0.70 mAP@0.5.
