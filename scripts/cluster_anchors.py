"""Re-cluster YOLOv4-tiny anchors on LOCO's actual box shapes.

The vendored ``yolov4-tiny.cfg`` still ships with anchors k-means'd on COCO,
which has very different object aspect ratios than LOCO's forklifts/pallets/
stillages. This script re-runs that clustering on LOCO itself and prints (or
writes) new anchors sized for the network's actual input resolution.

Only reads subsets 2/3/5 (``DEVELOPMENT_FILES`` in ``dataset.py``) -- subsets
1/4 are the held-out final-eval split and must never influence any
training-time decision, anchor shapes included. Uses each image's ``width``/
``height`` fields from the annotation JSON directly, so it does not need the
actual image files on disk -- only the ``rgb/*.json`` annotation files.

Usage:
    python scripts/cluster_anchors.py --raw-dir dataset
    python scripts/cluster_anchors.py --raw-dir dataset --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from dataset import ANNOTATIONS_DIRNAME, DEVELOPMENT_FILES

DEFAULT_NUM_ANCHORS = 6
DEFAULT_IMAGE_SIZE = 416


def load_letterboxed_box_sizes(raw_dir: Path, image_size: int) -> np.ndarray:
    """Return an ``(N, 2)`` array of box (width, height) in pixels, as they'd
    appear after the same letterbox resize used at train/inference time --
    otherwise clustering would be in the wrong scale for a 416x416 input."""
    sizes: list[tuple[float, float]] = []
    for filename in DEVELOPMENT_FILES:
        path = raw_dir / ANNOTATIONS_DIRNAME / filename
        subset = json.loads(path.read_text(encoding="utf-8"))
        image_dims = {image["id"]: (image["width"], image["height"]) for image in subset["images"]}
        for annotation in subset["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            image_w, image_h = image_dims[annotation["image_id"]]
            scale = image_size / max(image_w, image_h)
            _, _, box_w, box_h = annotation["bbox"]
            if box_w <= 0 or box_h <= 0:
                continue
            sizes.append((box_w * scale, box_h * scale))
    return np.array(sizes, dtype=np.float64)


def iou_wh(boxes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """IoU between every box and every centroid, treating both as w/h pairs
    sharing a common top-left corner (the standard YOLO anchor-matching
    assumption: only shape matters, not position)."""
    box_w, box_h = boxes[:, 0:1], boxes[:, 1:2]
    cent_w, cent_h = centroids[None, :, 0], centroids[None, :, 1]
    inter = np.minimum(box_w, cent_w) * np.minimum(box_h, cent_h)
    union = box_w * box_h + cent_w * cent_h - inter
    return inter / union


def kmeans_iou(
    boxes: np.ndarray, k: int, num_restarts: int = 10, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Standard YOLOv2-style anchor k-means: distance = 1 - IoU(box, centroid).

    Runs ``num_restarts`` random initializations and keeps the run with the
    best mean best-anchor IoU, since plain k-means is sensitive to init.
    """
    rng = np.random.default_rng(seed)
    best_centroids, best_score = None, -1.0

    for _ in range(num_restarts):
        centroids = boxes[rng.choice(len(boxes), size=k, replace=False)].copy()
        for _ in range(300):
            distances = 1 - iou_wh(boxes, centroids)
            assignments = distances.argmin(axis=1)
            new_centroids = np.array(
                [
                    boxes[assignments == cluster].mean(axis=0)
                    if np.any(assignments == cluster)
                    else centroids[cluster]
                    for cluster in range(k)
                ]
            )
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        score = float(iou_wh(boxes, centroids).max(axis=1).mean())
        if score > best_score:
            best_centroids, best_score = centroids, score

    return best_centroids, best_score


def format_anchors_line(centroids: np.ndarray) -> str:
    ordered = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    pairs = ", ".join(f"{w:.0f},{h:.0f}" for w, h in ordered)
    return f"anchors = {pairs}"


def write_cfg(cfg_path: Path, anchors_line: str) -> int:
    """Replace every ``anchors = ...`` line in the cfg. Returns the count replaced."""
    text = cfg_path.read_text(encoding="utf-8")
    new_text, count = re.subn(r"anchors[ \t]*=[ \t]*[\d, \t]+", anchors_line, text)
    cfg_path.write_text(new_text, encoding="utf-8")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--num-anchors", type=int, default=DEFAULT_NUM_ANCHORS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("vendor/pytorch_yolov4/yolov4-tiny.cfg"),
        help="cfg file to patch when --write is passed",
    )
    parser.add_argument(
        "--write", action="store_true", help="patch --cfg in place instead of only printing"
    )
    args = parser.parse_args()

    boxes = load_letterboxed_box_sizes(args.raw_dir, args.image_size)
    print(
        f"Clustering {len(boxes)} boxes from subsets {DEVELOPMENT_FILES} into "
        f"{args.num_anchors} anchors..."
    )
    centroids, mean_iou = kmeans_iou(boxes, args.num_anchors, args.restarts, args.seed)

    anchors_line = format_anchors_line(centroids)
    print(f"mean best-anchor IoU: {mean_iou:.4f}")
    print(anchors_line)

    if args.write:
        count = write_cfg(args.cfg, anchors_line)
        print(f"wrote {count} anchors= line(s) in {args.cfg}")


if __name__ == "__main__":
    main()
