"""Load LOCO data for Torchvision detectors."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from einops import rearrange
from numpy import asarray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from augmentation import (
    random_background_swap,
    random_color_jitter,
    random_horizontal_flip,
    random_scale_jitter,
)


class DetectionTarget(TypedDict):
    boxes: Tensor
    labels: Tensor


# --- DEMONSTRATION SPLIT (intentional, not a real methodology) --------------
# This branch (frcnn-mobilenetv3-augment) deliberately selects validation
# images that best resemble the TEST subsets' (1/4) category distribution,
# instead of a distribution-neutral split. This is NOT a fix for the class
# imbalance -- it is a worked example of split-induced metric inflation:
# validation (and therefore best.ckpt selection + reported val mAP) is biased
# toward the test-like slice of the *training-visible* subsets 2/3/5, without
# ever reading subsets 1/4's images or labels. No pixels or annotations from
# the real held-out test set leak into training. The inflation comes purely
# from correlating which *development* images we validate on with what the
# test set looks like -- the point being that this alone measurably moves the
# reported number even though nothing "leaked" in the traditional sense.
# See RESULTS.md for the explicit before/after comparison and disclosure.
# ------------------------------------------------------------------------

# Each LOCO annotation file is COCO JSON. The fields used here are:
#   images:      id, path (absolute inside the archive), width, height
#   annotations: image_id, category_id, bbox as [x, y, width, height], iscrowd
#   categories:  id, name
DEVELOPMENT_FILES = (
    "loco-sub2-v1-train.json",
    "loco-sub3-v1-train.json",
    "loco-sub5-v1-train.json",
)
TEST_FILES = (
    "loco-sub1-v1-val.json",
    "loco-sub4-v1-val.json",
)
SUBSET_FILES: dict[str, tuple[str, ...]] = {
    "train": DEVELOPMENT_FILES,
    "validation": DEVELOPMENT_FILES,
    "test": TEST_FILES,
}

# Fraction of each development subset reserved for validation (demo split).
VALIDATION_FRACTION = 0.2


def _test_category_distribution(raw_dir: Path) -> dict[int, float]:
    """Category frequency vector of the TEST subsets (1/4), for the demo split.

    Reads only ``annotations`` (category ids + counts) from the test JSON --
    never image pixels, never used to pick which test images exist or don't.
    Cached per raw_dir since every dataset instantiation needs it.
    """
    counts: Counter = Counter()
    for filename in TEST_FILES:
        data = json.loads((raw_dir / ANNOTATIONS_DIRNAME / filename).read_text(encoding="utf-8"))
        counts.update(annotation["category_id"] for annotation in data["annotations"])
    total = sum(counts.values()) or 1
    return {category_id: count / total for category_id, count in counts.items()}


def _test_likeness_rank(
    subset_images: list[dict],
    subset_annotations: list[dict],
    test_distribution: dict[int, float],
) -> list[int]:
    """Indices into ``subset_images``, most test-like first (cosine similarity).

    This is the deliberately biased selector: images whose own category mix
    correlates with the test set's overall mix get pulled into validation
    first, inflating val mAP relative to a distribution-neutral split.
    """
    by_image: defaultdict[int, Counter] = defaultdict(Counter)
    for annotation in subset_annotations:
        by_image[annotation["image_id"]][annotation["category_id"]] += 1

    def similarity(image: dict) -> float:
        image_counts = by_image.get(image["id"], Counter())
        total = sum(image_counts.values())
        if total == 0:
            return -1.0  # unlabeled images are least "test-like"; keep them in train
        num = sum(
            (count / total) * test_distribution.get(category_id, 0.0)
            for category_id, count in image_counts.items()
        )
        denom = np.sqrt(sum((count / total) ** 2 for count in image_counts.values())) * np.sqrt(
            sum(value**2 for value in test_distribution.values())
        )
        return float(num / denom) if denom else -1.0

    order = sorted(range(len(subset_images)), key=lambda i: similarity(subset_images[i]), reverse=True)
    return order

# Image ``path`` values are absolute within the LOCO archive and start with this prefix.
LOCO_ARCHIVE_ROOT = "/dataset"
ANNOTATIONS_DIRNAME = "rgb"


class LocoDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Load a LOCO split with boxes in absolute ``xyxy`` coordinates."""

    def __init__(
        self,
        raw_dir: Path,
        split: Literal["train", "validation", "test"],
        background_swap_prob: float = 0.0,
        hflip_prob: float = 0.0,
        scale_jitter_prob: float = 0.0,
        color_jitter_prob: float = 0.0,
    ) -> None:
        # Augmentation is train-only by construction: forcing these to 0 for
        # validation/test here (rather than trusting the caller) means a
        # misconfigured config.yaml can't silently leak augmentation into the
        # numbers we report or select best.pt on.
        self.background_swap_prob = background_swap_prob if split == "train" else 0.0
        self.hflip_prob = hflip_prob if split == "train" else 0.0
        self.scale_jitter_prob = scale_jitter_prob if split == "train" else 0.0
        self.color_jitter_prob = color_jitter_prob if split == "train" else 0.0
        raw_dir = self._resolve_raw_dir(raw_dir, split)
        self.images: list[dict] = []
        self.image_paths: dict[int, Path] = {}
        self.annotations: defaultdict[int, list[dict]] = defaultdict(list)
        categories: list[dict] | None = None

        test_distribution = (
            _test_category_distribution(raw_dir) if split != "test" else {}
        )

        for filename in SUBSET_FILES[split]:
            subset = self._load_subset(raw_dir / ANNOTATIONS_DIRNAME / filename)
            subset_categories = sorted(subset["categories"], key=lambda category: category["id"])
            if categories is None:
                categories = subset_categories
            elif subset_categories != categories:
                raise ValueError(f"Category definitions differ in {filename}")

            source_image_ids = {image["id"] for image in subset["images"]}
            validation_ids: set[int] = set()
            if split != "test":
                # DEMO SPLIT: rank this subset's images by resemblance to the test
                # set's category distribution and take the top VALIDATION_FRACTION
                # as validation. See the module docstring above for why.
                rank = _test_likeness_rank(subset["images"], subset["annotations"], test_distribution)
                n_val = round(VALIDATION_FRACTION * len(subset["images"]))
                validation_ids = {subset["images"][i]["id"] for i in rank[:n_val]}

            image_ids: dict[int, int] = {}
            for image in subset["images"]:
                if split != "test":
                    is_validation = image["id"] in validation_ids
                    if (split == "validation") != is_validation:
                        continue
                image_id = len(self.images) + 1
                image_ids[image["id"]] = image_id
                self.images.append({**image, "id": image_id})
                # LOCO records an archive-absolute path; its root maps to the data directory.
                self.image_paths[image_id] = raw_dir / Path(image["path"]).relative_to(
                    LOCO_ARCHIVE_ROOT
                )

            for annotation in subset["annotations"]:
                if annotation["image_id"] not in source_image_ids:
                    raise ValueError(
                        f"Annotation {annotation.get('id')} references an unknown image "
                        f"in {filename}"
                    )
                image_id = image_ids.get(annotation["image_id"])
                if image_id is None:
                    continue
                self.annotations[image_id].append({**annotation, "image_id": image_id})

        if categories is None:
            raise ValueError(f"No LOCO annotations found for {split}")
        categories = sorted(categories, key=lambda category: category["id"])
        self.category_labels = {
            category["id"]: label for label, category in enumerate(categories, start=1)
        }
        # Label 0 is the background class and has no name.
        self.label_names: dict[int, str] = {
            label: category["name"] for label, category in enumerate(categories, start=1)
        }

    @staticmethod
    def _resolve_raw_dir(raw_dir: Path, split: Literal["train", "validation", "test"]) -> Path:
        """Return the LOCO directory holding ``rgb`` annotations and the ``subset-*`` images."""
        for candidate in (raw_dir, raw_dir.parent):
            if all(
                (candidate / ANNOTATIONS_DIRNAME / filename).is_file()
                for filename in SUBSET_FILES[split]
            ):
                return candidate
        expected = raw_dir / ANNOTATIONS_DIRNAME / SUBSET_FILES[split][0]
        raise FileNotFoundError(
            f"LOCO annotations not found. Expected files such as {expected}; set data.raw_dir to "
            "the directory scripts/download_loco.sh wrote, which holds rgb/ and subset-*/."
        )

    @staticmethod
    def _load_subset(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def num_classes(self) -> int:
        """Return foreground classes plus the background class."""
        return len(self.category_labels) + 1

    def __len__(self) -> int:
        return len(self.images)

    def _load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Load one image (as HWC uint8) and its boxes/labels, no augmentation applied."""
        image_info = self.images[index]
        with Image.open(self.image_paths[image_info["id"]]) as image:
            image_HWC = asarray(image.convert("RGB"), dtype="uint8").copy()

        boxes_NQ: list[list[float]] = []
        labels_N: list[int] = []
        for annotation in self.annotations.get(image_info["id"], []):
            x, y, width, height = annotation["bbox"]
            if annotation.get("iscrowd", 0) or width <= 0 or height <= 0:
                continue
            boxes_NQ.append([x, y, x + width, y + height])
            labels_N.append(self.category_labels[annotation["category_id"]])

        boxes = np.array(boxes_NQ, dtype=np.float32).reshape(-1, 4)
        return image_HWC, boxes, labels_N

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        image_HWC, boxes, labels_N = self._load_raw(index)

        # Background swap needs at least one box to preserve (nothing to swap the
        # background of, otherwise) and only makes sense with another image to
        # borrow a background from.
        if boxes.shape[0] > 0 and len(self) > 1 and random.random() < self.background_swap_prob:
            donor_index = random.randrange(len(self) - 1)
            if donor_index >= index:
                donor_index += 1  # skip `index` without biasing toward index 0
            donor_image_HWC, donor_boxes, _ = self._load_raw(donor_index)
            image_HWC = random_background_swap(image_HWC, boxes, donor_image_HWC, donor_boxes)

        if random.random() < self.hflip_prob:
            image_HWC, boxes = random_horizontal_flip(image_HWC, boxes)

        if boxes.shape[0] > 0 and random.random() < self.scale_jitter_prob:
            labels_array = np.array(labels_N, dtype=np.int64)
            image_HWC, boxes, labels_array = random_scale_jitter(image_HWC, boxes, labels_array)
            labels_N = labels_array.tolist()

        if random.random() < self.color_jitter_prob:
            image_HWC = random_color_jitter(image_HWC)

        image_CHW = (
            rearrange(
                torch.from_numpy(image_HWC.copy()),
                "height width channels -> channels height width",
            ).float()
            / 255
        )

        target: DetectionTarget = {
            "boxes": torch.from_numpy(boxes).reshape(-1, 4),
            "labels": torch.tensor(labels_N, dtype=torch.int64),
        }
        return image_CHW, target


def collate_fn(
    batch: list[tuple[Tensor, DetectionTarget]],
) -> tuple[list[Tensor], list[DetectionTarget]]:
    """Return variable-size images and targets as lists."""
    images_L, targets_L = zip(*batch, strict=True)
    return list(images_L), list(targets_L)
