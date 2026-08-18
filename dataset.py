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


# Each LOCO annotation file is COCO JSON. The fields used here are:
#   images:      id, path (absolute inside the archive), width, height
#   annotations: image_id, category_id, bbox as [x, y, width, height], iscrowd
#   categories:  id, name
#
# ALTERNATE SPLIT (this branch only, per Malek's sign-off): subsets 2/3/5 are
# used whole for training (no held-out slice cut from them -- compare
# yolo26s-coco, which reserves ~20% of 2/3/5 for validation via
# compute_balanced_split and keeps 1+4 untouched until final eval). Here,
# subset 4 is promoted to the validation split instead, and only subset 1
# remains as the blind test set.
#
# Consequence worth knowing before trusting numbers from this branch: best.pt
# is selected on subset-4 performance, so subset 4 is no longer a held-out
# set for this run -- only the final subset-1 test number is unbiased.
# Reported "val_mAP50" during training is a validation metric now, not a
# second held-out test number the way it was on yolo26s-coco.
TRAIN_FILES = (
    "loco-sub2-v1-train.json",
    "loco-sub3-v1-train.json",
    "loco-sub5-v1-train.json",
)
VALIDATION_FILES = ("loco-sub4-v1-val.json",)
TEST_FILES = ("loco-sub1-v1-val.json",)
SUBSET_FILES: dict[str, tuple[str, ...]] = {
    "train": TRAIN_FILES,
    "validation": VALIDATION_FILES,
    "test": TEST_FILES,
}

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
        # misconfigured config.yaml can't silently leak augmentation into
        # the numbers we report or select best.pt on.
        self.background_swap_prob = background_swap_prob if split == "train" else 0.0
        self.hflip_prob = hflip_prob if split == "train" else 0.0
        self.scale_jitter_prob = scale_jitter_prob if split == "train" else 0.0
        self.color_jitter_prob = color_jitter_prob if split == "train" else 0.0
        raw_dir = self._resolve_raw_dir(raw_dir, split)
        self.images: list[dict] = []
        self.image_paths: dict[int, Path] = {}
        self.annotations: defaultdict[int, list[dict]] = defaultdict(list)
        categories: list[dict] | None = None

        # Every split on this branch maps to a fixed, disjoint set of whole subset
        # files (see SUBSET_FILES above) -- no in-code split decision needed, unlike
        # yolo26s-coco's compute_balanced_split over a shared 2/3/5 pool.
        for filename in SUBSET_FILES[split]:
            subset = self._load_subset(raw_dir / ANNOTATIONS_DIRNAME / filename)
            categories = self._check_categories(subset, categories, filename)
            self._extend(subset, raw_dir)

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
    def _check_categories(subset: dict, categories: list[dict] | None, filename: str) -> list[dict]:
        subset_categories = sorted(subset["categories"], key=lambda category: category["id"])
        if categories is None:
            return subset_categories
        if subset_categories != categories:
            raise ValueError(f"Category definitions differ in {filename}")
        return categories

    def _extend(self, subset: dict, raw_dir: Path) -> None:
        """Append every image/annotation in ``subset`` (used for the untouched test split)."""
        source_image_ids = {image["id"] for image in subset["images"]}
        image_ids: dict[int, int] = {}
        for image in subset["images"]:
            image_id = len(self.images) + 1
            image_ids[image["id"]] = image_id
            self.images.append({**image, "id": image_id})
            self.image_paths[image_id] = raw_dir / Path(image["path"]).relative_to(
                LOCO_ARCHIVE_ROOT
            )
        for annotation in subset["annotations"]:
            if annotation["image_id"] not in source_image_ids:
                raise ValueError(
                    f"Annotation {annotation.get('id')} references an unknown image in {subset}"
                )
            image_id = image_ids.get(annotation["image_id"])
            if image_id is None:
                continue
            self.annotations[image_id].append({**annotation, "image_id": image_id})

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
