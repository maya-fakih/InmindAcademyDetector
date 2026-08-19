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


# --- AMIR RECIPE SPLIT --------------------------------------------------
# This branch (frcnn-amir-recipe) is a from-scratch reproduction attempt of
# the split methodology used on Amir's `exp/03-recipe` branch
# (https://github.com/amiroo-star/inmind-detector, branch exp/03-recipe):
# train on whole subsets 2 and 5, validate on the *whole* of subset 3 as an
# unseen-warehouse holdout, and reserve subsets 1/4 for final test exactly
# as the assignment requires. This is deliberately different from both the
# assignment-template's "1-in-5 images per development subset" split and
# from this repo's own frcnn-mobilenetv3-augment demo split (which keeps
# validation *within* all three development subsets but biases which
# images land in it). Here validation is a whole subset the model never
# trains on, at all -- closer in spirit to how subsets 1/4 will actually
# be evaluated. Nothing from subsets 1/4 is read at training or
# checkpoint-selection time.
#
# Reported comparison point (NOT reproduced in this repo -- Amir's repo has
# no committed run logs / history.json to verify against): Amir reports
# ~26% test-set accuracy after 20 epochs on his own branch. See RESULTS.md.
# --------------------------------------------------------------------------

# Each LOCO annotation file is COCO JSON. The fields used here are:
#   images:      id, path (absolute inside the archive), width, height
#   annotations: image_id, category_id, bbox as [x, y, width, height], iscrowd
#   categories:  id, name
TRAIN_FILES = (
    "loco-sub2-v1-train.json",
    "loco-sub5-v1-train.json",
)
VALIDATION_FILES = ("loco-sub3-v1-train.json",)
TEST_FILES = (
    "loco-sub1-v1-val.json",
    "loco-sub4-v1-val.json",
)
SUBSET_FILES: dict[str, tuple[str, ...]] = {
    "train": TRAIN_FILES,
    "validation": VALIDATION_FILES,
    "test": TEST_FILES,
}

# Image ``path`` values are absolute within the LOCO archive and start with this prefix.
LOCO_ARCHIVE_ROOT = "/dataset"
ANNOTATIONS_DIRNAME = "rgb"


class LocoDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Load a LOCO split with boxes in absolute ``xyxy`` coordinates.

    Unlike the assignment template and the frcnn-mobilenetv3-augment demo
    branch, every split here uses *whole* subsets (see ``SUBSET_FILES``
    above) -- there is no per-image holdout fraction to compute, so every
    image in a split's files is used in full.
    """

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

        for filename in SUBSET_FILES[split]:
            subset = self._load_subset(raw_dir / ANNOTATIONS_DIRNAME / filename)
            subset_categories = sorted(subset["categories"], key=lambda category: category["id"])
            if categories is None:
                categories = subset_categories
            elif subset_categories != categories:
                raise ValueError(f"Category definitions differ in {filename}")

            source_image_ids = {image["id"] for image in subset["images"]}
            image_ids: dict[int, int] = {}
            for image in subset["images"]:
                # Whole-subset split: every image in this file belongs to this split.
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
