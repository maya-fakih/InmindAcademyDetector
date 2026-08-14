"""Load LOCO data for Torchvision detectors."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from einops import rearrange
from numpy import asarray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from augmentation import random_background_swap, random_horizontal_flip


class DetectionTarget(TypedDict):
    boxes: Tensor
    labels: Tensor


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

# Target fraction of development images (subsets 2/3/5) held out for validation.
VALIDATION_FRACTION = 0.2

# Seed for the validation split itself, kept separate from the training seed in
# config.yaml so shuffling the split doesn't require touching training config.
SPLIT_SEED = 0

# Image ``path`` values are absolute within the LOCO archive and start with this prefix.
LOCO_ARCHIVE_ROOT = "/dataset"
ANNOTATIONS_DIRNAME = "rgb"


def compute_balanced_split(
    images: list[dict],
    annotations: list[dict],
    val_fraction: float,
    seed: int,
) -> dict[int, Literal["train", "validation"]]:
    """Assign each image to train/validation, balancing per-class annotation density.

    Adapted from Tadjine et al. 2025 ("Object detection based on Logistic
    Objects in Context (LOCO) dataset: an improved dataset split..."), which
    showed the default stride-based split leaves classes like forklift
    severely under/over-represented in some buckets. Their method spans all
    5 LOCO subsets; ours is restricted to subsets 2/3/5 only, since 1/4 are
    reserved for final evaluation and must never influence training-time
    decisions.
    """
    annotations_by_image: defaultdict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    class_ids = sorted({annotation["category_id"] for annotation in annotations})
    total_counts = Counter(annotation["category_id"] for annotation in annotations)
    total_annotations = sum(total_counts.values())
    target_density = {
        class_id: total_counts[class_id] / total_annotations for class_id in class_ids
    }

    image_ids = [image["id"] for image in images]
    random.Random(seed).shuffle(image_ids)

    bucket_counts: dict[str, Counter] = {"train": Counter(), "validation": Counter()}
    bucket_sizes = {"train": 0, "validation": 0}
    max_validation = round(val_fraction * len(image_ids))
    max_train = len(image_ids) - max_validation

    assignment: dict[int, Literal["train", "validation"]] = {}
    for image_id in image_ids:
        image_classes = Counter(
            annotation["category_id"] for annotation in annotations_by_image[image_id]
        )
        candidates = ["train", "validation"]
        if bucket_sizes["validation"] >= max_validation:
            candidates = ["train"]
        elif bucket_sizes["train"] >= max_train:
            candidates = ["validation"]

        best_bucket = min(
            candidates,
            key=lambda bucket: (
                _density_mse(bucket_counts[bucket], image_classes, target_density, class_ids),
                -_remaining_capacity_ratio(bucket, bucket_sizes, max_train, max_validation),
            ),
        )
        assignment[image_id] = best_bucket
        bucket_counts[best_bucket].update(image_classes)
        bucket_sizes[best_bucket] += 1

    return assignment


def _remaining_capacity_ratio(
    bucket: str, bucket_sizes: dict[str, int], max_train: int, max_validation: int
) -> float:
    """Fraction of ``bucket``'s target size still unfilled; used to break exact MSE ties.

    Without this, ``min()`` always favors whichever bucket is listed first among
    the tied candidates (``"train"``), which can starve validation on ties even
    when it has plenty of room left. Breaking ties toward the bucket furthest
    behind its own target keeps both buckets filling at a similar pace.
    """
    max_size = max_train if bucket == "train" else max_validation
    if max_size == 0:
        return 0.0
    return (max_size - bucket_sizes[bucket]) / max_size


def _density_mse(
    running_counts: Counter,
    image_classes: Counter,
    target_density: dict[int, float],
    class_ids: list[int],
) -> float:
    """MSE vs. target density if ``image_classes`` were added to ``running_counts``."""
    hypothetical = running_counts + image_classes
    total = sum(hypothetical.values())
    if total == 0:
        return 0.0
    return sum(
        (hypothetical[class_id] / total - target_density[class_id]) ** 2 for class_id in class_ids
    ) / len(class_ids)


class LocoDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Load a LOCO split with boxes in absolute ``xyxy`` coordinates."""

    def __init__(
        self,
        raw_dir: Path,
        split: Literal["train", "validation", "test"],
        background_swap_prob: float = 0.0,
        hflip_prob: float = 0.0,
    ) -> None:
        # Augmentation is train-only by construction: forcing these to 0 for
        # validation/test here (rather than trusting the caller) means a
        # misconfigured config.yaml can't silently leak augmentation into
        # the numbers we report or select best.pt on.
        self.background_swap_prob = background_swap_prob if split == "train" else 0.0
        self.hflip_prob = hflip_prob if split == "train" else 0.0
        raw_dir = self._resolve_raw_dir(raw_dir, split)
        self.images: list[dict] = []
        self.image_paths: dict[int, Path] = {}
        self.annotations: defaultdict[int, list[dict]] = defaultdict(list)
        categories: list[dict] | None = None

        if split == "test":
            # Final-evaluation subsets: load everything, no split decision needed.
            for filename in SUBSET_FILES[split]:
                subset = self._load_subset(raw_dir / ANNOTATIONS_DIRNAME / filename)
                categories = self._check_categories(subset, categories, filename)
                self._extend(subset, raw_dir)
        else:
            # Training subsets: gather everything from 2/3/5 first (pass 1), decide
            # the whole train/validation split at once (needs the full picture to
            # balance classes), then keep only the images belonging to this split
            # (pass 2).
            all_images: list[dict] = []
            all_annotations: list[dict] = []
            all_image_paths: dict[int, Path] = {}
            for filename in SUBSET_FILES[split]:
                subset = self._load_subset(raw_dir / ANNOTATIONS_DIRNAME / filename)
                categories = self._check_categories(subset, categories, filename)
                source_image_ids = {image["id"] for image in subset["images"]}
                id_map: dict[int, int] = {}
                for image in subset["images"]:
                    new_id = len(all_images) + 1
                    id_map[image["id"]] = new_id
                    all_images.append({**image, "id": new_id})
                    all_image_paths[new_id] = raw_dir / Path(image["path"]).relative_to(
                        LOCO_ARCHIVE_ROOT
                    )
                for annotation in subset["annotations"]:
                    if annotation["image_id"] not in source_image_ids:
                        raise ValueError(
                            f"Annotation {annotation.get('id')} references an unknown image "
                            f"in {filename}"
                        )
                    new_image_id = id_map.get(annotation["image_id"])
                    if new_image_id is None:
                        continue
                    all_annotations.append({**annotation, "image_id": new_image_id})

            split_assignment = compute_balanced_split(
                all_images, all_annotations, VALIDATION_FRACTION, SPLIT_SEED
            )
            self.images = [image for image in all_images if split_assignment[image["id"]] == split]
            kept_ids = {image["id"] for image in self.images}
            self.image_paths = {
                image_id: path for image_id, path in all_image_paths.items() if image_id in kept_ids
            }
            for annotation in all_annotations:
                if annotation["image_id"] in kept_ids:
                    self.annotations[annotation["image_id"]].append(annotation)

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
