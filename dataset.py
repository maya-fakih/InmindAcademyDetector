"""Load LOCO data for Torchvision detectors."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import torch
from einops import rearrange
from numpy import asarray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class DetectionTarget(TypedDict):
    boxes: Tensor
    labels: Tensor


# --- AMIR RECIPE SPLIT --------------------------------------------------
# This branch (frcnn-amir-recipe) mirrors Amir's exp/03-recipe branch
# (https://github.com/amiroo-star/inmind-detector) end to end: train on
# whole subsets 2 and 5, validate on the *whole* of subset 3 as an unseen
# warehouse holdout, test on subsets 1/4 exactly as the assignment
# requires. This is disclosed, collaborative reuse of a teammate's split +
# augmentation approach for comparison purposes (see RESULTS.md), not an
# independent methodology.
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


def _augment(image_CHW: Tensor, target: DetectionTarget) -> tuple[Tensor, DetectionTarget]:
    """Apply a horizontal flip and photometric jitter to one training sample.

    Ported from Amir's exp/03-recipe branch. The graded subsets are warehouses the model
    never sees, and the measured cost of that shift is large: the same weights score 0.55
    on held-out images from the training warehouses and 0.24 across warehouses. Colour and
    brightness are exactly what differs between sites -- lighting, camera, white balance --
    so jittering them attacks the gap directly, while a horizontal flip is a
    label-preserving symmetry of these scenes.

    Boxes are in absolute ``xyxy``. A flip mirrors x, so the new left edge is the image
    width minus the old right edge; forgetting to swap the two produces boxes with
    ``x1 > x2``, which silently trains on empty regions.

    Vertical flips are deliberately excluded: warehouses have a floor, and an upside-down
    pallet is not a view the detector will ever meet.
    """
    if torch.rand(1).item() < 0.5:
        image_CHW = torch.flip(image_CHW, dims=[2])
        boxes = target["boxes"]
        if boxes.numel():
            width = image_CHW.shape[2]
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            target["boxes"] = flipped

    # Brightness, contrast and saturation, each within +/-20%. Geometry is untouched, so
    # the boxes stay valid without any adjustment.
    brightness = 0.8 + 0.4 * torch.rand(1).item()
    contrast = 0.8 + 0.4 * torch.rand(1).item()
    saturation = 0.8 + 0.4 * torch.rand(1).item()

    image_CHW = image_CHW * brightness
    mean = image_CHW.mean()
    image_CHW = (image_CHW - mean) * contrast + mean
    grey = image_CHW.mean(dim=0, keepdim=True)
    image_CHW = (image_CHW - grey) * saturation + grey
    return image_CHW.clamp(0.0, 1.0), target


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
        augment: bool = False,
    ) -> None:
        self.augment = augment
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

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        image_info = self.images[index]
        with Image.open(self.image_paths[image_info["id"]]) as image:
            image_HWC = asarray(image.convert("RGB"), dtype="uint8").copy()
        image_CHW = (
            rearrange(
                torch.from_numpy(image_HWC),
                "height width channels -> channels height width",
            ).float()
            / 255
        )

        boxes_NQ: list[list[float]] = []
        labels_N: list[int] = []
        for annotation in self.annotations.get(image_info["id"], []):
            x, y, width, height = annotation["bbox"]
            if annotation.get("iscrowd", 0) or width <= 0 or height <= 0:
                continue
            boxes_NQ.append([x, y, x + width, y + height])
            labels_N.append(self.category_labels[annotation["category_id"]])

        target: DetectionTarget = {
            "boxes": torch.tensor(boxes_NQ, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels_N, dtype=torch.int64),
        }
        if self.augment:
            image_CHW, target = _augment(image_CHW, target)
        return image_CHW, target


def collate_fn(
    batch: list[tuple[Tensor, DetectionTarget]],
) -> tuple[list[Tensor], list[DetectionTarget]]:
    """Return variable-size images and targets as lists."""
    images_L, targets_L = zip(*batch, strict=True)
    return list(images_L), list(targets_L)
