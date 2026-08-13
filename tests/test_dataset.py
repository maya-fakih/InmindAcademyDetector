import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image

from dataset import LocoDataset, compute_balanced_split


def write_subset(
    root: Path,
    filename: str,
    subset: int,
    image_specs: list[tuple[int, str, int, int]],
    annotations: list[dict],
) -> None:
    """Write an annotation file into ``root/rgb`` and its images into ``root/subset-N``."""
    images = []
    for image_id, relative_path, width, height in image_specs:
        image_path = root / f"subset-{subset}" / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (width, height), color=(255, 128, 0)).save(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "path": f"/dataset/subset-{subset}/{relative_path}",
                "width": width,
                "height": height,
            }
        )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 20, "name": "cart"}, {"id": 10, "name": "pallet"}],
    }
    annotations_dir = root / "rgb"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / filename).write_text(json.dumps(coco), encoding="utf-8")


def make_dataset(
    tmp_path: Path, split: Literal["train", "validation", "test"] = "train"
) -> LocoDataset:
    write_subset(
        tmp_path,
        "loco-sub2-v1-train.json",
        2,
        [
            (1, "validation.png", 8, 6),
            (2, "nested/dir/annotated.png", 8, 6),
            (3, "train.png", 5, 4),
        ],
        [
            {"id": 1, "image_id": 2, "category_id": 20, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 2, "category_id": 10, "bbox": [0, 0, 2, 1]},
        ],
    )
    write_subset(tmp_path, "loco-sub3-v1-train.json", 3, [(1, "empty.png", 5, 4)], [])
    write_subset(tmp_path, "loco-sub5-v1-train.json", 5, [], [])
    write_subset(tmp_path, "loco-sub1-v1-val.json", 1, [(1, "test-one.png", 5, 4)], [])
    write_subset(tmp_path, "loco-sub4-v1-val.json", 4, [(1, "test-four.png", 5, 4)], [])
    return LocoDataset(tmp_path, split=split)


def find_item(dataset: LocoDataset, *, has_annotations: bool):
    """Return the first (image, target) pair matching ``has_annotations``, regardless
    of which split index it lands at (the balanced split can reorder which image is
    where, so tests must not assume a fixed position)."""
    for index in range(len(dataset)):
        image_CHW, target = dataset[index]
        if (target["boxes"].shape[0] > 0) == has_annotations:
            return image_CHW, target
    raise AssertionError(f"No item with has_annotations={has_annotations} found")


def test_loads_image_boxes_and_contiguous_labels(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    image_CHW, target = find_item(dataset, has_annotations=True)

    assert image_CHW.dtype == torch.float32
    assert image_CHW.min() >= 0 and image_CHW.max() <= 1
    assert target["boxes"].tolist() == [[1, 2, 4, 6], [0, 0, 2, 1]]
    assert target["labels"].tolist() == [2, 1]
    assert dataset.num_classes == 3


def test_empty_annotations_have_detection_shapes(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    _, target = find_item(dataset, has_annotations=False)

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_validation_holdout_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    train_dataset = make_dataset(tmp_path, split="train")
    validation_dataset = make_dataset(tmp_path, split="validation")

    # 4 development images total (3 in sub2, 1 in sub3, 0 in sub5).
    assert len(train_dataset) + len(validation_dataset) == 4
    assert set(validation_dataset.image_paths.values()).isdisjoint(
        train_dataset.image_paths.values()
    )

    # Rebuilding from scratch must reproduce the exact same split (same seed).
    train_again = make_dataset(tmp_path, split="train")
    assert set(train_dataset.image_paths.values()) == set(train_again.image_paths.values())


def test_test_split_uses_subsets_one_and_four(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, split="test")

    assert len(dataset) == 2


def test_resolves_when_pointed_at_annotations_dir(tmp_path: Path) -> None:
    root_dataset = make_dataset(tmp_path)
    rgb_dataset = LocoDataset(tmp_path / "rgb", split="train")

    assert len(rgb_dataset) == len(root_dataset)


def test_image_paths_do_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    monkeypatch.chdir(tmp_path.parent)
    image_CHW, _ = dataset[0]

    assert image_CHW.dtype == torch.float32


def test_balanced_split_covers_every_image_exactly_once() -> None:
    images = [{"id": i} for i in range(1, 11)]
    annotations = [{"image_id": i, "category_id": 1} for i in range(1, 11)]

    assignment = compute_balanced_split(images, annotations, val_fraction=0.3, seed=0)

    assert set(assignment.keys()) == {image["id"] for image in images}
    assert set(assignment.values()) <= {"train", "validation"}


def test_balanced_split_is_deterministic_for_the_same_seed() -> None:
    images = [{"id": i} for i in range(1, 11)]
    annotations = [{"image_id": i, "category_id": 1} for i in range(1, 11)]

    first = compute_balanced_split(images, annotations, val_fraction=0.3, seed=0)
    second = compute_balanced_split(images, annotations, val_fraction=0.3, seed=0)

    assert first == second


def test_balanced_split_does_not_starve_validation_on_ties() -> None:
    """Regression test: images with identical class content produce exact MSE ties
    between buckets. An earlier version of this function broke ties by always
    preferring 'train' (the first candidate), which silently starved validation
    even when it had room. Ties must instead favor whichever bucket is furthest
    behind its target size.
    """
    images = [{"id": i} for i in range(1, 11)]
    annotations = [{"image_id": i, "category_id": 1} for i in range(1, 11)]

    assignment = compute_balanced_split(images, annotations, val_fraction=0.4, seed=0)

    validation_count = sum(1 for bucket in assignment.values() if bucket == "validation")
    assert validation_count == 4
