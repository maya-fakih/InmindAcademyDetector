import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image

from dataset import LocoDataset


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


def test_loads_image_boxes_and_contiguous_labels(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    annotated = next(i for i in range(len(dataset)) if dataset[i][1]["boxes"].shape[0] > 0)
    image_CHW, target = dataset[annotated]

    assert image_CHW.dtype == torch.float32
    assert image_CHW.min() >= 0 and image_CHW.max() <= 1
    assert target["boxes"].tolist() == [[1, 2, 4, 6], [0, 0, 2, 1]]
    assert target["labels"].tolist() == [2, 1]
    assert dataset.num_classes == 3


def test_empty_annotations_have_detection_shapes(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    empty = next(i for i in range(len(dataset)) if dataset[i][1]["boxes"].shape[0] == 0)
    _, target = dataset[empty]

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_validation_holdout_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    train_dataset = make_dataset(tmp_path, split="train")
    validation_dataset = make_dataset(tmp_path, split="validation")

    assert len(train_dataset) + len(validation_dataset) == 4  # whole subsets-2/3/5 pool
    assert set(validation_dataset.image_paths.values()).isdisjoint(
        train_dataset.image_paths.values()
    )
    # Deterministic: re-running the split (a second LocoDataset construction) agrees exactly.
    again = make_dataset(tmp_path, split="validation")
    assert set(again.image_paths.values()) == set(validation_dataset.image_paths.values())


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

    assert image_CHW.shape == (3, 6, 8)
