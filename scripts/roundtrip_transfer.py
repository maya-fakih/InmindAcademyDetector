"""Carry a checkpoint's weights across a class-count/order change between two
datasets (e.g. LOCO -> Roboflow pretrain -> back to LOCO), transferring
whatever classification-head rows match by name and leaving the rest
(backbone, neck, box-regression head, and any head rows with no match in the
source) to be trained/fine-tuned normally from there.

Two cases, auto-detected:
1. Source and target class lists are identical (same names, same order) --
   no head surgery needed. The checkpoint is copied through unchanged; you
   can resume training on it directly.
2. Class lists differ -- runs `transfer_matched_class_rows` from
   models/class_weight_transfer.py, which copies only the rows for classes
   present in both, and prints exactly which classes were and weren't
   transferred so the run's provenance is auditable at each step of the
   round trip, not just at the end.

This is intentionally direction-agnostic: run it LOCO->Roboflow before the
Roboflow pretrain, and again Roboflow->LOCO afterward, with the arguments
swapped.

Usage:
    uv run scripts/roundtrip_transfer.py \
        --source-checkpoint /path/to/loco_best.pt \
        --source-data-yaml dataset/data.yaml \
        --target-data-yaml /path/to/roboflow_export/data.yaml \
        --output /path/to/transferred_checkpoint.pt
"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics.nn.tasks import DetectionModel

from models.class_weight_transfer import list_candidate_tensors, transfer_matched_class_rows


def load_class_names(data_yaml_path: Path) -> list[str]:
    data = yaml.safe_load(data_yaml_path.read_text())
    names = data["names"]
    # Ultralytics data.yaml can store names as a list or a {index: name} dict.
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-data-yaml", required=True, type=Path)
    parser.add_argument("--target-data-yaml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_names = load_class_names(args.source_data_yaml)
    target_names = load_class_names(args.target_data_yaml)
    print(f"Source classes ({len(source_names)}): {source_names}")
    print(f"Target classes ({len(target_names)}): {target_names}")

    if source_names == target_names:
        print("Class lists are identical (same names, same order) -- no head "
              "surgery needed. Copying checkpoint through unchanged; resume "
              "training on it directly.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(args.source_checkpoint.read_bytes())
        print(f"Wrote {args.output}")
        return

    print("Class lists differ -- transferring matched classification-head rows.")
    from ultralytics import YOLO

    pretrained = YOLO(str(args.source_checkpoint)).model
    target_model = DetectionModel(cfg=pretrained.yaml, nc=len(target_names))

    # Flag which tensors this will touch, per the ASSUMPTION FLAGGED comment
    # in class_weight_transfer.py -- print for a human to eyeball before
    # trusting this on a real run.
    print("Candidate source tensors (verify these are classification-head, "
          f"not box-regression, before trusting the transfer): "
          f"{list_candidate_tensors(pretrained, len(source_names))}")
    print(f"Candidate target tensors: "
          f"{list_candidate_tensors(target_model, len(target_names))}")

    transferred = transfer_matched_class_rows(
        pretrained_model=pretrained,
        target_model=target_model,
        source_class_names=source_names,
        target_class_names=target_names,
    )
    print(f"Transferred {len(transferred)}/{len(target_names)} target classes: {transferred}")
    not_transferred = [name for name in target_names if name not in transferred]
    if not_transferred:
        print(f"NOT transferred (will train from random init): {not_transferred}")

    # DetectionModel.load() already copied everything else that matches by
    # shape (backbone, neck, box-regression head) as a side effect of
    # constructing target_model above via the same cfg -- but to be safe and
    # explicit, do it again here so this script's output is self-contained
    # and doesn't rely on transfer_matched_class_rows having already done it.
    target_model.load(pretrained)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target_model}, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
