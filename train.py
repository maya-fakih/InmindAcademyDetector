"""Train the LOCO Faster R-CNN baseline."""

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb.sdk.wandb_run import Run

from dataset import LocoDataset, collate_fn
from metrics.accuracy import compute_map50
from models.baseline import create_model
from utils.config import load_config


def compute_lr(
    epoch: int, base_lr: float, warmup_epochs: int, total_epochs: int, final_fraction: float
) -> float:
    """Linear warmup for `warmup_epochs`, then cosine decay to `final_fraction` of base_lr."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    span = max(1, total_epochs - warmup_epochs)
    progress = min((epoch - warmup_epochs) / span, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * final_fraction + (base_lr - base_lr * final_fraction) * cosine


def train(config: dict[str, Any], run: Run | None, resume_from: str | None = None) -> None:
    """Train the detector and save the best weights."""
    settings = config["train"]
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])

    data = config["data"]
    augment = config.get("augment", {})
    if augment.get("enabled", False):
        train_dataset = LocoDataset(
            Path(data["raw_dir"]),
            split="train",
            background_swap_prob=augment.get("background_swap_prob", 0.0),
            hflip_prob=augment.get("hflip_prob", 0.0),
            scale_jitter_prob=augment.get("scale_jitter_prob", 0.0),
            color_jitter_prob=augment.get("color_jitter_prob", 0.0),
        )
    else:
        train_dataset = LocoDataset(Path(data["raw_dir"]), split="train")
    # Validation must stay clean -- it's what best.pt is selected on, so augmenting
    # it would make "best" mean "best on artificially varied backgrounds" instead
    # of "best on this warehouse's real val images". LocoDataset itself also
    # forces these to 0 off the train split, this is just not passing them at all.
    val_dataset = LocoDataset(Path(data["raw_dir"]), split="validation")
    if train_dataset.category_labels != val_dataset.category_labels:
        raise ValueError("Training and validation splits define different categories")

    generator = torch.Generator().manual_seed(settings["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        collate_fn=collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(config["output_dir"]) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = weights_dir / "last.ckpt"
    best_checkpoint_path = weights_dir / "best.ckpt"

    # Peek the checkpoint's epoch (if resuming) the same way yolo26s-coco/yolov4t-loco
    # do, so a Colab disconnect mid-run can actually be continued instead of losing
    # everything but best.pt's weights. See those branches for why this dict form
    # (not the old weights-only best.pt) is what makes real resume possible.
    resume_path = {"last": last_checkpoint_path, "best": best_checkpoint_path}.get(resume_from)
    checkpoint = None
    start_epoch = 0
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        start_epoch = checkpoint["epoch"] + 1

    model = create_model(train_dataset.num_classes).to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = SGD(
        params,
        lr=settings["learning_rate"],
        momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )
    best_map = -1.0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except (ValueError, RuntimeError) as error:
            print(
                f"[resume] optimizer state didn't match new param groups, "
                f"starting fresh momentum: {error}"
            )
        best_map = checkpoint["best_map"]
        print(
            f"[resume] continuing from {resume_from}.ckpt, epoch {start_epoch + 1}, "
            f"best_map so far {best_map:.4f}"
        )

    for epoch in range(start_epoch, settings["epochs"]):
        current_lr = compute_lr(
            epoch,
            settings["learning_rate"],
            settings.get("warmup_epochs", 0),
            settings["epochs"],
            settings.get("lr_final_fraction", 1.0),
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{settings['epochs']}", leave=False)
        for step, (images_L, targets_L) in enumerate(progress):
            images_L = [image_CHW.to(device) for image_CHW in images_L]
            targets_L = [
                {key: value.to(device) for key, value in target.items()} for target in targets_L
            ]
            losses = model(images_L, targets_L)
            loss = sum(losses.values())

            if not torch.isfinite(loss):
                # Same guard as the other branches: skip a degenerate batch instead
                # of poisoning every weight with a backward() through a NaN/inf loss.
                print(f"  [warn] non-finite loss ({loss.item()}) at step {step} -- skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

        epoch_loss /= len(train_loader)
        map50 = compute_map50(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{settings['epochs']}  loss {epoch_loss:.4f}  "
            f"val_mAP50 {map50:.4f}  lr {current_lr:.6f}  {epoch_seconds:.0f}s"
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_mAP50": map50,
                    "epoch_seconds": epoch_seconds,
                }
            )
        if map50 > best_map:
            best_map = map50
            torch.save(model.state_dict(), weights_dir / "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_map": best_map,
                },
                best_checkpoint_path,
            )
            print(f"  saved new best: {best_map:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_map": best_map,
            },
            last_checkpoint_path,
        )

    print(f"[done] best weights: {weights_dir / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--resume",
        choices=["last", "best"],
        default=None,
        help="continue from weights/last.ckpt or weights/best.ckpt if present",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    wandb_config = config["wandb"]

    if not wandb_config["enabled"]:
        train(config, run=None, resume_from=args.resume)
        return

    with wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        config=config,
    ) as run:
        train(config, run, resume_from=args.resume)


if __name__ == "__main__":
    main()
