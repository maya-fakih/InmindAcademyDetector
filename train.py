"""Train the LOCO Faster R-CNN baseline."""

import argparse
import json
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
from models.yolo_wrapper import create_yolo_model
from utils.config import load_config


def compute_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    final_fraction: float,
    schedule_start_epoch: int = 0,
    peak_lr_fraction: float = 1.0,
) -> float:
    """Linear warmup for `warmup_epochs`, then cosine decay to `final_fraction` of base_lr.

    ``schedule_start_epoch``/``peak_lr_fraction`` support an SGDR-style warm
    restart when resuming into an *extended* schedule (settings["epochs"]
    raised past what the checkpoint was originally trained for): instead of
    naively resuming into the old absolute cosine curve -- which, this deep
    into decay, would jump LR from near-zero straight to ~50% of base_lr in a
    single step -- warmup/decay are computed relative to the restart point,
    ramping over `warmup_epochs` up to only `peak_lr_fraction * base_lr`
    rather than the abrupt full jump. Left at their defaults (0, 1.0), this
    is identical to the original schedule.
    """
    relative_epoch = epoch - schedule_start_epoch
    relative_total = total_epochs - schedule_start_epoch
    peak_lr = base_lr * peak_lr_fraction
    if warmup_epochs > 0 and relative_epoch < warmup_epochs:
        return peak_lr * (relative_epoch + 1) / warmup_epochs
    span = max(1, relative_total - warmup_epochs)
    progress = min((relative_epoch - warmup_epochs) / span, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * final_fraction + (peak_lr - peak_lr * final_fraction) * cosine


def set_backbone_frozen(model: torch.nn.Module, num_layers: int, frozen: bool) -> None:
    """Freeze or unfreeze the first `num_layers` layers of the underlying YOLO backbone.

    Layer indices match the printed model summary (Conv/C3k2/.../C2PSA at index 10
    is the last backbone block; freeze_backbone_layers=11 covers layers 0-10).
    """
    for index, layer in enumerate(model.model.model):
        if index >= num_layers:
            break
        for parameter in layer.parameters():
            parameter.requires_grad = not frozen


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
    # Validation must stay clean -- it's what best.pt is selected on, so
    # augmenting it would make "best" mean "best on artificially varied
    # backgrounds" instead of "best on this warehouse's real val images".
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
    # Held-out test set (subsets 1/4) -- never trained/tuned on. Checked periodically
    # during training (see `test_eval_every` below) purely so long unattended runs
    # leave a trail comparing val-selection mAP against true held-out mAP, in case
    # the session dies before a human gets to run eval.py by hand. Same pattern as
    # yolov4t-loco; never used to pick best.pt, which stays selected on val_loader.
    test_dataset = LocoDataset(Path(data["raw_dir"]), split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=collate_fn,
    )
    test_eval_every = settings.get("test_eval_every", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(config["output_dir"]) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = weights_dir / "last.ckpt"
    best_checkpoint_path = weights_dir / "best.ckpt"

    # Peek the checkpoint's epoch (if resuming) before deciding whether to freeze the
    # backbone -- freezing only makes sense for the initial epochs of a fresh run.
    resume_path = {"last": last_checkpoint_path, "best": best_checkpoint_path}.get(resume_from)
    checkpoint = None
    start_epoch = 0
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        start_epoch = checkpoint["epoch"] + 1

    # Read the intended architecture from config instead of relying on
    # create_yolo_model()'s default ("yolo26n.pt") -- that silent default is
    # exactly why yolo26s-coco and yolo26s-small-coco both trained nano
    # despite their names/config. num_classes only, not this, so this got
    # missed once already; not again.
    model_checkpoint = config.get("model", {}).get("checkpoint", "yolo26n.pt")
    model = create_yolo_model(train_dataset.num_classes, checkpoint=model_checkpoint).to(device)

    freeze_epochs = settings.get("freeze_backbone_epochs", 0)
    freeze_layers = settings.get("freeze_backbone_layers", 0)
    backbone_frozen = freeze_epochs > 0 and start_epoch < freeze_epochs
    if freeze_layers > 0:
        set_backbone_frozen(model, freeze_layers, frozen=backbone_frozen)

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

    # Warm restart (opt-in): only kicks in when resuming AND config.yaml sets
    # restart_peak_lr_fraction -- i.e. you've raised train.epochs past what this
    # checkpoint already finished and want a deliberate SGDR-style LR bump to
    # try to escape wherever the cosine decay had it converging, rather than
    # silently resuming into the tail of the *old* absolute schedule (which
    # would jump LR from near-zero to ~50% of base_lr in one step). See
    # compute_lr's docstring. No effect on a plain interrupted-run resume.
    schedule_start_epoch = 0
    peak_lr_fraction = 1.0
    warmup_epochs = settings.get("warmup_epochs", 0)
    restart_peak_lr_fraction = settings.get("restart_peak_lr_fraction")
    if checkpoint is not None and restart_peak_lr_fraction is not None:
        schedule_start_epoch = start_epoch
        peak_lr_fraction = restart_peak_lr_fraction
        warmup_epochs = settings.get("restart_warmup_epochs", warmup_epochs)
        print(
            f"[restart] warm-restarting LR at epoch {start_epoch + 1}: ramping to "
            f"{peak_lr_fraction * settings['learning_rate']:.6f} over {warmup_epochs} "
            f"epochs, then cosine-decaying to epoch {settings['epochs']}"
        )

    for epoch in range(start_epoch, settings["epochs"]):
        if freeze_layers > 0 and backbone_frozen and epoch >= freeze_epochs:
            print(f"  [freeze] unfreezing backbone at epoch {epoch + 1}")
            set_backbone_frozen(model, freeze_layers, frozen=False)
            backbone_frozen = False
            optimizer = SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=settings["learning_rate"],
                momentum=settings["momentum"],
                weight_decay=settings["weight_decay"],
            )

        current_lr = compute_lr(
            epoch,
            settings["learning_rate"],
            warmup_epochs,
            settings["epochs"],
            settings.get("lr_final_fraction", 1.0),
            schedule_start_epoch=schedule_start_epoch,
            peak_lr_fraction=peak_lr_fraction,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        component_totals: dict[str, float] = {}
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{settings['epochs']}", leave=False)
        for step, (images_L, targets_L) in enumerate(progress):
            images_L = [image_CHW.to(device) for image_CHW in images_L]
            targets_L = [
                {key: value.to(device) for key, value in target.items()} for target in targets_L
            ]
            losses = model(images_L, targets_L)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            for name, value in losses.items():
                component_totals[name] = component_totals.get(name, 0.0) + value.item()
            progress.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

        epoch_loss /= len(train_loader)
        component_avgs = {
            name: total / len(train_loader) for name, total in component_totals.items()
        }
        map50 = compute_map50(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start
        breakdown = "  ".join(f"{name} {value:.4f}" for name, value in component_avgs.items())
        print(
            f"epoch {epoch + 1}/{settings['epochs']}  loss {epoch_loss:.4f}  "
            f"val_mAP50 {map50:.4f}  lr {current_lr:.6f}  {epoch_seconds:.0f}s"
        )
        print(f"  loss breakdown: {breakdown}")
        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_mAP50": map50,
                    "epoch_seconds": epoch_seconds,
                }
            )
        if test_eval_every > 0 and (
            (epoch + 1) % test_eval_every == 0 or (epoch + 1) == settings["epochs"]
        ):
            test_map50 = compute_map50(model, test_loader, device)
            print(f"  [TEST subsets 1/4] epoch {epoch + 1}: test_mAP50 {test_map50:.4f}")
            if run is not None:
                run.log({"epoch": epoch + 1, "test_mAP50": test_map50})

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

    # Stamp which architecture these weights actually are. eval.py reads this
    # instead of re-guessing from config.yaml -- config.yaml on Drive/another
    # checkout can drift from what a given best.pt was actually trained with,
    # and a silent default has already caused one round of "every branch
    # secretly trained nano" (see yolo26s-small-coco history).
    arch_meta = {"checkpoint": model_checkpoint, "num_classes": train_dataset.num_classes}
    (weights_dir / "best.arch.json").write_text(json.dumps(arch_meta))

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
