"""Train the LOCO Faster R-CNN baseline."""

import argparse
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


def train(config: dict[str, Any], run: Run | None, resume_from: str | None = None) -> None:
    """Train the detector and save the best weights."""
    settings = config["train"]
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])

    data = config["data"]
    train_dataset = LocoDataset(Path(data["raw_dir"]), split="train")
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
    checkpoint = config.get("model", {}).get("checkpoint", "yolo26s.pt")
    model = create_yolo_model(train_dataset.num_classes, checkpoint=checkpoint).to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = SGD(
        params,
        lr=settings["learning_rate"],
        momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )
    weights_dir = Path(config["output_dir"]) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = weights_dir / "last.ckpt"
    best_checkpoint_path = weights_dir / "best.ckpt"
    best_map = -1.0
    start_epoch = 0

    resume_path = {"last": last_checkpoint_path, "best": best_checkpoint_path}.get(resume_from)
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint["best_map"]
        print(
            f"[resume] continuing from {resume_from}.ckpt, epoch {start_epoch + 1}, "
            f"best_map so far {best_map:.4f}"
        )

    for epoch in range(start_epoch, settings["epochs"]):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

        epoch_loss /= len(train_loader)
        map50 = compute_map50(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{settings['epochs']}  loss {epoch_loss:.4f}  "
            f"val_mAP50 {map50:.4f}  {epoch_seconds:.0f}s"
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
