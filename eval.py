"""Evaluate detector accuracy and batch-size-one latency."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import LocoDataset, collate_fn
from metrics.accuracy import compute_map50
from metrics.efficiency import count_parameters, measure_gflops
from models.yolo_wrapper import create_yolo_model
from utils.config import load_config


def synchronize(device: torch.device) -> None:
    """Wait for pending CUDA operations."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_latency_ms(model: nn.Module, sample_CHW: Tensor, device: torch.device) -> float:
    """Return median batch-size-one latency in milliseconds over 10 runs after two warmups."""
    model.eval()
    for _ in range(2):
        model([sample_CHW])
    timings_ms = []
    for _ in range(10):
        synchronize(device)
        start = time.perf_counter()
        model([sample_CHW])
        synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000)

    return statistics.median(timings_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")

    config: dict[str, Any] = load_config(args.config)
    data = config["data"]
    dataset = LocoDataset(Path(data["raw_dir"]), split="test")
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_fn,
    )
    # Read the architecture that was ACTUALLY trained, stamped by train.py next
    # to the weights (best.arch.json), rather than guessing from config.yaml.
    # A guessed default is exactly why every branch's eval used to silently
    # build yolo26n regardless of which weights file was passed in -- config.yaml
    # at eval time can be a different checkout than the one that produced these
    # weights (e.g. on Drive), so it is not a reliable source of truth here.
    arch_path = args.weights.parent / "best.arch.json"
    if arch_path.is_file():
        arch_meta = json.loads(arch_path.read_text())
        model_checkpoint = arch_meta["checkpoint"]
    else:
        model_checkpoint = config.get("model", {}).get("checkpoint", "yolo26n.pt")
        print(
            f"[warn] {arch_path.name} not found next to {args.weights.name} -- "
            f"falling back to config.yaml's model.checkpoint ({model_checkpoint!r}). "
            "This may not be the architecture these weights were actually trained "
            "with if config.yaml has changed since. Re-train to get a stamped "
            "best.arch.json and remove this ambiguity."
        )
    print(f"[eval] architecture: {model_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_yolo_model(dataset.num_classes, checkpoint=model_checkpoint)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    except RuntimeError as exc:
        parser.error(
            f"Weights in {args.weights} do not match the '{model_checkpoint}' architecture "
            f"(from {arch_path.name if arch_path.is_file() else 'config.yaml fallback'}). "
            f"This almost always means the wrong architecture was inferred -- check "
            f"model.checkpoint in the config this run actually used. Original error: {exc}"
        )
    model.to(device)

    # Both efficiency numbers use the same fixed test image so submissions compare.
    sample_CHW = dataset[0][0]
    map50 = compute_map50(model, loader, device)
    latency = measure_latency_ms(model, sample_CHW.to(device), device)
    parameters, trainable = count_parameters(model)
    gflops = measure_gflops(model, sample_CHW, device)

    print(f"mAP@0.5:                {map50:.4f}")
    print(f"parameters:             {parameters:,} ({parameters / 1e6:.2f} M)")
    print(f"  of which trainable:   {trainable:,}")
    print(f"GFLOPs per image:       {gflops:.3f}")
    print(f"median batch-1 latency: {latency:.2f} ms ({device}) — machine-dependent, not graded")


if __name__ == "__main__":
    main()
