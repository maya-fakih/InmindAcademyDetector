"""Draw ground-truth and predicted boxes side by side so students can debug by looking."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import torch
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch import Tensor, nn

from dataset import LocoDataset
from models.yolo_wrapper import create_yolo_model
from utils.config import load_config


def draw_boxes(
    ax: Axes,
    boxes_NQ: Tensor,
    labels_N: Tensor,
    label_names: dict[int, str],
    scores_N: Tensor | None = None,
) -> None:
    """Outline each box on ``ax``, coloured by class and labelled with its name."""
    colors = colormaps["tab10"]
    for i, (box, label) in enumerate(zip(boxes_NQ.tolist(), labels_N.tolist(), strict=True)):
        x1, y1, x2, y2 = box
        color = colors(label % colors.N)
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        )
        name = label_names[label]
        text = name if scores_N is None else f"{name} {scores_N[i]:.2f}"
        ax.text(x1, y1 - 5, text, color=color, fontsize=8, weight="bold")


@torch.no_grad()
def predict_image(
    model: nn.Module, image_CHW: Tensor, device: torch.device, threshold: float
) -> dict[str, Tensor]:
    """Run the model on one image and drop predictions scoring below ``threshold``."""
    prediction = model([image_CHW.to(device)])[0]
    keep = prediction["scores"] >= threshold
    return {key: value[keep].cpu() for key, value in prediction.items()}


def main() -> None:
    """Load a checkpoint and save ground-truth/prediction figures for validation images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predictions"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config: dict[str, Any] = load_config(args.config)
    dataset = LocoDataset(Path(config["data"]["raw_dir"]), split="validation")
    if not 0 <= args.index < len(dataset):
        parser.error(f"Index must be between 0 and {len(dataset) - 1}")
    if args.num_images < 1:
        parser.error("Number of images must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_yolo_model(dataset.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop = min(args.index + args.num_images, len(dataset))
    for index in range(args.index, stop):
        image_CHW, target = dataset[index]
        prediction = predict_image(model, image_CHW, device, args.score_threshold)
        image_HWC = image_CHW.permute(1, 2, 0).cpu().numpy()

        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(12, 6))
        for ax in (ax_gt, ax_pred):
            ax.imshow(image_HWC)
            ax.axis("off")
        draw_boxes(ax_gt, target["boxes"], target["labels"], dataset.label_names)
        draw_boxes(
            ax_pred,
            prediction["boxes"],
            prediction["labels"],
            dataset.label_names,
            prediction["scores"],
        )
        ax_gt.set_title("ground truth")
        ax_pred.set_title("prediction")
        fig.suptitle(
            f"image {index}: {len(target['labels'])} ground truth boxes, "
            f"{len(prediction['labels'])} predicted boxes"
        )

        fig.savefig(args.output_dir / f"prediction_{index}.png", bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close(fig)

    print(f"Wrote {stop - args.index} prediction images to {args.output_dir}")


if __name__ == "__main__":
    main()
