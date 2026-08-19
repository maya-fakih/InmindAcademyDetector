"""Build a Markdown report (loss/mAP/LR curves, eval metrics, sample predictions)
for one trained run, without modifying train.py/eval.py/predict.py.

Reads what colab_runner.sh already produces (train.log, and history.json if the
branch's train.py writes one) and shells out to the existing eval.py / predict.py
so the numbers/images are identical to what those scripts would print on their
own -- this file only assembles them into results/<run-name>/report.md.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

METRIC_LINE = re.compile(r"^([\w /@.-]+?):\s+(.+)$")
# The optional " lr <value>" group covers branches whose train.py prints the
# current learning rate on the epoch line (yolov4t-loco, yolo26s-small-coco,
# frcnn-mobilenetv3-augment, frcnn-amir-recipe all do). Older logs without it
# still match -- group 5 is simply absent.
EPOCH_LINE = re.compile(
    r"epoch (\d+)/(\d+)\s+loss ([\d.]+)\s+val_mAP50 ([\d.]+)(?:\s+lr ([\d.eE+-]+))?\s+(\d+)s"
)


def parse_train_log(log_path: Path) -> list[dict]:
    """Extract per-epoch (epoch, loss, val_mAP50, lr?, seconds) rows from train.log."""
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text().splitlines():
        match = EPOCH_LINE.search(line)
        if match:
            epoch, total, loss, map50, lr, seconds = match.groups()
            row = {
                "epoch": int(epoch),
                "loss": float(loss),
                "val_mAP50": float(map50),
                "seconds": int(seconds),
            }
            if lr is not None:
                row["lr"] = float(lr)
            rows.append(row)
    return rows


def parse_history_json(history_path: Path) -> list[dict]:
    """Extract per-epoch rows from a history.json written by train.py, if present.

    Preferred over train.log regex parsing when available -- it's structured
    data straight from the training loop (includes ``lr`` whenever the branch
    logs it) rather than something scraped back out of printed text.
    """
    if not history_path.is_file():
        return []
    data = json.loads(history_path.read_text(encoding="utf-8"))
    rows = []
    for entry in data.get("history", []):
        row = {
            "epoch": entry["epoch"],
            "loss": entry.get("train_loss"),
            "val_mAP50": entry.get("val_mAP50"),
        }
        if "epoch_seconds" in entry:
            row["seconds"] = entry["epoch_seconds"]
        if "lr" in entry:
            row["lr"] = entry["lr"]
        if "test_mAP50" in entry:
            row["test_mAP50"] = entry["test_mAP50"]
        rows.append(row)
    return rows


def write_report(
    run_name: str,
    out_dir: Path,
    rows: list[dict],
    metrics: dict[str, str],
    curve_path: Path | None,
    prediction_paths: list[Path],
) -> Path:
    """Assemble the parsed history/metrics/images into results/<run-name>/report.md."""
    lines = [f"# {run_name}", ""]

    lines += ["## Eval metrics (via eval.py)", ""]
    for key, value in metrics.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")

    if curve_path is not None:
        lines += ["## Training curves", "", f"![loss, mAP, and LR curves]({curve_path.name})", ""]
    if rows:
        has_lr = any("lr" in r for r in rows)
        header = (
            "| epoch | train loss | val mAP@0.5 |" + (" lr |" if has_lr else "") + " time (s) |"
        )
        sep = "|---|---|---|" + ("---|" if has_lr else "") + "---|"
        lines += [header, sep]
        for r in rows:
            lr_cell = f" {r['lr']:.6f} |" if has_lr else ""
            seconds = r.get("seconds", "")
            lines.append(
                f"| {r['epoch']} | {r['loss']:.4f} | {r['val_mAP50']:.4f} |{lr_cell} {seconds} |"
            )
        lines.append("")

    if prediction_paths:
        lines += ["## Sample predictions (ground truth vs. prediction)", ""]
        lines += [f"![{p.stem}](predictions/{p.name})" for p in prediction_paths]
        lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def run_eval(weights: Path, config: Path) -> dict[str, str]:
    """Run the repo's own eval.py unchanged and parse its printed metric lines.

    Reuses eval.py directly (subprocess) so the numbers in the report are
    exactly what eval.py itself reports -- nothing is recomputed here.
    """
    result = subprocess.run(
        [sys.executable, "eval.py", "--weights", str(weights), "--config", str(config)],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = {}
    for line in result.stdout.splitlines():
        match = METRIC_LINE.match(line.strip())
        if match:
            metrics[match.group(1).strip()] = match.group(2).strip()
    return metrics


def run_predict(weights: Path, config: Path, output_dir: Path, num_images: int) -> list[Path]:
    """Run the repo's own predict.py unchanged and return the PNGs it wrote."""
    subprocess.run(
        [
            sys.executable,
            "predict.py",
            "--weights",
            str(weights),
            "--config",
            str(config),
            "--num-images",
            str(num_images),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    return sorted(output_dir.glob("prediction_*.png"))


def plot_curves(rows: list[dict], output_path: Path) -> None:
    """Save a loss/mAP/(optional LR)-vs-epoch figure. No-op if there's no history."""
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr_rows = [r for r in rows if "lr" in r]
    panels = 3 if lr_rows else 2
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4))
    ax_loss, ax_map = axes[0], axes[1]

    epochs = [r["epoch"] for r in rows]
    ax_loss.plot(epochs, [r["loss"] for r in rows], marker="o")
    ax_loss.set(xlabel="epoch", ylabel="train loss", title="training loss")
    ax_map.plot(epochs, [r["val_mAP50"] for r in rows], marker="o", color="tab:green")
    ax_map.set(xlabel="epoch", ylabel="val mAP@0.5", title="validation mAP@0.5")

    test_rows = [r for r in rows if "test_mAP50" in r]
    if test_rows:
        ax_map.plot(
            [r["epoch"] for r in test_rows],
            [r["test_mAP50"] for r in test_rows],
            marker="x",
            linestyle="--",
            color="tab:red",
            label="held-out test mAP@0.5 (subsets 1/4, monitoring only)",
        )
        ax_map.legend(fontsize=7)

    if lr_rows:
        ax_lr = axes[2]
        ax_lr.plot(
            [r["epoch"] for r in lr_rows],
            [r["lr"] for r in lr_rows],
            marker="o",
            color="tab:orange",
        )
        ax_lr.set(xlabel="epoch", ylabel="learning rate", title="learning rate schedule")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="e.g. malek_rcnn_baseline")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--log-file", type=Path, default=Path("train.log"))
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Path to history.json. Defaults to <weights>/../../history.json (the "
        "output_dir/history.json layout train.py writes on branches that log one). "
        "Falls back to parsing --log-file if not found.",
    )
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    out_dir = args.results_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = args.history_file or (args.weights.parent.parent / "history.json")
    rows = parse_history_json(history_path) or parse_train_log(args.log_file)
    curve_path = out_dir / "curves.png"
    plot_curves(rows, curve_path)
    if not rows:
        curve_path = None

    metrics = run_eval(args.weights, args.config)
    prediction_paths = run_predict(
        args.weights, args.config, out_dir / "predictions", args.num_images
    )

    report_path = write_report(args.run_name, out_dir, rows, metrics, curve_path, prediction_paths)
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
