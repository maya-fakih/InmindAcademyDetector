"""Build a Markdown report (loss/mAP curve, eval metrics, sample predictions)
for one trained run, without modifying train.py/eval.py/predict.py.

Reads what colab_runner.sh already produces (train.log) and shells out to
the existing eval.py / predict.py so the numbers/images are identical to
what those scripts would print on their own -- this file only assembles
them into results/<run-name>/report.md.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

METRIC_LINE = re.compile(r"^([\w /@.-]+?):\s+(.+)$")
EPOCH_LINE = re.compile(r"epoch (\d+)/(\d+)\s+loss ([\d.]+)\s+val_mAP50 ([\d.]+)\s+(\d+)s")


def parse_train_log(log_path: Path) -> list[dict]:
    """Extract per-epoch (epoch, loss, val_mAP50, seconds) rows from train.log."""
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text().splitlines():
        match = EPOCH_LINE.search(line)
        if match:
            epoch, total, loss, map50, seconds = match.groups()
            rows.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss),
                    "val_mAP50": float(map50),
                    "seconds": int(seconds),
                }
            )
    return rows


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
    """Save a two-panel loss/mAP-vs-epoch figure. No-op if there's no history."""
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [r["loss"] for r in rows], marker="o")
    ax_loss.set(xlabel="epoch", ylabel="train loss", title="training loss")
    ax_map.plot(epochs, [r["val_mAP50"] for r in rows], marker="o", color="tab:green")
    ax_map.set(xlabel="epoch", ylabel="val mAP@0.5", title="validation mAP@0.5")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
