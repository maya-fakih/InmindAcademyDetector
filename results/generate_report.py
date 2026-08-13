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
        lines += ["## Training curves", "", f"![loss and mAP curves]({curve_path.name})", ""]
    if rows:
        lines += [
            "| epoch | train loss | val mAP@0.5 | time (s) |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r['epoch']} | {r['loss']:.4f} | {r['val_mAP50']:.4f} | {r['seconds']} |"
            for r in rows
        ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="e.g. malek_rcnn_baseline")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--log-file", type=Path, default=Path("train.log"))
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    out_dir = args.results_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_train_log(args.log_file)
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
