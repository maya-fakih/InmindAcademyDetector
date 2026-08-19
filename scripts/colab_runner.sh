#!/usr/bin/env bash
# Paste into a single Colab cell as:  !bash scripts/colab_runner.sh
# Or run from a terminal cell after `%cd InmindAcademyDetector`.
#
# Usage:
#   !bash scripts/colab_runner.sh          -- fresh run (default)
#   !bash scripts/colab_runner.sh resume   -- resume from weights/last.ckpt on Drive
#
# A fresh run never touches an existing checkpoint's training state, but if a
# previous run's weights/ directory exists it gets moved aside first (renamed
# with a timestamp) instead of being silently overwritten -- so switching
# training regimes (e.g. adding freeze/warmup) doesn't accidentally resume
# from or clobber a run trained under the old settings.
#
# What it does, in order:
#   1. Clone/pull the repo (idempotent — safe to rerun after a disconnect).
#   2. Install dependencies with uv.
#   3. Mount Drive; dataset + checkpoints persist under
#      MyDrive/InmindAcademyDetector-${BRANCH}/, not the ephemeral VM disk.
#   4. Train for config.yaml's epoch count -- resuming only if "resume" was
#      passed as the first argument and weights/last.ckpt exists on Drive.
#   5. Evaluate the best checkpoint on subsets 1/4 once training finishes.
#
# Everything is logged to train.log / eval.log so you can check progress or
# read the full output in the morning even if the cell itself scrolled away.

set -euo pipefail

MODE="${1:-fresh}"
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
    echo "[error] first argument must be 'resume' or omitted (defaults to fresh)" >&2
    exit 1
fi
BRANCH="${2:-yolo26s-coco}"

# This repo is private — set GITHUB_TOKEN before running, e.g. in a Colab cell:
#   from google.colab import userdata
#   import os; os.environ["GITHUB_TOKEN"] = userdata.get("GH_TOKEN")
# Use a fresh, read-only, repo-scoped token — don't reuse one you've already
# shared elsewhere, and revoke it once the run is done.
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a repo-scoped GitHub token before running this script}"

REPO_URL="https://${GITHUB_TOKEN}@github.com/maya-fakih/InmindAcademyDetector.git"
REPO_DIR="InmindAcademyDetector"

# --- 1. Clone or update -----------------------------------------------------
if [[ -f "train.py" && -f "config.yaml" ]]; then
    # Already running from inside a checkout of this repo (e.g. the calling
    # cell already cloned it) -- don't clone again into a nested subfolder.
    echo "[setup] already inside a repo checkout, skipping clone"
elif [[ -d "$REPO_DIR/.git" ]]; then
    echo "[setup] repo already present, pulling latest..."
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
    cd "$REPO_DIR"
else
    echo "[setup] cloning repo..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# --- 2. Install dependencies -------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    pip install -q uv
fi
echo "[setup] syncing dependencies..."
uv sync --quiet

# --- 3. Dataset + checkpoints on Drive (survives disconnects) ---------------
# NOTE: drive.mount() must be called from an actual Colab notebook cell, not
# from a subprocess spawned by this script -- it needs the Colab kernel
# connection, which a `python3 -c ...` child process doesn't have. Mount
# Drive yourself in the cell BEFORE running this script; we just check it.
if [[ ! -d "/content/drive/MyDrive" ]]; then
    echo "[error] Drive isn't mounted. Run this in a cell first, then rerun this script:" >&2
    echo "  from google.colab import drive; drive.mount('/content/drive')" >&2
    exit 1
fi
DRIVE_ROOT="/content/drive/MyDrive/InmindAcademyDetector-${BRANCH}"
DATA_DIR="${DRIVE_ROOT}/dataset"
RUNS_DIR="${DRIVE_ROOT}/runs/baseline"
mkdir -p "$DATA_DIR" "$RUNS_DIR"

echo "[setup] checking LOCO dataset..."
bash scripts/download_loco.sh "$DATA_DIR"

# --- 3b. Re-cluster yolov4-tiny anchors on LOCO (fresh runs only) -----------
# This was a documented prerequisite (see Handoff.md) that never actually got
# run -- the cfg still had stock COCO anchors through the entire 50-epoch run
# that only reached ~0.05 test mAP@0.5. Auto-running it here on every fresh
# run removes the "forgot the prerequisite" failure mode entirely. Skipped on
# resume deliberately: swapping anchor shapes out from under an
# already-training checkpoint would invalidate what its head learned to
# predict relative to the old anchors, which is its own kind of instability.
# Guarded on the cfg file existing so this is a no-op for other branches ($2).
if [[ "$MODE" == "fresh" && -f "vendor/pytorch_yolov4/yolov4-tiny.cfg" ]]; then
    echo "[setup] re-clustering yolov4-tiny anchors on LOCO (subsets 2/3/5 only)..."
    uv run python scripts/cluster_anchors.py --raw-dir "$DATA_DIR" --write
fi

# --- 3c. yolov4-tiny COCO-pretrained backbone (persisted on Drive) ----------
# scripts/download_yolov4tiny_backbone.sh exists but was never wired into this
# runner, so every fresh Colab VM hit FileNotFoundError on
# weights/yolov4-tiny.conv.29 at train.py startup. Downloading it straight to
# an ephemeral VM path would also mean re-downloading every session; instead
# it goes under DRIVE_ROOT like the dataset and checkpoints, and
# colab_config.yaml's model.checkpoint is rewritten to point at it.
# Guarded on scripts/download_yolov4tiny_backbone.sh existing, so this is a
# no-op for every other branch.
BACKBONE_WEIGHTS_PATH=""
if [[ -f "scripts/download_yolov4tiny_backbone.sh" ]]; then
    BACKBONE_DIR="${DRIVE_ROOT}/pretrained_weights"
    echo "[setup] checking yolov4-tiny COCO-pretrained backbone..."
    bash scripts/download_yolov4tiny_backbone.sh "$BACKBONE_DIR"
    BACKBONE_WEIGHTS_PATH="${BACKBONE_DIR}/yolov4-tiny.conv.29"
fi

python3 -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
config['data']['raw_dir'] = '${DATA_DIR}'
config['output_dir'] = '${RUNS_DIR}'
backbone_path = '${BACKBONE_WEIGHTS_PATH}'
if backbone_path and 'model' in config and 'checkpoint' in config.get('model', {}):
    config['model']['checkpoint'] = backbone_path
with open('colab_config.yaml', 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False)
"

# --- 4. Train (resume only if explicitly requested) --------------------------
RESUME_FLAG=""
if [[ "$MODE" == "resume" ]]; then
    if [[ ! -f "${RUNS_DIR}/weights/last.ckpt" ]]; then
        echo "[error] resume requested but no ${RUNS_DIR}/weights/last.ckpt found" >&2
        exit 1
    fi
    echo "[train] resume requested — continuing from last.ckpt"
    RESUME_FLAG="--resume last"
elif [[ -d "${RUNS_DIR}/weights" ]]; then
    BACKUP_DIR="${RUNS_DIR}/weights-backup-$(date +%Y%m%d-%H%M%S)"
    echo "[train] fresh run requested but weights/ already exists — moving it to"
    echo "        $BACKUP_DIR instead of overwriting"
    mv "${RUNS_DIR}/weights" "$BACKUP_DIR"
fi

echo "[train] starting training (see train.log for full output)..."
export MPLBACKEND=Agg  # Colab exports MPLBACKEND=matplotlib_inline's backend globally, but
                        # that package isn't in this project's uv venv -- matplotlib (pulled
                        # in by torchmetrics) then fails to import. Nothing here needs
                        # interactive plots, so force a headless backend instead.
uv run train.py --config colab_config.yaml $RESUME_FLAG 2>&1 | tee train.log

# --- 5. Evaluate the best checkpoint -----------------------------------------
echo "[eval] evaluating ${RUNS_DIR}/weights/best.pt on subsets 1/4..."
uv run eval.py --config colab_config.yaml --weights "${RUNS_DIR}/weights/best.pt" 2>&1 | tee eval.log

echo "[done] training + eval finished. See train.log and eval.log for full output."