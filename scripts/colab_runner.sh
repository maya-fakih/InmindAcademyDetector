#!/usr/bin/env bash
# Paste into a single Colab cell as:  !bash scripts/colab_runner.sh
# Or run from a terminal cell after `%cd InmindAcademyDetector`.
#
# What it does, in order:
#   1. Clone/pull the repo (idempotent — safe to rerun after a disconnect).
#   2. Install dependencies with uv.
#   3. Mount Drive; dataset + checkpoints persist under
#      MyDrive/InmindAcademyDetector-${BRANCH}/, not the ephemeral VM disk.
#   4. Train for config.yaml's epoch count, auto-resuming from
#      weights/last.ckpt on Drive if a previous run left one behind — so
#      if Colab disconnects overnight, rerunning this same cell picks up
#      where it left off instead of starting over.
#   5. Evaluate the best checkpoint on subsets 1/4 once training finishes.
#
# Everything is logged to train.log / eval.log so you can check progress or
# read the full output in the morning even if the cell itself scrolled away.

set -euo pipefail

# This repo is private — set GITHUB_TOKEN before running, e.g. in a Colab cell:
#   from google.colab import userdata
#   import os; os.environ["GITHUB_TOKEN"] = userdata.get("GH_TOKEN")
# Use a fresh, read-only, repo-scoped token — don't reuse one you've already
# shared elsewhere, and revoke it once the run is done.
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a repo-scoped GitHub token before running this script}"

REPO_URL="https://${GITHUB_TOKEN}@github.com/maya-fakih/InmindAcademyDetector.git"
BRANCH="yolo26s-coco"
REPO_DIR="InmindAcademyDetector"

# --- 1. Clone or update -----------------------------------------------------
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[setup] repo already present, pulling latest..."
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
    echo "[setup] cloning repo..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

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

python3 -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
config['data']['raw_dir'] = '${DATA_DIR}'
config['output_dir'] = '${RUNS_DIR}'
with open('colab_config.yaml', 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False)
"

# --- 4. Train (auto-resume if a checkpoint from a previous attempt exists) --
RESUME_FLAG=""
if [[ -f "${RUNS_DIR}/weights/last.ckpt" ]]; then
    echo "[train] found an existing last.ckpt on Drive — resuming instead of starting over"
    RESUME_FLAG="--resume last"
fi

echo "[train] starting training (see train.log for full output)..."
uv run train.py --config colab_config.yaml $RESUME_FLAG 2>&1 | tee train.log

# --- 5. Evaluate the best checkpoint -----------------------------------------
echo "[eval] evaluating ${RUNS_DIR}/weights/best.pt on subsets 1/4..."
uv run eval.py --config colab_config.yaml --weights "${RUNS_DIR}/weights/best.pt" 2>&1 | tee eval.log

echo "[done] training + eval finished. See train.log and eval.log for full output."
