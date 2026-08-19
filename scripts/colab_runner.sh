#!/usr/bin/env bash
# Paste into a single Colab cell as:  !bash scripts/colab_runner.sh
# Or run from a terminal cell after `%cd InmindAcademyDetector`.
#
# Usage:
#   !bash scripts/colab_runner.sh                             -- auto: resumes yolo26s-coco
#                                                                 if a checkpoint exists on
#                                                                 Drive, else starts fresh
#   !bash scripts/colab_runner.sh resume                      -- resume yolo26s-coco from last.ckpt
#   !bash scripts/colab_runner.sh resume-best                 -- fine-tune yolo26s-coco from best.ckpt
#   !bash scripts/colab_runner.sh fresh yolo26s-small-coco    -- explicitly discard and restart
#   !bash scripts/colab_runner.sh resume yolo26s-small-coco   -- resume another branch from last.ckpt
#   !bash scripts/colab_runner.sh resume-best yolo26s-small-coco -- fine-tune another branch from best.ckpt
#
# The only way to discard an existing checkpoint is to type "fresh" explicitly.
# Omitting the mode used to mean the same thing as typing "fresh" -- which is
# exactly the footgun that kept quietly restarting runs from scratch (each
# restart moving the previous weights/ into a weights-backup-<timestamp>/
# folder instead of erroring, so it looked "successful" every time while
# silently throwing away all prior training). Now, omitting it auto-detects:
# if weights/last.ckpt is already there, it resumes; only an empty weights/
# dir (a genuinely first run) proceeds fresh with nothing to discard.
#
# What it does, in order:
#   1. Clone/pull the repo (idempotent — safe to rerun after a disconnect).
#   2. Install dependencies with uv.
#   3. Mount Drive; dataset + checkpoints persist under
#      MyDrive/InmindAcademyDetector-${BRANCH}/, not the ephemeral VM disk.
#   4. Train for config.yaml's epoch count, resuming/fine-tuning per MODE above.
#   5. Evaluate the best checkpoint on subsets 1/4 once training finishes.
#
# Everything is logged to train.log / eval.log so you can check progress or
# read the full output in the morning even if the cell itself scrolled away.

set -euo pipefail

MODE="${1:-auto}"
if [[ "$MODE" != "fresh" && "$MODE" != "resume" && "$MODE" != "resume-best" && "$MODE" != "auto" ]]; then
    echo "[error] first argument must be 'resume', 'resume-best', 'fresh', or omitted (auto-detects)" >&2
    exit 1
fi

# This repo is private — set GITHUB_TOKEN before running, e.g. in a Colab cell:
#   from google.colab import userdata
#   import os; os.environ["GITHUB_TOKEN"] = userdata.get("GH_TOKEN")
# Use a fresh, read-only, repo-scoped token — don't reuse one you've already
# shared elsewhere, and revoke it once the run is done.
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a repo-scoped GitHub token before running this script}"

REPO_URL="https://${GITHUB_TOKEN}@github.com/maya-fakih/InmindAcademyDetector.git"
# Bug fixed here: this used to be hardcoded regardless of $2 -- harmless for
# the "already inside a checkout" clone-skip path, but DRIVE_ROOT below reads
# $BRANCH unconditionally, so any branch other than yolo26s-coco would read/
# write checkpoints from *that* branch's Drive folder instead of its own
# (silently mixing an incompatible-architecture checkpoint in, in the worst
# case). ${2:-yolo26s-coco} keeps old one-arg invocations working unchanged.
BRANCH="${2:-yolo26s-coco}"
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

python3 -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
config['data']['raw_dir'] = '${DATA_DIR}'
config['output_dir'] = '${RUNS_DIR}'
with open('colab_config.yaml', 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False)
"

# --- 4. Train (mode resolved above -- "auto" is decided now that RUNS_DIR is known) ---
if [[ "$MODE" == "auto" ]]; then
    if [[ -f "${RUNS_DIR}/weights/last.ckpt" ]]; then
        echo "[train] found an existing checkpoint at ${RUNS_DIR}/weights/last.ckpt --"
        echo "        resuming automatically (pass 'fresh' explicitly to discard it instead)"
        MODE="resume"
    else
        echo "[train] no existing checkpoint found -- starting fresh"
        MODE="fresh"
    fi
fi

RESUME_FLAG=""
if [[ "$MODE" == "resume" || "$MODE" == "resume-best" ]]; then
    CKPT_NAME="last.ckpt"
    RESUME_ARG="last"
    if [[ "$MODE" == "resume-best" ]]; then
        CKPT_NAME="best.ckpt"
        RESUME_ARG="best"
    fi
    if [[ ! -f "${RUNS_DIR}/weights/${CKPT_NAME}" ]]; then
        echo "[error] $MODE requested but no ${RUNS_DIR}/weights/${CKPT_NAME} found" >&2
        exit 1
    fi
    echo "[train] $MODE requested — continuing from ${CKPT_NAME}"
    RESUME_FLAG="--resume ${RESUME_ARG}"
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
