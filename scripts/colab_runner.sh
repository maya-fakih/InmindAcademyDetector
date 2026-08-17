#!/usr/bin/env bash
# Paste into a single Colab cell as:  !bash scripts/colab_runner.sh
# Or run from a terminal cell after `%cd InmindAcademyDetector`.
#
# Usage:
#   !bash scripts/colab_runner.sh                              -- fresh run (default)
#   !bash scripts/colab_runner.sh resume                        -- resume last.ckpt on Drive
#   !bash scripts/colab_runner.sh fresh <other-branch>           -- fresh run on another branch
#   !bash scripts/colab_runner.sh resume <other-branch>          -- resume another branch
#
# This branch's train.py has no --resume flag (the plain baseline never had
# one, and this branch didn't add one), so "resume" here is best-effort: it
# just checks weights/last.ckpt doesn't exist and warns rather than silently
# starting fresh over it. Add real --resume support to train.py before
# relying on interrupting/continuing a run.
#
# What it does, in order:
#   1. Clone/pull the repo (idempotent -- safe to rerun after a disconnect).
#   2. Install dependencies with uv.
#   3. Mount Drive; dataset + checkpoints persist under
#      MyDrive/InmindAcademyDetector-${BRANCH}/, not the ephemeral VM disk.
#   4. Train for config.yaml's epoch count.
#   5. Evaluate the best checkpoint on subsets 1/4 once training finishes.
#
# Everything is logged to train.log / eval.log so you can check progress or
# read the full output later even if the cell itself scrolled away.

set -euo pipefail

MODE="${1:-fresh}"
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
    echo "[error] first argument must be 'resume' or omitted (defaults to fresh)" >&2
    exit 1
fi
BRANCH="${2:-frcnn-mobilenetv3-augment}"

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

# --- 4. Train -----------------------------------------------------------------
if [[ "$MODE" == "resume" ]]; then
    echo "[warn] this branch's train.py has no --resume flag -- 'resume' here is" >&2
    echo "       currently a no-op beyond the check below. Add --resume support" >&2
    echo "       to train.py (see yolo26s-coco/yolov4t-loco for the pattern)" >&2
    echo "       before relying on it." >&2
    if [[ -f "${RUNS_DIR}/weights/best.pt" ]]; then
        echo "[error] weights/best.pt already exists at ${RUNS_DIR} and this script" >&2
        echo "        can't resume into it -- move it aside yourself if you want a" >&2
        echo "        fresh run, or add real resume support to train.py first." >&2
        exit 1
    fi
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
uv run train.py --config colab_config.yaml 2>&1 | tee train.log

# --- 5. Evaluate the best checkpoint -----------------------------------------
echo "[eval] evaluating ${RUNS_DIR}/weights/best.pt on subsets 1/4..."
uv run eval.py --config colab_config.yaml --weights "${RUNS_DIR}/weights/best.pt" 2>&1 | tee eval.log

echo "[done] training + eval finished. See train.log and eval.log for full output."
