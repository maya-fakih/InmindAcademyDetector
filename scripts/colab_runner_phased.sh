#!/usr/bin/env bash
# Paste into a single Colab cell as:  !bash scripts/colab_runner_phased.sh
#
# Same setup as scripts/colab_runner.sh (clone/pull, uv sync, mount Drive,
# download LOCO), but trains in phases of PHASE_EPOCHS epochs each, running
# a full `eval.py` on subsets 1/4 after every phase and resuming from
# last.ckpt for the next one -- so you get an honest eval.py number every
# 10 epochs without editing train.py's internals or juggling background
# subprocesses in a notebook cell.
#
# Usage:
#   !bash scripts/colab_runner_phased.sh                 -- frcnn-amir-recipe, 3x10=30 epochs
#   !bash scripts/colab_runner_phased.sh <branch>         -- another branch, same phasing
#   !bash scripts/colab_runner_phased.sh <branch> <n>      -- n epochs per phase instead of 10
#
# Everything is logged to train-phaseN.log / eval-phaseN.log so progress
# survives even if the cell output scrolls away.

set -euo pipefail

BRANCH="${1:-frcnn-amir-recipe}"
PHASE_EPOCHS="${2:-10}"
TOTAL_EPOCHS=30

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

export MPLBACKEND=Agg  # see scripts/colab_runner.sh for why this is needed here

# --- 4. Phased train -> eval -> resume ---------------------------------------
phase=1
target_epochs=0
while [[ "$target_epochs" -lt "$TOTAL_EPOCHS" ]]; do
    target_epochs=$(( phase * PHASE_EPOCHS ))
    if [[ "$target_epochs" -gt "$TOTAL_EPOCHS" ]]; then
        target_epochs="$TOTAL_EPOCHS"
    fi

    python3 -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
config['data']['raw_dir'] = '${DATA_DIR}'
config['output_dir'] = '${RUNS_DIR}'
config['train']['epochs'] = ${target_epochs}
with open('colab_config.yaml', 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False)
"

    RESUME_FLAG=""
    if [[ "$phase" -gt 1 ]]; then
        RESUME_FLAG="--resume last"
    fi

    echo "[phase $phase] training to epoch ${target_epochs}/${TOTAL_EPOCHS}..."
    uv run train.py --config colab_config.yaml $RESUME_FLAG 2>&1 | tee "train-phase${phase}.log"

    echo "[phase $phase] evaluating best.pt on subsets 1/4 (full eval.py)..."
    uv run eval.py --config colab_config.yaml --weights "${RUNS_DIR}/weights/best.pt" \
        2>&1 | tee "eval-phase${phase}.log"

    phase=$(( phase + 1 ))
done

echo "[done] ${TOTAL_EPOCHS} epochs across $(( phase - 1 )) phases. See eval-phase*.log for each checkpoint's test mAP@0.5."
