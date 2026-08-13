# --- 3. Dataset + checkpoints on Drive (survives disconnects) ---------------
python3 -c "
from google.colab import drive
drive.mount('/content/drive')
"
DRIVE_ROOT="/content/drive/MyDrive/InmindAcademyDetector-${BRANCH}"
DATA_DIR="${DRIVE_ROOT}/dataset"
RUNS_DIR="${DRIVE_ROOT}/runs/baseline"
mkdir -p "$DATA_DIR" "$RUNS_DIR"

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
RESUME_FLAG=""
if [[ -f "${RUNS_DIR}/weights/last.ckpt" ]]; then
    RESUME_FLAG="--resume last"
fi
uv run train.py --config colab_config.yaml $RESUME_FLAG 2>&1 | tee train.log