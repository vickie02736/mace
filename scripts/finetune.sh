#!/usr/bin/env bash
# MACE few-shot finetuning script.
# Usage: ./finetune.sh --ckpt <path> --train_file <xyz> --nframes <N> --test_file <xyz> \
#                      --csv_log_dir <dir> --work_dir <dir> [--device cuda]
set -e

# Defaults
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SIZE=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2"; shift 2 ;;
        --train_file) TRAIN_FILE="$2"; shift 2 ;;
        --nframes)    NFRAMES="$2"; shift 2 ;;
        --test_file)  TEST_FILE="$2"; shift 2 ;;
        --csv_log_dir) CSV_LOG_DIR="$2"; shift 2 ;;
        --work_dir)   WORK_DIR="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --ddp)        shift ;;  # MACE does not use DDP flag
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -z "$CKPT" ] && { echo "ERROR: --ckpt required"; exit 1; }
[ -z "$TRAIN_FILE" ] && { echo "ERROR: --train_file required"; exit 1; }
[ -z "$NFRAMES" ] && { echo "ERROR: --nframes required"; exit 1; }
[ -z "$TEST_FILE" ] && { echo "ERROR: --test_file required"; exit 1; }
[ -z "$CSV_LOG_DIR" ] && { echo "ERROR: --csv_log_dir required"; exit 1; }
[ -z "$WORK_DIR" ] && { echo "ERROR: --work_dir required"; exit 1; }

# Activate environment
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate /media/damoxing/che-liu-fileset/kwz/kwz-data/envs/mace_env
export PYTHONPATH="/media/damoxing/che-liu-fileset/kwz/mace:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR" "$CSV_LOG_DIR"

# Step 1: Extract first N frames
FT_TRAIN_FILE="${WORK_DIR}/finetune_train_${NFRAMES}frames.xyz"
python3 -c "
from ase.io import read, write
frames = read('${TRAIN_FILE}', ':')
n = min(${NFRAMES}, len(frames))
write('${FT_TRAIN_FILE}', frames[:n])
print(f'Extracted {n} frames for few-shot finetune')
"

# Step 2: Compute max_steps = ceil(nframes / batch_size) for 1 epoch
MAX_STEPS=$(python3 -c "import math; print(math.ceil(${NFRAMES} / ${BATCH_SIZE}))")
echo "Finetune: nframes=${NFRAMES}, batch_size=${BATCH_SIZE}, max_steps=${MAX_STEPS}"

# Step 3: Finetune (1 epoch)
PYTHONUNBUFFERED=1 python -m mace.cli.run_train \
    --name="ft_${NFRAMES}f" \
    --foundation_model="$CKPT" \
    --train_file="$FT_TRAIN_FILE" \
    --valid_fraction=0.0 \
    --max_steps="$MAX_STEPS" \
    --max_num_epochs=1 \
    --patience=9999 \
    --csv_log_dir="$CSV_LOG_DIR" \
    --work_dir="$WORK_DIR" \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size="$BATCH_SIZE" \
    --energy_key="energy" \
    --forces_key="forces" \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device="$DEVICE" \
    --E0s="average"

# Step 4: Test on full file
BEST_MODEL=$(find "$WORK_DIR" -name "*.model" 2>/dev/null | sort | tail -1)
if [ -z "$BEST_MODEL" ]; then
    echo "WARNING: No .model file found after finetune, skipping test"
    exit 0
fi

bash "$SCRIPT_DIR/test.sh" \
    --ckpt "$BEST_MODEL" \
    --test_file "$TEST_FILE" \
    --csv_log_dir "$CSV_LOG_DIR" \
    --device "$DEVICE"
