#!/usr/bin/env bash
# MACE Task 2 - Base Training: Mixed 0ads+16ads training (90/10 split).
# Usage: ./task2_train.sh [--ddp]

set -e

# ============================================================
# Environment
# ============================================================
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate /media/damoxing/che-liu-fileset/kwz/kwz-data/envs/mace_env
export PYTHONPATH="/media/damoxing/che-liu-fileset/kwz/mace:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/data/CALF20_CO2"
RUNS_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/ckpt/MACE"
LOG_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/log/MACE"

LOG_FILE="$SCRIPT_DIR/task2_train.log"
PID_FILE="$SCRIPT_DIR/task2_train.pid"
echo $$ > "$PID_FILE"
echo "task2_train.sh started at $(date), PID=$$" > "$LOG_FILE"

# ============================================================
# Common MACE training arguments
# ============================================================
COMMON_ARGS=(
    --model="MACE"
    --hidden_irreps='128x0e + 128x1o'
    --r_max=5.0
    --batch_size=4
    --valid_batch_size=4
    --max_steps=50000
    --eval_interval_steps=200
    --csv_log_interval=10
    --patience=20
    --config_type_weights='{"Default":1.0}'
    --energy_key="energy"
    --forces_key="forces"
    --stage_two
    --start_stage_two=1200
    --ema
    --ema_decay=0.99
    --amsgrad
    --restart_latest
    --device=cuda
)

# ============================================================
# DDP setup
# ============================================================
USE_DDP=false
for arg in "$@"; do
    case "$arg" in
        --ddp) USE_DDP=true ;;
    esac
done

NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"

if [ "$NGPUS" -gt 1 ] || $USE_DDP; then
    DDP_LAUNCHER=("torchrun" "--nproc_per_node=$NGPUS")
    DDP_ARGS=(--distributed --launcher=torchrun)
    echo "DDP enabled with $NGPUS GPU(s)" | tee -a "$LOG_FILE"
else
    DDP_LAUNCHER=("python")
    DDP_ARGS=()
    echo "Single-GPU mode" | tee -a "$LOG_FILE"
fi

# ============================================================
# Task 2 Base: Mixed training (0ads + 16ads)
# ============================================================
TASK2_BASE_DIR="${RUNS_DIR}/task2/base_0ads_16ads"
TASK2_BASE_LOG_DIR="${LOG_DIR}/task2/base_0ads_16ads"

echo "========== TASK 2: Mixed 0ads+16ads base training ==========" | tee -a "$LOG_FILE"
mkdir -p "$TASK2_BASE_DIR" "$TASK2_BASE_LOG_DIR"

# Concatenate the two train files into a combined file
COMBINED_TRAIN="${TASK2_BASE_DIR}/combined_0ads_16ads.xyz"
cat "${DATA_DIR}/training_data_0ads.xyz" "${DATA_DIR}/training_data_16ads.xyz" > "$COMBINED_TRAIN"
echo "  Combined train file: $COMBINED_TRAIN" | tee -a "$LOG_FILE"

PYTHONUNBUFFERED=1 "${DDP_LAUNCHER[@]}" -m mace.cli.run_train \
    --name="base_0ads_16ads" \
    --log_dir="${LOG_DIR}/task2" \
    --train_file="$COMBINED_TRAIN" \
    --valid_fraction=0.1 \
    --work_dir="$TASK2_BASE_DIR" \
    --E0s="average" \
    --csv_log_dir="$TASK2_BASE_LOG_DIR" \
    "${COMMON_ARGS[@]}" \
    "${DDP_ARGS[@]}"

echo "  Task2 base training finished at $(date)" | tee -a "$LOG_FILE"
echo "========== TASK 2 BASE TRAINING DONE ==========" | tee -a "$LOG_FILE"
