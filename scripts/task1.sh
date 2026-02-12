#!/usr/bin/env bash
# MACE Task 1: Train each of 7 xyz files independently (90/10 train/val split).
# Usage: ./task1.sh [--ddp]

# set -e

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

LOG_FILE="$SCRIPT_DIR/task1.log"
PID_FILE="$SCRIPT_DIR/task1.pid"
echo $$ > "$PID_FILE"
echo "task1.sh started at $(date), PID=$$" > "$LOG_FILE"

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
# Task 1: Independent training on each xyz file (90/10 split)
# ============================================================
ALL_FILES=(
    training_data_0ads
    training_data_2ads
    training_data_4ads
    training_data_8ads
    training_data_16ads
    training_data_24ads
    training_data_32ads
)

echo "========== TASK 1: Independent per-file training ==========" | tee -a "$LOG_FILE"
for base in "${ALL_FILES[@]}"; do
    xyz="${DATA_DIR}/${base}.xyz"
    [ -f "$xyz" ] || { echo "SKIP: $xyz not found" >> "$LOG_FILE"; continue; }

    work_dir="${RUNS_DIR}/task1/${base}"
    csv_log_dir="${LOG_DIR}/task1/${base}"
    mkdir -p "$work_dir" "$csv_log_dir"

    echo "  Task1: $base started at $(date)" | tee -a "$LOG_FILE"

    if PYTHONUNBUFFERED=1 "${DDP_LAUNCHER[@]}" -m mace.cli.run_train \
        --name="${base}" \
        --log_dir="${LOG_DIR}/task1" \
        --train_file="$xyz" \
        --valid_fraction=0.1 \
        --work_dir="$work_dir" \
        --E0s="average" \
        --csv_log_dir="$csv_log_dir" \
        "${COMMON_ARGS[@]}" \
        "${DDP_ARGS[@]}"; then
        echo "  Task1: $base finished at $(date)" | tee -a "$LOG_FILE"
    else
        echo "  Task1: $base FAILED at $(date)" | tee -a "$LOG_FILE"
    fi
done
echo "========== TASK 1 DONE ==========" | tee -a "$LOG_FILE"
