#!/usr/bin/env bash
# MACE Task 2 - Zero-shot eval on all test files using best checkpoint.
# Requires a base checkpoint from task2_train.sh.
# Usage: ./task2_finetune.sh [--ddp]

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

LOG_FILE="$SCRIPT_DIR/task2_finetune.log"
PID_FILE="$SCRIPT_DIR/task2_finetune.pid"
echo $$ > "$PID_FILE"
echo "task2_finetune.sh started at $(date), PID=$$" > "$LOG_FILE"

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
# Locate base checkpoint
# ============================================================
TASK2_BASE_DIR="${RUNS_DIR}/task2/base_0ads_16ads"
TASK2_TEST_FILES=("training_data_2ads" "training_data_4ads" "training_data_8ads" "training_data_24ads" "training_data_32ads")

BEST_CKPT="${TASK2_BASE_DIR}/base_0ads_16ads_run-123_stagetwo.model"
if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT=$(find "$TASK2_BASE_DIR" -name "*.model" -newer "$TASK2_BASE_DIR" 2>/dev/null | sort | tail -1)
fi
if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT=$(find "$TASK2_BASE_DIR" -name "*.model" 2>/dev/null | sort | tail -1)
fi
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: No base model checkpoint found in $TASK2_BASE_DIR" | tee -a "$LOG_FILE"
    echo "Please run task2_train.sh first." | tee -a "$LOG_FILE"
    exit 1
fi
echo "Using base checkpoint: $BEST_CKPT" | tee -a "$LOG_FILE"

# ============================================================
# Zero-shot eval on all test files
# ============================================================
echo "========== TASK 2: Zero-shot eval ==========" | tee -a "$LOG_FILE"

total_files=${#TASK2_TEST_FILES[@]}
current_index=0

for test_base in "${TASK2_TEST_FILES[@]}"; do
    current_index=$((current_index + 1))
    echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Progress: [$current_index/$total_files] Processing $test_base..." | tee -a "$LOG_FILE"
    echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"

    test_xyz="${DATA_DIR}/${test_base}.xyz"
    [ -f "$test_xyz" ] || { echo "SKIP: $test_xyz not found" | tee -a "$LOG_FILE"; continue; }

    # --- Zero-shot evaluation ---
    zs_dir="${RUNS_DIR}/task2/zeroshot/${test_base}"
    zs_log_dir="${LOG_DIR}/task2/zeroshot/${test_base}"
    mkdir -p "$zs_dir" "$zs_log_dir"

    echo "  Zero-shot eval: $test_base at $(date)" | tee -a "$LOG_FILE"
    PYTHONUNBUFFERED=1 "${DDP_LAUNCHER[@]}" -m mace.cli.run_train \
        --name="zs_${test_base}" \
        --log_dir="${LOG_DIR}/task2/zeroshot" \
        --train_file="$test_xyz" \
        --valid_fraction=0.1 \
        --work_dir="$zs_dir" \
        --E0s="average" \
        --foundation_model="$BEST_CKPT" \
        --max_num_epochs=0 \
        --csv_log_dir="$zs_log_dir" \
        --model="MACE" \
        --hidden_irreps='128x0e + 128x1o' \
        --r_max=5.0 \
        --batch_size=4 \
        --energy_key="energy" \
        --forces_key="forces" \
        --device=cuda \
        "${DDP_ARGS[@]}" \
        || echo "  Zero-shot eval $test_base returned non-zero (may be expected for 0 epoch)" | tee -a "$LOG_FILE"
done

echo "========== TASK 2 ZERO-SHOT EVAL DONE ==========" | tee -a "$LOG_FILE"
echo "Task 2 zero-shot eval finished at $(date)." | tee -a "$LOG_FILE"
