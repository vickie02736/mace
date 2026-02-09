#!/usr/bin/env bash
# MACE training tasks for CALF20_CO2 dataset.
# Task 1: Train each of 7 xyz files independently (90/10 train/val split).
# Task 2: Mixed 0ads+16ads training, then zero-shot/few-shot finetune on remaining test files.
# Usage: ./task.sh [--ddp] [--task1] [--task2] [--task2-base] [--task2-finetune]
#   Default (no flag): run both task1 and task2 sequentially.
#   --ddp: enable multi-GPU training via torchrun (auto-detects GPU count)

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

LOG_FILE="$SCRIPT_DIR/task.log"
PID_FILE="$SCRIPT_DIR/task.pid"
echo $$ > "$PID_FILE"
echo "task.sh started at $(date), PID=$$" > "$LOG_FILE"

# ============================================================
# Common MACE training arguments
# ============================================================
COMMON_ARGS=(
    --model="MACE"
    --hidden_irreps='128x0e + 128x1o'
    --r_max=5.0
    --batch_size=4
    --max_num_epochs=99999
    --max_steps=1000000
    --eval_interval_steps=500
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
# Parse which tasks to run
# ============================================================
USE_DDP=false
RUN_TASK1=false
RUN_TASK2_BASE=false
RUN_TASK2_FT=false

for arg in "$@"; do
    case "$arg" in
        --ddp)          USE_DDP=true ;;
        --task1)        RUN_TASK1=true ;;
        --task2)        RUN_TASK2_BASE=true; RUN_TASK2_FT=true ;;
        --task2-base)   RUN_TASK2_BASE=true ;;
        --task2-finetune) RUN_TASK2_FT=true ;;
    esac
done

# Default: run everything if no task flag is given
if ! $RUN_TASK1 && ! $RUN_TASK2_BASE && ! $RUN_TASK2_FT; then
    RUN_TASK1=true
    RUN_TASK2_BASE=true
    RUN_TASK2_FT=true
fi

# DDP setup: auto-detect GPUs; use torchrun when >1 GPU available
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"

if [ "$NGPUS" -gt 1 ] || $USE_DDP; then
    DDP_LAUNCHER=("torchrun" "--nproc_per_node=$NGPUS")
    DDP_ARGS=(--distributed --launcher=torchrun)
    echo "DDP auto-enabled with $NGPUS GPU(s)" | tee -a "$LOG_FILE"
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

if $RUN_TASK1; then
    echo "========== TASK 1: Independent per-file training ==========" | tee -a "$LOG_FILE"
    for base in "${ALL_FILES[@]}"; do
        xyz="${DATA_DIR}/${base}.xyz"
        [ -f "$xyz" ] || { echo "SKIP: $xyz not found" >> "$LOG_FILE"; continue; }

        work_dir="${RUNS_DIR}/task1/${base}"
        csv_log_dir="${LOG_DIR}/task1/${base}"
        mkdir -p "$work_dir" "$csv_log_dir"

        echo "  Task1: $base started at $(date)" | tee -a "$LOG_FILE"
        echo "  work_dir=$work_dir, csv_log=$csv_log_dir" >> "$LOG_FILE"

        PYTHONUNBUFFERED=1 "${DDP_LAUNCHER[@]}" -m mace.cli.run_train \
            --name="${base}" \
            --log_dir="${LOG_DIR}/task1" \
            --train_file="$xyz" \
            --valid_fraction=0.1 \
            --work_dir="$work_dir" \
            --E0s="average" \
            --csv_log_dir="$csv_log_dir" \
            "${COMMON_ARGS[@]}" \
            "${DDP_ARGS[@]}"

        echo "  Task1: $base finished at $(date)" | tee -a "$LOG_FILE"
    done
    echo "========== TASK 1 DONE ==========" | tee -a "$LOG_FILE"
fi

# ============================================================
# Task 2 Phase A: Mixed training (0ads + 16ads), 90/10 split
# ============================================================
TASK2_TRAIN_FILES=("training_data_0ads" "training_data_16ads")
TASK2_TEST_FILES=("training_data_2ads" "training_data_4ads" "training_data_8ads" "training_data_24ads" "training_data_32ads")

# Few-shot frame counts for finetune
FEWSHOT_NFRAMES=(1 2 5 10 20 50)

TASK2_BASE_DIR="${RUNS_DIR}/task2/base_0ads_16ads"
TASK2_BASE_LOG_DIR="${LOG_DIR}/task2/base_0ads_16ads"

if $RUN_TASK2_BASE; then
    echo "========== TASK 2 Phase A: Mixed 0ads+16ads base training ==========" | tee -a "$LOG_FILE"
    mkdir -p "$TASK2_BASE_DIR" "$TASK2_BASE_LOG_DIR"

    # Concatenate the two train files into a temporary combined file
    COMBINED_TRAIN="${TASK2_BASE_DIR}/combined_0ads_16ads.xyz"
    cat "${DATA_DIR}/training_data_0ads.xyz" "${DATA_DIR}/training_data_16ads.xyz" > "$COMBINED_TRAIN"
    echo "  Combined train file: $COMBINED_TRAIN" >> "$LOG_FILE"

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
fi

# ============================================================
# Task 2 Phase B: Zero-shot eval + Few-shot finetune on test files
# ============================================================
if $RUN_TASK2_FT; then
    echo "========== TASK 2 Phase B: Zero-shot / Few-shot finetune ==========" | tee -a "$LOG_FILE"

    # Locate the best checkpoint from base training
    BEST_CKPT="${TASK2_BASE_DIR}/base_0ads_16ads_run-123_stagetwo.model"
    # Try to find the actual best checkpoint
    if [ ! -f "$BEST_CKPT" ]; then
        BEST_CKPT=$(find "$TASK2_BASE_DIR" -name "*.model" -newer "$TASK2_BASE_DIR" 2>/dev/null | sort | tail -1)
    fi
    if [ ! -f "$BEST_CKPT" ]; then
        BEST_CKPT=$(find "$TASK2_BASE_DIR" -name "*.model" 2>/dev/null | sort | tail -1)
    fi
    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "ERROR: No base model checkpoint found in $TASK2_BASE_DIR" | tee -a "$LOG_FILE"
        echo "Please run --task2-base first." | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "  Using base checkpoint: $BEST_CKPT" | tee -a "$LOG_FILE"

    for test_base in "${TASK2_TEST_FILES[@]}"; do
        test_xyz="${DATA_DIR}/${test_base}.xyz"
        [ -f "$test_xyz" ] || { echo "SKIP: $test_xyz not found" >> "$LOG_FILE"; continue; }

        # --- Zero-shot evaluation (no finetune, just eval) ---
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
            || echo "  Zero-shot eval $test_base returned non-zero (may be expected for 0 epoch)" >> "$LOG_FILE"

        # --- Few-shot finetune with increasing frame counts ---
        for nframes in "${FEWSHOT_NFRAMES[@]}"; do
            ft_dir="${RUNS_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
            ft_log_dir="${LOG_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
            mkdir -p "$ft_dir" "$ft_log_dir"

            # Extract first N frames from the test xyz for finetune training
            ft_train_file="${ft_dir}/finetune_train_${nframes}frames.xyz"
            python3 -c "
from ase.io import read, write
import sys
frames = read('${test_xyz}', ':')
n = min(${nframes}, len(frames))
write('${ft_train_file}', frames[:n])
print(f'Extracted {n} frames for few-shot finetune')
" >> "$LOG_FILE" 2>&1

            echo "  Few-shot finetune: $test_base nframes=$nframes at $(date)" | tee -a "$LOG_FILE"
            PYTHONUNBUFFERED=1 "${DDP_LAUNCHER[@]}" -m mace.cli.run_train \
                --name="ft_${test_base}_${nframes}f" \
                --log_dir="${LOG_DIR}/task2/fewshot/${test_base}" \
                --train_file="$ft_train_file" \
                --valid_fraction=0.1 \
                --work_dir="$ft_dir" \
                --E0s="average" \
                --foundation_model="$BEST_CKPT" \
                --csv_log_dir="$ft_log_dir" \
                --model="MACE" \
                --hidden_irreps='128x0e + 128x1o' \
                --r_max=5.0 \
                --batch_size=4 \
                --max_num_epochs=99999 \
                --max_steps=50000 \
                --eval_interval_steps=500 \
                --csv_log_interval=10 \
                --patience=20 \
                --energy_key="energy" \
                --forces_key="forces" \
                --ema \
                --ema_decay=0.99 \
                --amsgrad \
                --restart_latest \
                --device=cuda \
                "${DDP_ARGS[@]}"

            echo "  Few-shot finetune: $test_base nframes=$nframes finished at $(date)" | tee -a "$LOG_FILE"
        done
    done
    echo "========== TASK 2 DONE ==========" | tee -a "$LOG_FILE"
fi

echo "All tasks finished at $(date)." | tee -a "$LOG_FILE"
