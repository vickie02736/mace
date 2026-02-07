#!/usr/bin/env bash
# MACE standalone evaluation script.
# Usage: ./test.sh --ckpt <path> --test_file <xyz> --csv_log_dir <dir> [--device cuda]
set -e

# Defaults
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2"; shift 2 ;;
        --test_file)  TEST_FILE="$2"; shift 2 ;;
        --csv_log_dir) CSV_LOG_DIR="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -z "$CKPT" ] && { echo "ERROR: --ckpt required"; exit 1; }
[ -z "$TEST_FILE" ] && { echo "ERROR: --test_file required"; exit 1; }
[ -z "$CSV_LOG_DIR" ] && { echo "ERROR: --csv_log_dir required"; exit 1; }

# Activate environment
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate /media/damoxing/che-liu-fileset/kwz/kwz-data/envs/mace_env

WORK_DIR=$(mktemp -d)
mkdir -p "$CSV_LOG_DIR"

# MACE zero-shot evaluation: load model, run 0 epochs (initial validation pass only)
PYTHONUNBUFFERED=1 mace_run_train \
    --name="test_eval" \
    --foundation_model="$CKPT" \
    --train_file="$TEST_FILE" \
    --valid_fraction=0.1 \
    --max_num_epochs=0 \
    --csv_log_dir="$CSV_LOG_DIR" \
    --work_dir="$WORK_DIR" \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=4 \
    --energy_key="energy" \
    --forces_key="forces" \
    --device="$DEVICE" \
    --E0s="average"

# Clean up temp work dir
rm -rf "$WORK_DIR"
