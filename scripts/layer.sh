#!/usr/bin/env bash
# MACE Layer Monitor Training: 16ads only, with diagnostic per-layer TensorBoard.
# All output (CSV + TensorBoard) goes to /kwz-data/tensorboard/MACE/
# Usage: ./layer.sh [--ddp]

set -e

# ============================================================
# Environment
# ============================================================
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate /media/damoxing/che-liu-fileset/kwz/kwz-data/envs/mace_env
export PYTHONPATH="/media/damoxing/che-liu-fileset/kwz/mace:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/data/CALF20_CO2"
TB_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/tensorboard/MACE"
# Match TrumbleMOF structure: TB_DIR/layer_16ads/{train,val,CO2,framework,...}/log.csv
# and TB_DIR/tensorboard_layers/ for LayerMonitor
LAYER_OUT="${TB_DIR}/layer_16ads"

LOG_FILE="$SCRIPT_DIR/layer.log"
PID_FILE="$SCRIPT_DIR/layer.pid"
echo $$ > "$PID_FILE"
echo "layer.sh started at $(date), PID=$$" > "$LOG_FILE"

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
# Layer Monitor Training: 16ads only
# ============================================================
TRAIN_FILE="${DATA_DIR}/training_data_16ads.xyz"
CKPT_DIR="${TB_DIR}/checkpoints"

echo "========== Layer Monitor: MACE on 16ads ==========" | tee -a "$LOG_FILE"
echo "  Train file: $TRAIN_FILE" | tee -a "$LOG_FILE"
echo "  CSV output: $LAYER_OUT" | tee -a "$LOG_FILE"
echo "  TensorBoard layers: $TB_DIR/tensorboard_layers" | tee -a "$LOG_FILE"
mkdir -p "$LAYER_OUT" "$CKPT_DIR"

PYTHONUNBUFFERED=1 "${DDP_LAUNCHER[@]}" -m mace.cli.run_train \
    --name="layer_16ads" \
    --log_dir="$TB_DIR" \
    --train_file="$TRAIN_FILE" \
    --valid_fraction=0.1 \
    --work_dir="$CKPT_DIR" \
    --E0s="average" \
    --csv_log_dir="$LAYER_OUT" \
    --tensorboard_dir="$TB_DIR" \
    "${COMMON_ARGS[@]}" \
    "${DDP_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

echo "Layer monitor training finished at $(date)." | tee -a "$LOG_FILE"
