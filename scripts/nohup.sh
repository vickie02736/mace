#!/usr/bin/env bash
# Launch task.sh in background with nohup.
# Usage: ./nohup.sh [task.sh args...]
#   e.g. ./nohup.sh --task1
#        ./nohup.sh --task2-finetune

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/nohup.out"

nohup bash "${SCRIPT_DIR}/task.sh" "$@" > "$LOG" 2>&1 &
BG_PID=$!
echo "$BG_PID" > "${SCRIPT_DIR}/nohup.pid"
echo "task.sh started in background, PID=${BG_PID}, log: $LOG"


# ./nohup.sh              # 默认全部任务
# ./nohup.sh --task1      # 只跑 task1
# ./nohup.sh --task2-finetune  # 只跑 task2 微调

#nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null; echo "---"; ps aux | grep -E 'python|mace|torch' | grep -v grep
# kill -9 4185538 1999368 1999353 2>/dev/null; sleep 2; nvidia-smi

# 8卡 DDP 训练（自动检测 GPU 数量）
# ./nohup.sh --ddp --task1

# 手动指定 GPU 数量
# NGPUS=4 ./nohup.sh --ddp --task1

# 单卡（不变，向后兼容）
# ./nohup.sh --task1