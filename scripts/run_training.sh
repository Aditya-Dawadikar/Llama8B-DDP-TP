#!/usr/bin/env bash
# Wrapper script to launch DDP training with LoRA or AdaLoRA.
#
# Usage:
#   bash scripts/run_training.sh lora sst2
#   bash scripts/run_training.sh adalora mnli

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <lora|adalora> <glue_task> [num_gpus]"
  exit 1
fi

MODE="$1"
TASK="$2"
NGPUS="${3:-4}"
PORT=29501

if [[ "$MODE" == "lora" ]]; then
  echo "[run_training] Launching LoRA training on task $TASK with $NGPUS GPUs"
  torchrun --nproc_per_node="$NGPUS" --master_port="$PORT" \
    train_ddp_lora.py \
    --task_name "$TASK" \
    --output_dir "runs/llama3_lora_$TASK" \
    --per_device_batch_size 8 \
    --epochs 3
elif [[ "$MODE" == "adalora" ]]; then
  echo "[run_training] Launching AdaLoRA training on task $TASK with $NGPUS GPUs"
  torchrun --nproc_per_node="$NGPUS" --master_port="$PORT" \
    train_ddp_adalora.py \
    --task_name "$TASK" \
    --output_dir "runs/llama3_adalora_$TASK" \
    --per_device_batch_size 8 \
    --epochs 3
else
  echo "Unknown mode: $MODE (expected 'lora' or 'adalora')"
  exit 1
fi