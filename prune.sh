#!/bin/bash

# Config
INITIAL_MODEL="meta-llama/Llama-3.2-3B-Instruct"
SPARSITY_FILE="current_sparsity.txt"
LOG_FILE="prune_log.txt"
FINAL_SPARSITY=0.5
STEP=0.05

# Load or initialize sparsity
if [ -f "$SPARSITY_FILE" ]; then
  SPARSITY=$(cat "$SPARSITY_FILE")
  echo "[RESUME] Resuming from sparsity $SPARSITY"
  PREV_MODEL="./llama-3.2-3B-pruned-$(printf %.2f $(echo "$SPARSITY + $STEP" | bc))-retrained"
else
  SPARSITY=0.95
  PREV_MODEL="$INITIAL_MODEL"
  echo "[INIT] Starting fresh from $SPARSITY"
  echo "$SPARSITY" > "$SPARSITY_FILE"
fi

# Round function
round() {
  printf "%.2f" "$1"
}

while (( $(echo "$SPARSITY >= $FINAL_SPARSITY" | bc -l) )); do
  SPARSITY_STR=$(round $SPARSITY)
  echo "[RUN] Pruning to sparsity $SPARSITY_STR" | tee -a "$LOG_FILE"

  PRUNED_MODEL="./llama-3.2-3B-pruned-${SPARSITY_STR}"
  RETRAINED_MODEL="${PRUNED_MODEL}-retrained"

  # Prune
  python ffnpruner.py --model_name_or_path "$PREV_MODEL" --output_path "$PRUNED_MODEL" 2>&1 | tee -a "$LOG_FILE"
  if [ $? -ne 0 ]; then echo "[ERROR] Pruning failed at $SPARSITY_STR"; exit 1; fi

  # Retrain
  echo "[RUN] Retraining $PRUNED_MODEL" | tee -a "$LOG_FILE"
  python train.py --model_name_or_path "$PRUNED_MODEL" --export "$RETRAINED_MODEL" 2>&1 | tee -a "$LOG_FILE"
  if [ $? -ne 0 ]; then echo "[ERROR] Training failed at $SPARSITY_STR"; exit 1; fi

  # Update loop variables
  PREV_MODEL="$RETRAINED_MODEL"
  SPARSITY=$(echo "$SPARSITY - $STEP" | bc)
  echo "$SPARSITY" > "$SPARSITY_FILE"
done

echo "[DONE] Pruning completed down to $FINAL_SPARSITY" | tee -a "$LOG_FILE"
