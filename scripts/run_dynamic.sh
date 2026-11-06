#!/usr/bin/env bash
# Dynamic M3DocVQA evaluation (Sec. 6.2-6.3, Table 1): ReAct, PlanRAG and AutoGen, 300 questions x 3 agents per policy.
# Runs one shard per GPU listed in GPUS, then summarizes. Same environment variables as run_static.sh.
#   GPUS="0 1 2 3" bash scripts/run_dynamic.sh                     # all six policies
#   POLICIES="admit fail_open" bash scripts/run_dynamic.sh         # a subset
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
read -r -a G <<< "${GPUS:-0}"
P=${POLICIES:-fail_open hard_refusal admit final_support gate_only gate_repair}
W=${ADMIT_WORK:-$REPO/work}/dynamic; mkdir -p "$W"
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} python "$REPO/experiments/dynamic/run_dynamic.py" \
    --shard_id "$i" --num_shards "${#G[@]}" --policies $P > "$W/shard$i.log" 2>&1 &
  sleep 60   # stagger vLLM start-up
done
wait
python "$REPO/experiments/dynamic/summarize_dynamic.py"
