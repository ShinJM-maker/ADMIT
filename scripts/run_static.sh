#!/usr/bin/env bash
# Static M3DocVQA evaluation (Sec. 6.1, Table 2; SupportRouter and learned alternatives, App. D.3).
# Four reader passes over the frozen HiKEY retrieval (inputs/static): strict initial read and guess RepairRead,
# on the calibration and reporting partitions (1,162 questions each). One GPU per pass; set GPUS to spread them.
#   ADMIT_INDEX  HiKEY M3DocVQA index (chunks, section cards, docs)   ADMIT_DATA  datasets root (M3DocVQA/val.json)
#   ADMIT_READER_MODEL  Qwen3-VL-8B-Instruct path or hub id         ADMIT_WORK  output root (default ./work)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${ADMIT_WORK:-$REPO/work}/static; mkdir -p "$W"
MODEL=${ADMIT_READER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
read -r -a GPUS <<< "${GPUS:-0 0 0 0}"
for part in calibration reporting; do
  [ -s "$W/${part}_1162_retrieval.jsonl" ] || gzip -dc "$REPO/inputs/static/${part}_retrieval.jsonl.gz" > "$W/${part}_1162_retrieval.jsonl"
done
cp "$W/reporting_1162_retrieval.jsonl" "$W/heldout_1162_retrieval.jsonl"
gzip -dc "$REPO/inputs/static/router_candidate_pool.jsonl.gz" > "$W/cands_with_ocr_lp.jsonl"

read_pass() {  # gpu partition prompt pred_out packet_out
  CUDA_VISIBLE_DEVICES=$1 python "$REPO/experiments/static/run_reader_vllm.py" \
    --dataset M3DocVQA --retrieval "$W/$2_1162_retrieval.jsonl" --out "$W/$4" --evidence_packet_out "$W/$5" \
    --model "$MODEL" --token_budget 14000 --image_token_cost 300 --max_images_total 1 \
    --max_model_len 24576 --max_new_tokens 64 --prompt_variant "$3" --batch_size 8 --gpu_mem_util 0.85 \
    --fail_on_generation_error > "$W/$4.log" 2>&1
}
read_pass "${GPUS[0]}" calibration strict calibration_strict_pred.jsonl calibration_strict_packet.jsonl &
read_pass "${GPUS[1]}" reporting   strict baseline_strict_pred.jsonl    baseline_strict_packets.jsonl &
read_pass "${GPUS[2]}" reporting   guess  repair_guess_pred.jsonl       repair_guess_packets.jsonl &
read_pass "${GPUS[3]}" calibration guess  calibration_guess_pred.jsonl  calibration_guess_packets.jsonl &
wait
python "$REPO/experiments/static/evaluate_static.py"                 # ADMIT, comparators, leakage-free SupportRouter
python "$REPO/experiments/learned_alternatives/refusal_selector.py"  # repair-gain decomposition, refusal selector
python "$REPO/experiments/learned_alternatives/verifier_stage0.py"   # second-chance verifier, static partitions
