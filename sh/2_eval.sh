#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./logs

RAW_DIR="data/raw_questions"
EVAL_DIR="data/evaluated_questions"
CFG_DIR="configs"

shopt -s nullglob

# Space-separated list of models to use for evaluation
models="gemini-3-flash-preview-nothinking"

for raw_path in "$RAW_DIR"/*.json; do
  raw_file="$(basename "$raw_path")"
  stem="${raw_file%.json}"  # e.g. 20260208_192754_without_innovative_vocab

  eval_folder="$EVAL_DIR/$stem"
  ranked_file="$eval_folder/ranked_questions.json"

  # Skip if already evaluated
  if [[ -d "$eval_folder" && -f "$ranked_file" ]]; then
    echo "[SKIP] Already evaluated: $stem"
    continue
  fi

  # Find the corresponding config file (strip the _without_innovative_vocab suffix)
  cfg_stem="${stem%_without_innovative_vocab}"
  cfg_path="$CFG_DIR/$cfg_stem.json"

  if [[ ! -f "$cfg_path" ]]; then
    echo "[WARN] Missing config: $cfg_path (from raw: $raw_file) -> skip"
    continue
  fi

  # Extract the 'field' value from the config (prefer jq; fall back to python)
  if command -v jq >/dev/null 2>&1; then
    field="$(jq -r '.field // empty' "$cfg_path")"
  else
    field="$(
  python - "$cfg_path" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f)
print(obj.get("field",""))
PY
)"
  fi

  if [[ -z "${field:-}" || "$field" == "null" ]]; then
    echo "[WARN] Empty field in config: $cfg_path -> skip"
    continue
  fi

  echo "[RUN] $stem -> field='$field'"
  python -u agents/step_2_evaluator.py \
    --input_file ./data/raw_questions/${raw_file} \
    --output_dir ./data/evaluated_questions/${stem} \
    --field "$field" \
    --similarity_threshold 0.85 \
    --filter_batch_size 10 \
    --comparison_rounds 3 \
    --group_size 5 \
    --models $models \
    --max_concurrent_tasks 64 > ./logs/step_2_evaluator_${stem}.log 2>&1 &
done
