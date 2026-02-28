#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="configs"
RAW_DIR="data/raw_questions"

shopt -s nullglob

for config_path in "$CONFIG_DIR"/*.json; do
    config_file="$(basename "$config_path")"
    stem="${config_file%.json}"   # e.g. 20260208_192754

    # Derive the expected raw output filename
    raw_file="${stem}_without_innovative_vocab.json"
    raw_path="$RAW_DIR/$raw_file"

    # Skip if already generated
    if [[ -f "$raw_path" ]]; then
        echo "[SKIP] Already generated: $raw_file"
        continue
    fi

    echo "[RUN] Generating from $config_file..."
    python agents/step_1_generator.py --config_path "$config_path" > ./logs/step_1_generator_${raw_file}.log 2>&1 &

    echo ""
    echo ""
done
