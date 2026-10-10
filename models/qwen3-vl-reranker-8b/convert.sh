#!/bin/bash
set -euox pipefail

OUTPUT_DIR="$1"
LLAMA_CPP="$2"
DISPLAY_NAME="Qwen3-VL-Reranker-8B"
QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"

# Preserve the source README so llama.cpp detects the VL reranker and
# extracts the yes/no classifier instead of exporting a generation model.
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$PATH_PRIMARY" \
    --outfile "$OUTPUT_DIR/${DISPLAY_NAME}-BF16.gguf" --outtype bf16 --model-name "$DISPLAY_NAME"

"$QUANTIZE" "$OUTPUT_DIR/${DISPLAY_NAME}-BF16.gguf" "$OUTPUT_DIR/${DISPLAY_NAME}-Q8_0.gguf" Q8_0 1>&2
"$QUANTIZE" "$OUTPUT_DIR/${DISPLAY_NAME}-BF16.gguf" "$OUTPUT_DIR/${DISPLAY_NAME}-Q4_K_M.gguf" Q4_K_M 1>&2

python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$PATH_PRIMARY" --mmproj \
    --outfile "$OUTPUT_DIR/mmproj-${DISPLAY_NAME}-F16.gguf" --outtype f16 --model-name "$DISPLAY_NAME"

printf '%s\n' "${DISPLAY_NAME}-BF16.gguf" "${DISPLAY_NAME}-Q8_0.gguf" \
    "${DISPLAY_NAME}-Q4_K_M.gguf" "mmproj-${DISPLAY_NAME}-F16.gguf" >> "$OUTPUT_DIR/.produced_files"
