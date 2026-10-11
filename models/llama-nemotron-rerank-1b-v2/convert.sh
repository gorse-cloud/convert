#!/bin/bash
set -euo pipefail
OUTPUT_DIR="$1"
LLAMA_CPP="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAME="llama-nemotron-rerank-1b-v2"
PYTHON="${PYTHON:-python3}"
QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/.produced_files" "$OUTPUT_DIR/.produced_files.partial"
cleanup() {
    rm -f "$OUTPUT_DIR/$NAME-BF16.partial.gguf" \
        "$OUTPUT_DIR/$NAME-Q8_0.partial.gguf" "$OUTPUT_DIR/$NAME-Q4_K_M.partial.gguf" \
        "$OUTPUT_DIR/.produced_files.partial"
}
trap cleanup EXIT
# Require explicit F32 classifier preservation rather than silently changing scores.
HELP="$({ "$QUANTIZE" --help; } 2>&1 || true)"
if [[ "$HELP" != *"--tensor-type"* ]]; then
    printf '%s\n' 'Quantizer must support --tensor-type for the F32 classifier.' >&2
    exit 1
fi
"$PYTHON" "$SCRIPT_DIR/convert.py" --llama-cpp "$LLAMA_CPP" "$PATH_PRIMARY" \
    --outfile "$OUTPUT_DIR/$NAME-BF16.partial.gguf" --outtype bf16 --model-name "$NAME"
mv "$OUTPUT_DIR/$NAME-BF16.partial.gguf" "$OUTPUT_DIR/$NAME-BF16.gguf"
for TYPE in Q8_0 Q4_K_M; do
    "$QUANTIZE" --tensor-type cls.output.weight=f32 "$OUTPUT_DIR/$NAME-BF16.gguf" \
        "$OUTPUT_DIR/$NAME-$TYPE.partial.gguf" "$TYPE" >&2
    mv "$OUTPUT_DIR/$NAME-$TYPE.partial.gguf" "$OUTPUT_DIR/$NAME-$TYPE.gguf"
done
cp "$SCRIPT_DIR/llama-classifier.patch" "$OUTPUT_DIR/llama-classifier.patch"
cp "$PATH_PRIMARY/LICENSE" "$OUTPUT_DIR/LICENSE"
printf '%s\n' "$NAME-BF16.gguf" "$NAME-Q8_0.gguf" "$NAME-Q4_K_M.gguf" \
    llama-classifier.patch LICENSE > "$OUTPUT_DIR/.produced_files.partial"
mv "$OUTPUT_DIR/.produced_files.partial" "$OUTPUT_DIR/.produced_files"
