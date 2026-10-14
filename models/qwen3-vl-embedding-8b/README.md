---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: feature-extraction
base_model:
- Qwen/Qwen3-VL-Embedding-8B
tags:
- gguf
- embeddings
- multimodal
---

# Qwen3-VL-Embedding-8B GGUF

GGUF conversion of [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B),
part of the official [Qwen3-VL-Embedding collection](https://huggingface.co/collections/Qwen/qwen3-vl-embedding).
These are embedding models for retrieval, not text generation or reranking.
The full embedding dimension is 4096.

## Files

- `Qwen3-VL-Embedding-8B-BF16.gguf`
- `Qwen3-VL-Embedding-8B-Q8_0.gguf`
- `Qwen3-VL-Embedding-8B-Q4_K_M.gguf`
- `mmproj-Qwen3-VL-Embedding-8B-F16.gguf`

The language GGUF contains the text backbone. Image/video processing also
requires the matching vision projector and a compatible multimodal runtime.
Do not mix projectors from another model or size.

## Conversion

```bash
# From the convert repository; no upload or HF_TOKEN required
bash convert.sh --owner __owner__ --one qwen3-vl-embedding-8b --keep --no-upload
```

Remove `--no-upload` and set `HF_TOKEN` with write access to publish to
`__owner__/Qwen3-VL-Embedding-8B-GGUF`. Use a recent llama.cpp; converter and server
sources were inspected at `091b6a7fc561b78da90a0ed2d524ec995458c657`.
For reproducibility, pass that revision with `--llama-commit`.

Keep the downloaded source `modules.json`, `1_Pooling/config.json`,
`chat_template.jinja`, and `README.md` intact. The converter reads the pooling
configuration (`pooling_mode: lasttoken`) to emit LAST pooling, not RANK.
The upload README is generated separately and does not replace the source card.
Reserve disk for the original weights, HF cache, temporary tensors, all three
language outputs and the projector. The 8B model needs substantially more
RAM and disk than 2B. The pipeline builds `llama-quantize`, not `llama-server`.

## Text embeddings

Build/install a compatible `llama-server` separately, then run, for example:

```bash
llama-server \
  -hf __owner__/Qwen3-VL-Embedding-8B-GGUF:Q4_K_M \
  --embedding --pooling last --embd-normalize 2 \
  --ctx-size 4096 --batch-size 4096 --ubatch-size 4096 \
  --parallel 1 --port 8080
```

The embeddings API does **not** automatically apply the model's chat template.
Supply the complete formatted input, including the assistant prefix. This
example uses the official default instruction and last-token/L2 pooling:

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "$(python3 - <<'PY'
import json
prompt = ("<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
          "<|im_start|>user\nA photo of a cat.<|im_end|>\n"
          "<|im_start|>assistant\n")
print(json.dumps({"model": "__owner__/Qwen3-VL-Embedding-8B-GGUF",
                  "input": prompt, "encoding_format": "float"}))
PY
)"
```

Use task-specific instructions consistently with the official model's guidance.
This script exports the full-dimensional embedding backbone; any dimension
truncation and subsequent normalization are client/runtime responsibilities.

## Multimodal scope and validation

The inspected llama.cpp native `/embedding` endpoint documents multimodal
prompts; image/video inputs need `--mmproj` pointing to the downloaded
`mmproj-Qwen3-VL-Embedding-8B-F16.gguf` and the runtime's multimodal prompt format.
Do not pass image URLs as plain text to `/v1/embeddings` and expect image
embeddings. See [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/091b6a7fc561b78da90a0ed2d524ec995458c657/tools/server/README.md)
and [the official reference implementation](https://github.com/QwenLM/Qwen3-VL-Embedding).

Repository tests exercise conversion commands, failure handling, source metadata
preservation and the four-file manifest using placeholder executables. They do
not validate real GGUF tensor contents, text/image/video inference, or embedding
parity with the official processor. Quantization can change retrieval quality;
evaluate the selected format on your own workload before deployment.
