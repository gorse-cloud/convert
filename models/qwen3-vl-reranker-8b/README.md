---
license: apache-2.0
pipeline_tag: text-ranking
tags:
- gguf
- quantized
- reranker
- multimodal
base_model:
- Qwen/Qwen3-VL-Reranker-8B
---

# Qwen3-VL-Reranker-8B

GGUF conversion of [Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B),
a query/document reranker, not a general-purpose text-generation or embedding model.

## Files

- `Qwen3-VL-Reranker-8B-BF16.gguf`
- `Qwen3-VL-Reranker-8B-Q8_0.gguf`
- `Qwen3-VL-Reranker-8B-Q4_K_M.gguf`
- `mmproj-Qwen3-VL-Reranker-8B-F16.gguf`

The language GGUF contains the yes/no classifier; the separate F16 projector
preserves the vision encoder and projection weights. Keep both for an eventual
multimodal integration; do not quantize the projector with the language quantizer.

## Text reranking with llama.cpp

Use llama.cpp with Qwen3-VL reranker conversion and RANK pooling support
(the pipeline was inspected against commit `abee0c8476b585b0d081d580b5ce21d0addc22c4`):

```bash
llama-server -hf __owner__/Qwen3-VL-Reranker-8B-GGUF:Q8_0 \
    --embedding --reranking --pooling rank --host 127.0.0.1 --port 8080
```

For the 4-bit variant, replace `:Q8_0` with `:Q4_K_M`. For a local file,
replace `-hf ...` with `-m Qwen3-VL-Reranker-8B-Q8_0.gguf`.
The pipeline builds `llama-quantize`; install or build `llama-server` separately.

```bash
curl http://127.0.0.1:8080/v1/rerank \
    -H 'Content-Type: application/json' \
    -d '{"query":"What is the capital of China?","documents":["Beijing is the capital of China.","Paris is the capital of France."],"top_n":2}'
```

**Multimodal limitation:** the inspected llama.cpp `/v1/rerank` API accepts text
queries and text documents only. Image URLs are treated as text; adding
`--mmproj` does not enable image/video reranking through that endpoint. The
projector is provided as an artifact, not as a claim of end-to-end multimodal
reranking support. Use the upstream Transformers implementation for multimodal
inputs until a compatible GGUF runtime integration is available and tested.
The converter's fixed text rerank prompt is not the official VL processor;
custom instructions and upstream score parity are not guaranteed.

## Conversion

Preserve the original source `README.md` beside the weights: llama.cpp uses
README/name hints to recognize the reranker and emit RANK pooling,
`yes`/`no` classifier labels, and `cls.output.weight` instead of the LM head.
Reserve space for source weights, conversion temporaries, and all four outputs;
8B needs substantially more RAM and disk than 2B. Offline script tests validate
commands and manifests, not actual GGUF contents or inference quality.

> This model is converted using https://github.com/gorse-cloud/convert.
