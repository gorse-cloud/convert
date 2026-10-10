---
license: apache-2.0
pipeline_tag: text-ranking
tags:
- gguf
- quantized
- reranker
base_model:
- Qwen/Qwen3-Reranker-8B
---

# Qwen3-Reranker-8B

GGUF conversion of [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
for dedicated query/document reranking, not text generation or general-purpose
embedding vectors. Available files:

- `Qwen3-Reranker-8B-BF16.gguf`
- `Qwen3-Reranker-8B-Q8_0.gguf`
- `Qwen3-Reranker-8B-Q4_K_M.gguf`

## Run with llama.cpp

Use a recent llama.cpp build with Qwen3 reranker conversion and server support:

```bash
llama-server -hf __owner__/Qwen3-Reranker-8B-GGUF:Q8_0 \
    --embedding --reranking --pooling rank --host 127.0.0.1 --port 8080
```

For the 4-bit variant, use `-hf __owner__/Qwen3-Reranker-8B-GGUF:Q4_K_M`
in the same command. For a local file, replace `-hf ...` with
`-m Qwen3-Reranker-8B-Q8_0.gguf` or `-m Qwen3-Reranker-8B-Q4_K_M.gguf`.
The conversion pipeline builds `llama-quantize`, not `llama-server`; build or
install the server separately.

```bash
curl http://127.0.0.1:8080/v1/rerank \
    -H 'Content-Type: application/json' \
    -d '{"query":"What is the capital of China?","documents":["Beijing is the capital of China.","Paris is the capital of France."],"top_n":2}'
```

`/reranking`, `/rerank`, and `/v1/reranking` are aliases. The endpoint scores
query/document pairs; it is not a chat-completions endpoint. See the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-reranking-rerank-documents-according-to-a-given-query).

## Conversion notes

Keep the original downloaded source `README.md` alongside the weights and
configuration. llama.cpp detects Qwen3 rerankers using the source directory name
or original README, emits `RANK` pooling, and extracts the `yes`/`no` rows into
`cls.output.weight`. The pipeline's temporary directory names use underscores,
so preserving the upstream README matters. Do not replace it with this output
model card before conversion, especially when using a local source directory.

CPU conversion is supported, but budget RAM for conversion/quantization and disk
for the source weights, Hugging Face cache, and BF16, Q8_0, and Q4_K_M outputs.
The larger models need substantially more memory and storage; `--keep` retains
the downloads and all three output files.

> [!IMPORTANT]
> This model is converted using https://github.com/gorse-cloud/convert.
