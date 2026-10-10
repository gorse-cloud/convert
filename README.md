# convert

[![Model Conversion](https://github.com/ggml-org/convert/actions/workflows/main.yml/badge.svg)](https://github.com/ggml-org/convert/actions/workflows/main.yml)

Automated pipeline for converting models to GGUF format and uploading them to HF.

Supported models: [models/](models/)

Sample models converted with this pipeline:

- https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF
- https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF
- https://huggingface.co/ggml-org/gpt-oss-20b-GGUF

## Usage

Requires `HF_TOKEN` with write permission to the target org.

```bash
# Convert all models
bash convert.sh --owner <org>

# Convert a single model
bash convert.sh --owner <org> --one gemma-4-12b

# Convert models matching a filter
bash convert.sh --owner <org> --filter '^gemma'

# Force re-convert (ignore SHA checks)
bash convert.sh --owner <org> --force

# Keep downloaded models and generated upload directories
bash convert.sh --owner <org> --keep

# Keep downloaded models and generated upload directories, but don't upload
bash convert.sh --owner <org> --keep --no-upload

# Use a specific llama.cpp commit
bash convert.sh --owner <org> --llama-commit <commit>

# Run via HF Jobs (cloud infrastructure)
bash hf-job.sh --owner <org>

# Run via HF Jobs with a custom hardware flavor
bash hf-job.sh --owner <org> --hardware cpu-basic
```

## Qwen3 rerankers

The `qwen3-reranker-0.6b`, `qwen3-reranker-4b`, and `qwen3-reranker-8b`
configs convert `Qwen/Qwen3-Reranker-{0.6B,4B,8B}` to BF16 and Q8_0 GGUFs.
These are dedicated query/document rerankers, not text-generation models.

```bash
# Convert the collection and upload to gorse-io; retain downloads and outputs
bash convert.sh --owner gorse-io --filter '^qwen3-reranker-' --keep

# Convert locally without uploading (HF_TOKEN is not required)
bash convert.sh --owner gorse-io --filter '^qwen3-reranker-' --keep --no-upload
```

Use a recent llama.cpp with Qwen3 reranker support. Keep the original downloaded
source `README.md` intact: the converter uses it (or the source folder name) to
identify rerankers and produce `RANK` pooling and the `yes`/`no` classifier head.
This matters because the pipeline's download folder names use underscores.
The generated upload README is separate from the downloaded source README.

CPU conversion is supported, but reserve RAM for conversion and quantization,
and disk for the original weights, Hugging Face cache, BF16 output, and Q8_0
output. The 4B/8B models require substantially more resources than 0.6B, and
`--keep` retains downloads and generated files. Place the repository and caches
on a volume with sufficient space (for example, set `HF_HOME` on that volume).

Serve with `llama-server --embedding --reranking --pooling rank` and call
`POST /v1/rerank` with `query` and `documents`. See each model's README for a
complete server command and request. The pipeline builds the quantizer only;
build or install `llama-server` separately.

## Qwen3-VL rerankers

The `qwen3-vl-reranker-2b` and `qwen3-vl-reranker-8b` configs convert
`Qwen/Qwen3-VL-Reranker-{2B,8B}` into BF16, Q8_0, Q4_K_M language GGUFs
and a separate F16 vision projector (`mmproj`). Preserve the source README
for reranker detection. Use llama.cpp with Qwen3-VL reranker support (inspected
at `abee0c8476b585b0d081d580b5ce21d0addc22c4`).

```bash
# Convert 2B first and retain artifacts without uploading
bash convert.sh --owner gorse-io --one qwen3-vl-reranker-2b --keep --no-upload

# Convert and upload both sizes
bash convert.sh --owner gorse-io --filter '^qwen3-vl-reranker-' --keep
```

The inspected llama.cpp `/v1/rerank` endpoint is **text-only**. Providing a
projector does not enable image/video reranking through that endpoint.
The source models are multimodal; the runtime integration is a separate concern.
See the model cards for text serving commands and limitations. Allow disk for
source weights, temporary tensors, three language outputs and the projector;
8B requires substantially more resources than 2B.

## Tests

The offline tests use only the Python standard library and Bash. They execute
the model conversion scripts with fake converter/quantizer executables; no
weights are downloaded and these tests do not validate real GGUF contents.

```bash
python3 -m unittest discover -s tests -v
```

## Notes

- Models are converted only if at least one of the source models has been updated
- The README.md files are always updated, regardless if the source models have been updated
- All models in [ggml-org](https://huggingface.co/ggml-org) are auto-converted by a [GitHub Actions workflow](https://github.com/ggml-org/convert/actions/workflows/main.yml) once per week

  ```bash
  HF_TOKEN=xxx bash hf-job.sh --owner ggml-org
  ```

- A maintainer from the [ggml-org/hf](https://github.com/orgs/ggml-org/teams/hf) team can start the workflow manually from the [Actions pane](https://github.com/ggml-org/convert/actions/workflows/main.yml)
- The conversion uses the latest `master` of [llama.cpp](https://github.com/ggml-org/llama.cpp) unless `--llama-commit` is set
