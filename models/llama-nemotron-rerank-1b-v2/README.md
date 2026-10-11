---
library_name: gguf
pipeline_tag: text-ranking
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/llama-nemotron-rerank-1b-v2
tags:
  - gguf
  - reranker
  - cross-encoder
---

# llama-nemotron-rerank-1b-v2 GGUF

GGUF conversions of `nvidia/llama-nemotron-rerank-1b-v2` for
`__owner__/llama-nemotron-rerank-1b-v2-GGUF`.
Source inspected at revision `828765652b05bd439c9789d2a6d093db1caa1443`.
This is a **text-only** query/passage cross-encoder, not text generation.
It uses bidirectional Llama attention, mean pooling over the complete pair,
and a single learned relevance classifier. Outputs are **raw logits**, not
probabilities; larger values indicate more relevance. There is no sigmoid
or yes/no vocabulary projection in this conversion. Source temperature is 1.0.

## Files

- `llama-nemotron-rerank-1b-v2-BF16.gguf`
- `llama-nemotron-rerank-1b-v2-Q8_0.gguf`
- `llama-nemotron-rerank-1b-v2-Q4_K_M.gguf`
- `llama-classifier.patch` — required runtime loader change
- `LICENSE` — verbatim license from the downloaded NVIDIA source

All variants retain `cls.output.weight` as F32. Quantization can change model
outputs; quality and inference parity are **not validated** by the conversion
unit tests. Tests exercise synthetic GGUF serialization, not model inference.
The self-contained converter streams the large BF16 embedding in row chunks.
It does not rewrite the downloaded config or execute remote model Python code.
Source and generated model cards are separate.

## Required runtime patch

Ordinary unpatched llama.cpp binaries at the inspected revision do **not** load
this Llama classifier. Use llama.cpp commit
`abee0c8476b585b0d081d580b5ce21d0addc22c4` and apply the included patch yourself:

```bash
cd llama.cpp
# Run on the pinned checkout; replace the patch path if using an uploaded copy.
git apply --check ../models/llama-nemotron-rerank-1b-v2/llama-classifier.patch
git apply ../models/llama-nemotron-rerank-1b-v2/llama-classifier.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j2
```

The patch only loads `cls.output.weight` when the output-label count is positive.
Existing noncausal attention and central RANK/MEAN scalar classifier projection
are reused; there is no graph patch. The conversion script does not apply the
patch or build a server. Quantization requires `--tensor-type` support.

## Serving (example, not an inference validation)

```bash
llama.cpp/build/bin/llama-server \
  -hf __owner__/llama-nemotron-rerank-1b-v2-GGUF:Q4_K_M \
  --embedding --reranking --pooling rank \
  -c 512 -b 512 -ub 512 --parallel 1

curl http://localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is a cross-encoder?","documents":["A model that jointly encodes a query and document.","A weather forecast."]}'
```

Each bidirectional pair must fit entirely in **one microbatch**. These settings
are for pairs up to 512 tokens, including the prompt. For longer pairs, increase
`-c`, `-b`, and `-ub` together; source evaluation covers up to 8192 tokens.
Keep `--parallel 1` for this example. A small `-ub` that splits a pair is invalid
for this attention/pooling behavior.

The GGUF named `rerank` template includes exactly one literal BOS and the exact
source separators. In the escaped representation below, each `\n` is one actual
newline in the GGUF template:

```text
<|begin_of_text|>question:{query} \n \n passage:{document}
```

Automatic BOS insertion is disabled to prevent duplication. Do not replace the
prompt with a generic chat template or add EOS silently.

## Conversion and structural validation

```bash
bash convert.sh --one llama-nemotron-rerank-1b-v2 \
  --llama-commit abee0c8476b585b0d081d580b5ce21d0addc22c4 \
  --owner gorse-io --keep --no-upload

# Uses the same Python environment as llama.cpp conversion.
python3 models/llama-nemotron-rerank-1b-v2/validate.py \
  --llama-cpp llama.cpp --source /path/to/downloaded/source \
  /path/to/output/llama-nemotron-rerank-1b-v2-*.gguf
```

Validation checks GGUF architecture, bidirectional/RANK/MEAN metadata, exact
prompt and BOS, and the scalar classifier's F32 shape and values against source
safetensors. It never executes model inference. Reserve disk for the source,
three outputs and temporary output files. The manifest is published only after
all outputs and supplemental license/patch copies succeed; a failed rerun removes
a stale manifest. Successful stages can remain for debugging after a later failure.

## Attribution and license

Built with Llama. NVIDIA's source model card identifies the **OpenMDW License
Agreement, version 1.1 (openmdw-1.1)** and additionally references the **Llama 3.2
Community Model License Agreement**. Review both applicable terms before use
or redistribution. The conversion preserves the source `LICENSE` in the output.
Source attribution: NVIDIA `nvidia/llama-nemotron-rerank-1b-v2`; underlying model:
Meta `meta-llama/Llama-3.2-1B`. GGUF conversion does not replace those terms.
