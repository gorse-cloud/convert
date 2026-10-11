"""Inspect metadata and classifier bytes only; this utility never runs inference."""
import argparse
from pathlib import Path
import sys


def validate_gguf(path, source):
    import gguf
    import numpy as np
    from safetensors import safe_open
    from convert import RERANK_TEMPLATE

    reader = gguf.GGUFReader(str(path))

    def field(key):
        item = reader.get_field(key)
        if item is None:
            raise ValueError(f"Missing GGUF metadata: {key}")
        return item.contents()

    expected = {
        "general.architecture": "llama",
        "llama.attention.causal": False,
        "llama.pooling_type": gguf.PoolingType.RANK.value,
        "llama.classifier.pooling_type": gguf.PoolingType.MEAN.value,
        "llama.classifier.output_labels": ["relevance"],
        "tokenizer.chat_template.rerank": RERANK_TEMPLATE,
        "tokenizer.ggml.add_bos_token": False,
    }
    for key, value in expected.items():
        if field(key) != value:
            raise ValueError(f"Unexpected {key}: {field(key)!r}")
    tokens = field("tokenizer.ggml.tokens")
    bos = field("tokenizer.ggml.bos_token_id")
    if not isinstance(bos, int) or not 0 <= bos < len(tokens) or tokens[bos] != "<|begin_of_text|>":
        raise ValueError("BOS id does not identify <|begin_of_text|>")
    heads = [t for t in reader.tensors if t.name == "cls.output.weight"]
    if len(heads) != 1 or heads[0].tensor_type != gguf.GGMLQuantizationType.F32:
        raise ValueError("Exactly one F32 cls.output.weight is required")
    if any(t.name == "output.weight" for t in reader.tensors):
        raise ValueError("Unexpected vocabulary output head")
    found = []
    for shard in sorted(Path(source).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            if "score.weight" in f.keys():
                found.append(f.get_tensor("score.weight").float().numpy())
    if len(found) != 1 or found[0].ndim != 2 or found[0].shape[0] != 1:
        raise ValueError("Source must contain one scalar score.weight matrix")
    head, score = heads[0], found[0]
    if tuple(head.shape) != tuple(reversed(score.shape)):
        raise ValueError("Classifier shape differs from source")
    if not np.array_equal(head.data.reshape(score.shape), score):
        raise ValueError("Classifier F32 values differ from source")
    return {"file": str(path), "classifier_shape": list(score.shape), "classifier_dtype": "F32"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("gguf", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    for path in args.gguf:
        print(validate_gguf(path, args.source))


if __name__ == "__main__":
    main()
