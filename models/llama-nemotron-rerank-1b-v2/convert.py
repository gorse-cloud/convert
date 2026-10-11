"""Local llama.cpp plugin; never execute the source model's remote Python code."""
import argparse
from pathlib import Path
import runpy
import sys

ARCHITECTURE = "LlamaBidirectionalForSequenceClassification"
RERANK_TEMPLATE = "<|begin_of_text|>question:{query} \n \n passage:{document}"


def add_rerank_template(writer):
    writer.add_chat_template([{"name": "rerank", "template": RERANK_TEMPLATE}])


def install_adapter():
    """Register in both lazy module discovery and the upstream class registry."""
    import conversion
    import gguf
    import numpy as np
    import torch
    from safetensors import safe_open
    from conversion.base import ModelBase
    from conversion.llama import LlamaModel

    class ChunkedEmbedding(gguf.LazyChunkedTensor):
        def __init__(self, shard, name, chunk_rows=256):
            if chunk_rows <= 0:
                raise ValueError("chunk_rows must be positive")
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                shape = tuple(f.get_slice(name).get_shape())
            if len(shape) != 2:
                raise ValueError("embedding must be a matrix")

            def load(start, end):
                with safe_open(str(shard), framework="pt", device="cpu") as f:
                    return f.get_slice(name)[start:end].float().numpy()

            chunks = [lambda start=i, end=min(i + chunk_rows, shape[0]): load(start, end)
                      for i in range(0, shape[0], chunk_rows)]
            super().__init__(chunks, shape, np.float32)
            # Upstream's tensor pipeline sees a torch-compatible float32 input;
            # numpy() returns self, quantize() produces a normal chunked writer.
            self.dtype = torch.float32

    @ModelBase.register(ARCHITECTURE)
    class NemotronRerankerModel(LlamaModel):
        model_arch = gguf.MODEL_ARCH.LLAMA

        def set_gguf_parameters(self):
            if self.hparams.get("pooling") != "avg":
                raise ValueError("Only source avg pooling is supported")
            if self.hparams.get("temperature", 1.0) != 1.0:
                raise ValueError("Only temperature=1.0 is supported")
            super().set_gguf_parameters()
            self.gguf_writer.add_causal_attention(False)
            self.gguf_writer.add_pooling_type(gguf.PoolingType.RANK)
            self.gguf_writer.add_classifier_pooling_type(gguf.PoolingType.MEAN)
            self.gguf_writer.add_classifier_output_labels(["relevance"])

        def get_vocab_base(self):
            # AutoTokenizer may consult the custom model AutoConfig even when
            # tokenizer_class is built-in. Read only the tokenizer artifact.
            import json
            from transformers import PreTrainedTokenizerFast
            config = json.loads((self.dir_model / "tokenizer_config.json").read_text())
            special = {key: config[key] for key in ("bos_token", "eos_token", "pad_token", "unk_token")
                       if key in config}
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self.dir_model / "tokenizer.json"), **special)
            vocab = tokenizer.get_vocab()
            size = self.hparams["vocab_size"]
            if max(vocab.values()) >= size:
                raise ValueError("Tokenizer ids exceed configured vocabulary size")
            reverse = {index: token for token, index in vocab.items()}
            added = tokenizer.get_added_vocab()
            tokens, types = [], []
            for i in range(size):
                token = reverse.get(i, f"[PAD{i}]")
                if i not in reverse:
                    kind = gguf.TokenType.UNUSED
                elif token in added:
                    entry = tokenizer.added_tokens_decoder[i]
                    if not entry.normalized:
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                    kind = (gguf.TokenType.CONTROL if entry.special or self.does_token_look_special(token)
                            else gguf.TokenType.USER_DEFINED)
                    if kind == gguf.TokenType.USER_DEFINED:
                        token = token.replace("\u2581", " ")
                else:
                    kind = gguf.TokenType.NORMAL
                tokens.append(token)
                types.append(kind)
            return tokens, types, self.get_vocab_base_pre(tokenizer)

        def set_vocab(self):
            self._set_vocab_gpt2()
            # The literal BOS in the named rerank prompt is part of the input;
            # do not add another BOS automatically.
            self.gguf_writer.add_add_bos_token(False)
            add_rerank_template(self.gguf_writer)

        def get_tensors(self):
            if self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                for shard in sorted(self.dir_model.glob("*.safetensors")):
                    with safe_open(str(shard), framework="pt", device="cpu") as f:
                        names = list(f.keys())
                    for raw in names:
                        if raw == "model.embed_tokens.weight":
                            self.model_tensors[raw] = lambda shard=shard, raw=raw: ChunkedEmbedding(shard, raw)
            yield from super().get_tensors()

        def modify_tensors(self, data_torch, name, bid):
            if name == "score.weight":
                if tuple(data_torch.shape) != (1, self.hparams["hidden_size"]):
                    raise ValueError("score.weight must have shape [1, hidden_size]")
                # CLS_OUT is recognized by C++ but absent from Python's LLAMA
                # MODEL_TENSORS; bypass format_tensor_name only for this head.
                yield "cls.output.weight", data_torch.float()
                return
            if name == "lm_head.weight":
                raise ValueError("Unexpected vocabulary head in a scalar reranker")
            yield from super().modify_tensors(data_torch, name, bid)

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if new_name == "cls.output.weight":
                return gguf.GGMLQuantizationType.F32
            return super().tensor_force_quant(name, new_name, bid, n_dims)

    conversion.TEXT_MODEL_MAP[ARCHITECTURE] = "llama"
    return NemotronRerankerModel


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llama-cpp", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    llama = args.llama_cpp.resolve()
    sys.path[:0] = [str(llama), str(llama / "gguf-py")]
    install_adapter()
    sys.argv = [str(llama / "convert_hf_to_gguf.py"), *remaining]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
