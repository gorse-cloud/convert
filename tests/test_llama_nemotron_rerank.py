"""Offline conversion contracts and tiny synthetic GGUF tests; never inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/llama-nemotron-rerank-1b-v2"
NAME = "llama-nemotron-rerank-1b-v2"

DRIVER = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
if "--help" in args:
    print("--tensor-type tensor_name=ggml_type")
    sys.exit(0)
kind = "convert" if "--outfile" in args else "quantize"
with open(os.environ["COMMAND_LOG"], "a") as f:
    f.write(json.dumps([kind, *args]) + "\\n")
if os.environ.get("FAIL_STAGE") in (kind, args[-1]):
    sys.exit(7)
out = args[args.index("--outfile") + 1] if kind == "convert" else args[-2]
pathlib.Path(out).write_text("synthetic shell placeholder, NOT a GGUF")
'''

SEMANTIC = r'''
import importlib.util, json, pathlib, sys, tempfile
import numpy as np
import torch
from safetensors.torch import save_file
model, llama = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(llama), str(llama / 'gguf-py'), str(model)]
import gguf
import conversion
from conversion.base import ModelBase
from conversion.llama import LlamaModel
spec = importlib.util.spec_from_file_location('nemotron', model / 'convert.py')
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)
cls = plugin.install_adapter()
assert conversion.get_model_class('LlamaBidirectionalForSequenceClassification') is cls
with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    config = dict(model_type='llama', architectures=['LlamaBidirectionalForSequenceClassification'], pooling='avg',
                  temperature=1.0, hidden_size=8, num_attention_heads=2, num_key_value_heads=1,
                  num_hidden_layers=1, vocab_size=513, intermediate_size=16,
                  max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=500000.0)
    (tmp / 'config.json').write_text(json.dumps(config))
    score = torch.arange(8, dtype=torch.bfloat16).reshape(1,8)
    emb = torch.arange(513*8, dtype=torch.float32).reshape(513,8).to(torch.bfloat16)
    save_file({'score.weight':score, 'model.embed_tokens.weight':emb}, str(tmp/'model.safetensors'))
    obj = cls(tmp, gguf.LlamaFileType.MOSTLY_BF16, tmp/'tiny.gguf')
    obj.set_gguf_parameters()
    # Exercise real tokenizer serialization without AutoConfig/remote code.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    tokenizer = Tokenizer(WordLevel({'<|begin_of_text|>':0, 'x':1, '<unk>':2}, unk_token='<unk>'))
    tokenizer.save(str(tmp/'tokenizer.json'))
    (tmp/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'PreTrainedTokenizerFast', 'bos_token':'<|begin_of_text|>'}))
    remote_config = dict(config, model_type='llama_bidirec', bos_token_id=0, auto_map={'AutoConfig':'forbidden.RemoteConfig'})
    (tmp/'config.json').write_text(json.dumps(remote_config))
    obj.get_vocab_base_pre = lambda tokenizer: 'llama-bpe'
    from unittest.mock import patch
    with patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('AutoTokenizer must not load remote model config')):
        obj.set_vocab()
    got = dict(obj.get_tensors())
    assert isinstance(got['model.embed_tokens.weight'], gguf.LazyChunkedTensor)
    assert len(got['model.embed_tokens.weight']._chunks) == 3
    assert np.array_equal(np.concatenate([c() for c in got['model.embed_tokens.weight']._chunks]), emb.float().numpy())
    obj.prepare_tensors()
    obj.gguf_writer.write_header_to_file(tmp/'tiny.gguf')
    obj.gguf_writer.write_kv_data_to_file()
    obj.gguf_writer.write_tensors_to_file()
    obj.gguf_writer.close()
    reader = gguf.GGUFReader(str(tmp/'tiny.gguf'))
    def value(k): return reader.get_field(k).contents()
    assert value('llama.attention.causal') is False
    assert value('llama.pooling_type') == gguf.PoolingType.RANK.value
    assert value('llama.classifier.pooling_type') == gguf.PoolingType.MEAN.value
    assert value('llama.classifier.output_labels') == ['relevance']
    assert value('tokenizer.chat_template.rerank') == '<|begin_of_text|>question:{query} \n \n passage:{document}'
    head = next(t for t in reader.tensors if t.name == 'cls.output.weight')
    assert head.tensor_type == gguf.GGMLQuantizationType.F32
    assert list(head.shape) == [8,1]
    assert np.array_equal(head.data.reshape(1,8), score.float().numpy())
    from validate import validate_gguf
    validate_gguf(tmp/'tiny.gguf', tmp)
    # Validation rejects incompatible metadata and altered source classifier values.
    save_file({'score.weight':score+1}, str(tmp/'model.safetensors'))
    try: validate_gguf(tmp/'tiny.gguf', tmp)
    except ValueError: pass
    else: raise AssertionError('changed source head accepted')
    # Q/K RoPE permutations and norm naming must remain upstream Llama behavior.
    q = torch.arange(64, dtype=torch.float32).reshape(8,8)
    name = 'model.layers.0.self_attn.q_proj.weight'
    actual = list(obj.modify_tensors(q, name, 0))
    expected = list(LlamaModel.modify_tensors(obj, q, name, 0))
    assert actual[0][0] == expected[0][0]
    assert torch.equal(actual[0][1], expected[0][1])
    for key, bad in [('pooling','last'), ('temperature',2.0)]:
        obj.hparams[key] = bad
        try: obj.set_gguf_parameters()
        except ValueError: pass
        else: raise AssertionError('unsupported '+key+' accepted')
        obj.hparams[key] = config[key]
    try: list(obj.modify_tensors(torch.zeros(2,8), 'score.weight', None))
    except ValueError: pass
    else: raise AssertionError('multiple output scores accepted')
print('synthetic GGUF metadata, streamed embedding, F32 head, validation and Llama mapping passed')
'''


class NemotronTests(unittest.TestCase):
    def test_config(self):
        self.assertTrue((MODEL / 'config.sh').is_file(), 'Missing model config')
        result = subprocess.run(['bash', '-c', 'source "$1"; printf "%s\\n" "$DISPLAY_NAME" "$DEST_REPO" "$DEP_PRIMARY"',
                                 'bash', str(MODEL/'config.sh')], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), [NAME, NAME+'-GGUF', 'nvidia/'+NAME])

    def conversion(self, stage=''):
        self.assertTrue((MODEL/'convert.sh').is_file(), 'Missing model conversion script')
        with tempfile.TemporaryDirectory(prefix='nemotron-', dir=os.environ.get('TMPDIR')) as folder:
            folder = Path(folder)
            source, output, llama = [folder / p for p in ('source model', 'output files', 'llama cpp')]
            source.mkdir(); output.mkdir(); (llama/'build/bin').mkdir(parents=True)
            (source/'README.md').write_text('original source card')
            (source/'config.json').write_text('{"original": true}')
            (source/'LICENSE').write_text('test source license')
            (output/'.produced_files').write_text('stale.gguf\n')
            driver = folder/'python driver'; driver.write_text(DRIVER); driver.chmod(0o755)
            quantizer = llama/'build/bin/llama-quantize'; quantizer.write_text(DRIVER); quantizer.chmod(0o755)
            log = folder/'commands.jsonl'
            env = dict(os.environ, PATH_PRIMARY=str(source), PYTHON=str(driver), COMMAND_LOG=str(log), FAIL_STAGE=stage)
            result = subprocess.run(['bash', str(MODEL/'convert.sh'), str(output), str(llama)], env=env,
                                    capture_output=True, text=True)
            commands = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
            manifest = output/'.produced_files'
            files = manifest.read_text().splitlines() if manifest.exists() else None
            self.assertEqual((source/'README.md').read_text(), 'original source card')
            self.assertEqual((source/'config.json').read_text(), '{"original": true}')
            return result, commands, files, sorted(p.name for p in output.glob('*.partial.gguf'))

    def test_conversion_manifest_and_classifier_override(self):
        result, commands, files, partial = self.conversion()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(files, [NAME+'-'+q+'.gguf' for q in ('BF16','Q8_0','Q4_K_M')] + ['llama-classifier.patch','LICENSE'])
        self.assertEqual(partial, [])
        self.assertEqual([c[0] for c in commands], ['convert','quantize','quantize'])
        self.assertIn('--llama-cpp', commands[0])
        self.assertIn('--outtype', commands[0]); self.assertIn('bf16', commands[0])
        for command, q in zip(commands[1:], ['Q8_0','Q4_K_M']):
            self.assertEqual(command[-1], q)
            self.assertIn('--tensor-type', command)
            self.assertIn('cls.output.weight=f32', command)
            self.assertTrue(command[-3].endswith('-BF16.gguf'))
            self.assertTrue(command[-2].endswith('.partial.gguf'))

    def test_failure_removes_stale_manifest(self):
        for stage in ('convert', 'quantize', 'Q4_K_M'):
            with self.subTest(stage=stage):
                result, _, files, partial = self.conversion(stage)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIsNone(files)
                self.assertEqual(partial, [])

    def test_runtime_patch_and_card(self):
        self.assertTrue((MODEL/'llama-classifier.patch').is_file(), 'Missing runtime patch')
        patch = (MODEL/'llama-classifier.patch').read_text()
        self.assertIn('hparams.n_cls_out > 0', patch)
        self.assertIn('LLM_TENSOR_CLS_OUT', patch)
        self.assertNotIn('build_graph', patch)
        card = (MODEL/'README.md').read_text()
        for token in ('license_name: openmdw-1.1','Built with Llama','Llama 3.2', '__owner__/'+NAME+'-GGUF',
                      '--embedding --reranking --pooling rank', '-c 512 -b 512 -ub 512 --parallel 1',
                      'raw logits', 'text-only', 'git apply', 'not validated'):
            self.assertIn(token, card)

    def test_synthetic_gguf_semantics(self):
        self.assertTrue((MODEL/'convert.py').is_file(), 'Missing conversion adapter')
        python = os.environ.get('NEMOTRON_TEST_PYTHON', sys.executable)
        probe = subprocess.run([python,'-c','import torch, numpy, safetensors'], capture_output=True)
        if probe.returncode or not (ROOT/'llama.cpp/conversion').is_dir():
            self.skipTest('Semantic tests require torch/safetensors and the pinned llama.cpp checkout')
        result = subprocess.run([python,'-c',SEMANTIC,str(MODEL),str(ROOT/'llama.cpp')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
