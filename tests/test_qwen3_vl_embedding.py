"""Offline shell contracts; placeholder outputs do not validate GGUF semantics."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_qwen3_reranker import CONVERTER, QUANTIZER

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("2B", "8B")


class Qwen3VLEmbeddingTests(unittest.TestCase):
    def test_configs(self):
        for size in SIZES:
            with self.subTest(size=size):
                config = ROOT / "models" / f"qwen3-vl-embedding-{size.lower()}" / "config.sh"
                self.assertTrue(config.is_file(), f"Missing config: {config}")
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; printf "%s\\n" "$DISPLAY_NAME" "$DEST_REPO" "$DEP_PRIMARY"',
                     "bash", str(config)], capture_output=True, text=True, check=True)
                name = f"Qwen3-VL-Embedding-{size}"
                self.assertEqual(result.stdout.splitlines(), [name, name + "-GGUF", "Qwen/" + name])

    def test_conversion_and_failure_manifest(self):
        for size in SIZES:
            for stage in ("", "convert", "quantize", "Q4_K_M", "mmproj"):
                with self.subTest(size=size, stage=stage):
                    script = ROOT / "models" / f"qwen3-vl-embedding-{size.lower()}" / "convert.sh"
                    self.assertTrue(script.is_file(), f"Missing conversion script: {script}")
                    with tempfile.TemporaryDirectory(prefix="vl-embedding-test-") as tmp:
                        tmp = Path(tmp)
                        source = tmp / "source model"
                        source.mkdir()
                        metadata = {
                            "README.md": f"# Qwen3-VL-Embedding-{size}\nOriginal source README\n",
                            "modules.json": json.dumps([{"type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling", "path": "1_Pooling"}]),
                            "1_Pooling/config.json": json.dumps({"pooling_mode": "lasttoken", "include_prompt": True}),
                            "chat_template.jinja": "Original source embedding template\n",
                        }
                        for name, text in metadata.items():
                            path = source / name
                            path.parent.mkdir(exist_ok=True)
                            path.write_text(text)
                        output = tmp / "output files"
                        output.mkdir()
                        manifest = output / ".produced_files"
                        manifest.write_text("stale.gguf\n")
                        llama = tmp / "llama cpp"
                        (llama / "build" / "bin").mkdir(parents=True)
                        converter = CONVERTER.replace(
                            'if os.environ.get("FAIL_STAGE") == "convert":',
                            'if (os.environ.get("FAIL_STAGE") == "convert" or (os.environ.get("FAIL_STAGE") == "mmproj" and "--mmproj" in sys.argv)):')
                        (llama / "convert_hf_to_gguf.py").write_text(converter)
                        quantizer = llama / "build" / "bin" / "llama-quantize"
                        quantizer.write_text(QUANTIZER)
                        quantizer.chmod(0o755)
                        log = tmp / "commands.jsonl"
                        env = dict(os.environ, PATH_PRIMARY=str(source), COMMAND_LOG=str(log), FAIL_STAGE=stage)
                        result = subprocess.run(["bash", str(script), str(output), str(llama)],
                                                env=env, capture_output=True, text=True)
                        for name, text in metadata.items():
                            self.assertEqual((source / name).read_text(), text)
                        if stage:
                            self.assertEqual(result.returncode, 7 if stage in ("convert", "mmproj") else 8, result.stderr)
                            self.assertFalse(manifest.exists(), "Failed conversion must not retain a stale manifest")
                            continue
                        self.assertEqual(result.returncode, 0, result.stderr)
                        name = f"Qwen3-VL-Embedding-{size}"
                        files = [f"{name}-{fmt}.gguf" for fmt in ("BF16", "Q8_0", "Q4_K_M")]
                        files.append(f"mmproj-{name}-F16.gguf")
                        self.assertEqual(manifest.read_text().splitlines(), files)
                        self.assertEqual(sorted(p.name for p in output.glob("*.gguf")), sorted(files))
                        commands = [json.loads(line) for line in log.read_text().splitlines()]
                        self.assertEqual(commands, [
                            ["convert", str(source), "--outfile", str(output / files[0]), "--outtype", "bf16", "--model-name", name],
                            ["quantize", str(output / files[0]), str(output / files[1]), "Q8_0"],
                            ["quantize", str(output / files[0]), str(output / files[2]), "Q4_K_M"],
                            ["convert", str(source), "--mmproj", "--outfile", str(output / files[3]), "--outtype", "f16", "--model-name", name],
                        ])


if __name__ == "__main__":
    unittest.main()
