"""Offline contract tests: execute conversion scripts without model downloads."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SIZES = ("0.6B", "4B", "8B")

# These stand-ins record CLI arguments and create tiny placeholders, not GGUFs.
CONVERTER = '''import json, os, pathlib, sys
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps(["convert", *sys.argv[1:]]) + "\\n")
if os.environ.get("FAIL_STAGE") == "convert":
    sys.exit(7)
pathlib.Path(sys.argv[sys.argv.index("--outfile") + 1]).write_text("test BF16")
'''
QUANTIZER = '''#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps(["quantize", *sys.argv[1:]]) + "\\n")
if os.environ.get("FAIL_STAGE") == "quantize":
    sys.exit(8)
assert pathlib.Path(sys.argv[1]).is_file()
pathlib.Path(sys.argv[2]).write_text("test Q8_0")
'''


class Qwen3RerankerTests(unittest.TestCase):
    def model_dir(self, size):
        return ROOT / "models" / f"qwen3-reranker-{size.lower()}"

    def run_conversion(self, size, fail_stage=""):
        script = self.model_dir(size) / "convert.sh"
        self.assertTrue(script.is_file(), f"Missing conversion script: {script}")
        with tempfile.TemporaryDirectory(prefix="reranker-test-", dir=ROOT) as tmp:
            tmp = Path(tmp)
            # Spaces exercise shell quoting; README exercises source preservation.
            source = tmp / "source model"
            source.mkdir()
            readme = "# Qwen3-Reranker-" + size + "\nOriginal source README\n"
            (source / "README.md").write_text(readme)
            output = tmp / "output files"
            output.mkdir()
            llama = tmp / "llama cpp"
            (llama / "build" / "bin").mkdir(parents=True)
            (llama / "convert_hf_to_gguf.py").write_text(CONVERTER)
            quantizer = llama / "build" / "bin" / "llama-quantize"
            quantizer.write_text(QUANTIZER)
            quantizer.chmod(0o755)
            log = tmp / "commands.jsonl"
            env = dict(os.environ, PATH_PRIMARY=str(source), COMMAND_LOG=str(log),
                       FAIL_STAGE=fail_stage)
            result = subprocess.run(["bash", str(script), str(output), str(llama)],
                                    env=env, capture_output=True, text=True)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            manifest = output / ".produced_files"
            files = manifest.read_text().splitlines() if manifest.exists() else None
            artifacts = sorted(p.name for p in output.glob("*.gguf"))
            self.assertEqual((source / "README.md").read_text(), readme)
            return result, commands, files, artifacts, source, output

    def test_configs_register_all_sizes(self):
        for size in SIZES:
            with self.subTest(size=size):
                config = self.model_dir(size) / "config.sh"
                self.assertTrue(config.is_file(), f"Missing config: {config}")
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; printf "%s\\n" "$DISPLAY_NAME" "$DEST_REPO" "$DEP_PRIMARY"',
                     "bash", str(config)], capture_output=True, text=True, check=True)
                name = f"Qwen3-Reranker-{size}"
                self.assertEqual(result.stdout.splitlines(), [name, f"{name}-GGUF", f"Qwen/{name}"])

    def test_conversion_creates_bf16_q8_and_manifest(self):
        for size in SIZES:
            with self.subTest(size=size):
                result, commands, files, artifacts, source, output = self.run_conversion(size)
                self.assertEqual(result.returncode, 0, result.stderr)
                name = f"Qwen3-Reranker-{size}"
                bf16, q8 = f"{name}-BF16.gguf", f"{name}-Q8_0.gguf"
                self.assertEqual(commands, [
                    ["convert", str(source), "--outfile", str(output / bf16),
                     "--outtype", "bf16", "--model-name", name],
                    ["quantize", str(output / bf16), str(output / q8), "Q8_0"],
                ])
                self.assertEqual(files, [bf16, q8])
                self.assertEqual(artifacts, sorted([bf16, q8]))

    def test_failed_conversion_does_not_publish_manifest(self):
        for size in SIZES:
            for stage, code in (("convert", 7), ("quantize", 8)):
                with self.subTest(size=size, stage=stage):
                    result, commands, files, artifacts, _, _ = self.run_conversion(size, stage)
                    self.assertEqual(result.returncode, code, result.stderr)
                    self.assertIsNone(files)
                    self.assertEqual([command[0] for command in commands],
                                     ["convert"] if stage == "convert" else ["convert", "quantize"])
                    self.assertEqual(len(artifacts), 0 if stage == "convert" else 1)

    def test_model_cards_describe_reranking_not_generation(self):
        for size in SIZES:
            with self.subTest(size=size):
                readme = self.model_dir(size) / "README.md"
                self.assertTrue(readme.is_file(), f"Missing model card: {readme}")
                text = readme.read_text()
                self.assertIn(f"Qwen/Qwen3-Reranker-{size}", text)
                self.assertIn(f"__owner__/Qwen3-Reranker-{size}-GGUF", text)
                self.assertIn("pipeline_tag: text-ranking", text)
                self.assertNotIn("pipeline_tag: text-generation", text)
                self.assertIn("--reranking", text)
                self.assertIn("--embedding", text)
                self.assertIn("--pooling rank", text)
                self.assertIn("/v1/rerank", text)
                self.assertIn("not text generation", text)
                self.assertIn("README.md", text)


if __name__ == "__main__":
    unittest.main()
