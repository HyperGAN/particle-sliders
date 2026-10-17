"""H3 public migration gate, executable with stdlib-only ``python -S``."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "conceptmod/textsliders/train_lora_minimax_h3.py"
DIAGNOSTIC = ROOT / "conceptmod/textsliders/diag_minimax_h3_uni.py"


class H3ClosureTests(unittest.TestCase):
    def load_trainer(self):
        spec = importlib.util.spec_from_file_location("h3_closure_fixture", TRAINER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_closed_apis_import_no_runtime_or_touch_backend(self):
        imported = []
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            imported.append(name)
            if name.split(".")[0] in {
                "torch", "yaml", "diffusers", "safetensors", "particlegan", "particle_sliders"
            }:
                raise AssertionError(f"closed route imported runtime: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            module = self.load_trainer()
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / "must-not-exist"
                commands = [[], ["--dummy"], ["--steps", "0", "--load_h3_lora", "missing"],
                            ["--diag"], ["--diag", "--load_h3_lora", "missing"]]
                for command in commands:
                    with self.subTest(command=command):
                        with self.assertRaisesRegex(module.H3MigrationBlocked, "response/critic protocol"):
                            module.main(command + ["--save_dir", str(output), "--prompts_file", "missing"])
                for action in (
                    lambda: module.train(object(), backend=object()),
                    lambda: module.emit_h3_samples(object(), object(), output, object()),
                    lambda: module.write_h3_sample_mp4(output, object(), fps=24),
                    lambda: module.build_backend(module.parse_args([])),
                ):
                    with self.assertRaises(module.H3MigrationBlocked):
                        action()
                self.assertFalse(output.exists())
        self.assertTrue(imported)

    def test_stale_clis_reject_with_no_source_tree_or_site_packages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package = root / "conceptmod/textsliders"
            package.mkdir(parents=True)
            shutil.copy2(TRAINER, package / TRAINER.name)
            shutil.copy2(DIAGNOSTIC, package / DIAGNOSTIC.name)
            # The only supplied family files are the two CLI façades. There is
            # no backend, model dependency, source checkout or installed core.
            cases = [
                (TRAINER.name, []),
                (TRAINER.name, ["--dummy"]),
                (TRAINER.name, ["--steps", "0", "--load_h3_lora", "missing"]),
                (TRAINER.name, ["--diag"]),
                (TRAINER.name, ["--dummy", "--steps", "0"]),
                (DIAGNOSTIC.name, []),
                (DIAGNOSTIC.name, ["--load_h3_lora", "missing"]),
            ]
            env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
            for script, args in cases:
                with self.subTest(script=script, args=args):
                    output = root / "must-not-exist"
                    result = subprocess.run(
                        [sys.executable, "-I", "-S", str(package / script), *args,
                         "--save_dir", str(output)],
                        cwd=root, env=env, capture_output=True, text=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("MiniMax-H3 execution is closed", result.stderr)
                    self.assertIn("community license", result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)
                    self.assertFalse(output.exists())
            help_result = subprocess.run(
                [sys.executable, "-I", "-S", str(package / TRAINER.name), "--help"],
                cwd=root, env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("CPU diagnostics only", help_result.stdout)
            diag_help = subprocess.run(
                [sys.executable, "-I", "-S", str(package / DIAGNOSTIC.name), "--help"],
                cwd=root, env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(diag_help.returncode, 0, diag_help.stderr)
            self.assertIn("native execution closed", diag_help.stdout)


if __name__ == "__main__":
    unittest.main()
