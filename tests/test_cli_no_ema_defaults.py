"""Actual parser/shell defaults only: no base model, optimizer or GPU work."""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/particle-sliders-core/src'))
from particle_sliders.textsliders.gan_v2 import train as gan_module
assert Path(gan_module.__file__).resolve().is_relative_to(ROOT), "test must load actual isolated worktree source"
gan_args = gan_module.parse_args
from particle_sliders.textsliders import train_lora_yue2_arm_b as yue_module
assert Path(yue_module.__file__).resolve().is_relative_to(ROOT), "test must load actual isolated worktree source"
yue_args, run_recipe = yue_module.parse_args, yue_module.run_recipe
from particle_sliders.textsliders import yue2_particle_bridge
from particle_sliders.lifecycle import SessionConfig, SliderSession
import torch


class NoEMACLI(unittest.TestCase):
    def test_versioned_gan_defaults_to_raw(self):
        required = ['--prompts', 'unused.yaml', '--run-dir', 'unused']
        self.assertTrue(gan_args(required).no_ema)
        self.assertTrue(gan_args(required + ['--no-ema']).no_ema)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gan_args(required + ['--historical-ema'])

    def test_averaging_flags_are_exclusive(self):
        for parser, required in [(gan_args, ['--prompts', 'unused', '--run-dir', 'unused']),
                                 (yue_args, ['--recipe', 'particle_bridge', '--save_dir', 'unused'])]:
            with self.subTest(parser=parser.__module__), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser(required + ['--no-ema', '--historical-ema'])

    def test_yue_particle_defaults_to_no_shadow_recipe(self):
        required = ['--recipe', 'particle_bridge', '--save_dir', 'unused']
        current = yue_args(required)
        self.assertTrue(current.no_ema)
        self.assertIsNone(run_recipe(current, yue2_particle_bridge)['ema'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            yue_args(required + ['--historical-ema'])
        self.assertIsNone(yue2_particle_bridge.RECIPE['ema'])
        self.assertEqual(yue2_particle_bridge.HISTORICAL_RECIPE['ema'], .995)

    def test_historical_flag_cannot_enable_unrelated_recipe(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            yue_args(['--save_dir', 'unused', '--historical-ema'])

    def wrapper(self, *, arm, use_ema=None):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'arguments-only'
            executable.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            executable.chmod(0o700)
            env = dict(os.environ, MUSIC3_PYTHON=str(executable), ARM=arm)
            env.pop('USE_EMA', None)
            if use_ema is not None:
                env['USE_EMA'] = use_ema
            return subprocess.run(['bash', str(ROOT / 'scripts/train_lm_gan_v2.sh'), 'unused-new-run'],
                                  env=env, capture_output=True, text=True, timeout=10)

    def test_shell_all_arms_default_to_raw(self):
        for arm in ('baseline', 'fm_capped', 'repaired'):
            with self.subTest(arm=arm):
                result = self.wrapper(arm=arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                self.assertIn('--no-ema', args)
                self.assertNotIn('--historical-ema', args)

    def test_shell_historical_mode_is_rejected(self):
        result = self.wrapper(arm='repaired', use_ema='1')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Weight EMA is disabled', result.stderr)
        self.assertEqual(self.wrapper(arm='repaired', use_ema='invalid').returncode, 2)

    def test_current_shared_factory_defaults_stay_raw(self):
        config = SessionConfig('unused-backend', 'unused-new-cohort')
        self.assertIsNone(config.options.get('ema_decay'))
        self.assertEqual(inspect.signature(SliderSession.export).parameters['selection'].default, 'raw')
        self.assertEqual(inspect.signature(SliderSession.export_inference).parameters['selection'].default, 'raw')
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
