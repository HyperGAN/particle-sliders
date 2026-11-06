"""Probe the real import preamble with only the Python standard library."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / 'packages/particle-sliders-core/src'
          / 'particle_sliders/textsliders/generate_listen.py')
PROBE = r'''
import builtins
import json
import os
from pathlib import Path
import runpy
import sys

source, arguments = Path(sys.argv[1]), json.loads(sys.argv[2])
sys.argv = [str(source), *arguments]
original_import = builtins.__import__

class RuntimeBoundary(Exception):
    pass

def import_stdlib_only(name, *args, **kwargs):
    if name == 'numpy':
        raise RuntimeBoundary
    if name.split('.')[0] not in sys.stdlib_module_names:
        raise AssertionError('Unexpected non-stdlib import: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_stdlib_only
try:
    runpy.run_path(str(source), run_name='__visibility_probe__')
except RuntimeBoundary:
    assert not ({'numpy', 'torch', 'soundfile', 'yaml', 'conceptmod', 'particle_sliders'} & sys.modules.keys())
    print(json.dumps({'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}))
else:
    raise AssertionError('The runtime import boundary was not reached')
'''


class EarlyVisibilityTests(unittest.TestCase):
    def test_explicit_cpu_masks_and_visible_device_precedence(self):
        cases = [
            (None, [], '0'),
            (None, ['--device', '2'], '2'),
            (None, ['--device=3'], '3'),
            (None, ['--device=-1'], '-1'),
            ('', [], ''),
            ('', ['--device', '0'], ''),
            ('', ['--device=2'], ''),
            (' \t ', ['--device', '1'], ' \t '),
            ('-1', [], '-1'),
            ('-1', ['--device', '0'], '-1'),
            ('-1', ['--device=1'], '-1'),
            (' -1 ', ['--device', '2'], ' -1 '),
            ('1', [], '1'),
            ('1', ['--device', '0'], '1'),
            ('2,4', [], '2'),
            ('2,4', ['--device', '0'], '2'),
            ('2,4', ['--device=1'], '4'),
            ('GPU-first,GPU-second', ['--device', '1'], 'GPU-second'),
        ]
        parent_visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
        for visible, arguments, expected in cases:
            with self.subTest(visible=visible, arguments=arguments):
                env = os.environ.copy()
                if visible is None:
                    env.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    env['CUDA_VISIBLE_DEVICES'] = visible
                completed = subprocess.run(
                    [sys.executable, '-I', '-S', '-c', PROBE, str(SOURCE), json.dumps(arguments)],
                    env=env, text=True, capture_output=True, check=True, timeout=10,
                )
                self.assertEqual(json.loads(completed.stdout)['visible'], expected)
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), parent_visibility)


if __name__ == '__main__':
    unittest.main()
