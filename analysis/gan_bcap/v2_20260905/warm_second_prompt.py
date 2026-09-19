"""Precompute unchanged CPU measurements for an early, fixed render subset.

Warm one fixed, nonoverlapping slice of candidates. The default is the first
three; a second worker handles candidates 3:6. EMA remains for the final scorer,
leaving a rendering buffer after cache warming. No study output is overwritten.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from analysis.gan_bcap.autonomous_audio import Judge
from conceptmod.textsliders.gan_v2.data import sha

torch.set_num_threads(4)
start, stop = map(int, sys.argv[1:]) if len(sys.argv) == 3 else (0, 3)
if (start, stop) not in ((0, 3), (3, 6)):
    raise ValueError('Expected one of the two declared, nonoverlapping cache slices')
folder = ROOT / 'eval/listen/gan-v2-20260905/prompt-01'
spec_path = folder / 'render_spec.json'
while not spec_path.exists():
    time.sleep(10)
spec = json.loads(spec_path.read_text())
assert sha(spec['prompts']) == spec['prompts_sha256']
rows = yaml.safe_load(Path(spec['prompts']).read_text())
if isinstance(rows, dict):
    rows = rows['rows']
sheet = rows[spec['row']]['lyrics']
judge = Judge()
seen = set()
for checkpoint in spec['checkpoints'][start:stop]:
    for seed in spec['seeds']:
        current = folder / f'{Path(checkpoint["path"]).stem}-s{seed}'
        while not (current / 'checkpoint.json').exists():
            time.sleep(10)
        first = folder / f'{Path(spec["checkpoints"][0]["path"]).stem}-s{seed}'
        paths = [first / '01_slider_neutral_base_zero.wav',
                 next(first.glob('03_REF_prompt_*_no_slider.wav')),
                 next(current.glob('02_slider_*_plus1.wav'))]
        for path in paths:
            if path in seen:
                continue
            judge.measure(path, sheet, 'gender', spec['duration'])
            seen.add(path)
            print('Measured', path.parent.name, path.name, flush=True)
print('Finished fixed cache subset:', len(seen), 'clips', flush=True)
