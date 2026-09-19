"""Shared paths and provenance for the isolated merger study."""
import hashlib
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
APP_ROOT = WORK.parents[2]
OLD = WORK.parent/'studio_mix_20260907'
OUTPUT = APP_ROOT/'sliders-conceptmod/eval/listen/studio-mergers-20260907'
PAIRS = [('female', 'pop'), ('country', 'indie-rock'), ('house', 'acoustic-folk')]
sys.path.insert(0, str(APP_ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)
