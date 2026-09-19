"""Shared files for the studio combination and energy screen."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

WORK = Path(__file__).resolve().parent
APP_ROOT = WORK.parents[2]
OLD = WORK.parent / 'studio_mergers_20260907'
OUTPUT = APP_ROOT / 'sliders-conceptmod/eval/listen/studio-combo-energy-20260907'
PUBLIC = APP_ROOT / 'app/static/studio-combo-energy-20260907'
PAIRS = [('female', 'pop'), ('country', 'indie-rock'), ('house', 'acoustic-folk')]
URL = 'http://127.0.0.1:7860'
sys.path.insert(0, str(APP_ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def api(path, payload=None):
    req = urllib.request.Request(URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='GET' if payload is None else 'POST')
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def verify(spec):
    from app import sliders
    checks = {**spec['source_sha256'], str(sliders.REGISTRY_PATH): spec['registry_sha256']}
    checks[spec['fixture']['source_screen']] = spec['fixture']['source_screen_sha256']
    for ck in spec['checkpoints']:
        checks[ck['weights']] = ck['sha256']
        checks[str(sliders._sidecar_path(Path(ck['weights'])))] = ck['sidecar_sha256']
    for path, digest in checks.items():
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    for job in spec['jobs']:
        if sliders.resolve(job['sliders'], host_energy=job['energy']) != job['lora_components']:
            raise ValueError('Resolution changed for ' + job['id'])
