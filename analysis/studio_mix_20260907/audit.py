"""Verify frozen inputs, all rendered files, excerpts and studio delivery."""
import json
from pathlib import Path
import urllib.request

import numpy as np
import soundfile as sf

from geometry import WORK, sha
from render import verify


def main():
    spec = json.loads((WORK/'screen.json').read_text())
    verify(spec)
    state = json.loads((WORK/'renders.json').read_text())
    measured = json.loads((WORK/'measurements.json').read_text())
    assert state['screen_sha256'] == sha(WORK/'screen.json')
    assert state['renderer_sha256'] == sha(WORK/'render.py')
    assert state['status'] == 'complete'
    assert measured['status'] == 'complete'
    assert measured['screen_sha256'] == state['screen_sha256']
    for path, digest in measured['source_sha256'].items():
        assert sha(path) == digest
    assert set(state['records']) == {j['id'] for j in spec['jobs']}
    for job in spec['jobs']:
        rec = state['records'][job['id']]
        assert rec['status'] == 'complete' and rec['apply_mode'] == 'merge'
        assert rec['result']['seed'] == job['seed']
        for field in ('audio','excerpt'):
            assert sha(rec[field]) == rec[field+'_sha256']
        data, sr = sf.read(rec['audio'], dtype='float32', always_2d=True)
        excerpt, esr = sf.read(rec['excerpt'], dtype='float32', always_2d=True)
        assert np.isfinite(data).all() and len(data) > 0
        assert sr == esr and np.array_equal(excerpt, data[:20*sr])
        assert measured['records'][job['id']]['excerpt_sha256'] == rec['excerpt_sha256']
        for field in ('audio','excerpt'):
            url = 'http://127.0.0.1:7860/studio-mix-20260907/'+Path(rec[field]).name
            req = urllib.request.Request(url, headers={'Range':'bytes=0-43'})
            with urllib.request.urlopen(req, timeout=10) as response:
                assert response.status == 206 and len(response.read()) == 44
    with urllib.request.urlopen('http://127.0.0.1:7860/studio-mix-20260907/', timeout=10) as response:
        page = response.read().decode()
    assert '25/25 rendered; 0 errors. Status: complete.' in page
    assert page.count('Reveal energy settings') == 3
    result = dict(status='passed', renders=25, measurements=25,
                  checks=['frozen inputs', 'studio merge mode', 'seeds', 'full audio hashes',
                          'exact first-20-second excerpts', 'measurement hashes',
                          'all 50 audio URLs support range requests', 'complete listening page'])
    (WORK/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
