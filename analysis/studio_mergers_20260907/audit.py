"""Check source/artifact integrity, application, audio, reuse, and HTTP delivery."""
import json
from pathlib import Path
import urllib.request
import numpy as np
import soundfile as sf
from common import WORK, OLD, sha, write
from render import verify
from report import assignments


def main():
    spec=json.loads((WORK/'screen.json').read_text())
    verify(spec)
    state=json.loads((WORK/'renders.json').read_text())
    measured=json.loads((WORK/'measurements.json').read_text())
    assert state['screen_sha256']==sha(WORK/'screen.json')==measured['screen_sha256']
    assert state['renderer_sha256']==sha(WORK/'render.py')
    assert state['status']==measured['status']=='complete'
    assert set(state['records'])==set(measured['records'])=={j['id'] for j in spec['jobs']}
    for path,digest in measured['source_sha256'].items():assert sha(path)==digest
    reused=0
    for job in spec['jobs']:
        rec=state['records'][job['id']]
        assert rec['status']=='complete' and rec['apply_mode']=='merge'
        assert rec['result']['seed']==job['seed']
        if 'reuse' in job:
            reused+=1
            assert sha(job['reuse']['screen'])==job['reuse']['screen_sha256']
            source=json.loads((OLD/'renders.json').read_text())['records'][job['reuse']['job_id']]
            assert source['audio_sha256']==rec['audio_sha256']
        else:
            event=rec['apply_events'][-1]
            if job['method'] in ('ties','knots_ties'):
                assert event['mode']=='dense' and event['modules']==144
                assert event['method']==job['method']
                assert event['manifest_sha256']==job['lora_components'][0]['sha256']
            else:assert event['mode']=='ordinary' and event['components']==2
        for field in ('audio','excerpt'):assert sha(rec[field])==rec[field+'_sha256']
        data,sr=sf.read(rec['audio'],dtype='float32',always_2d=True)
        excerpt,esr=sf.read(rec['excerpt'],dtype='float32',always_2d=True)
        assert np.isfinite(data).all() and len(data)>0
        assert sr==esr and np.array_equal(excerpt,data[:20*sr])
        assert measured['records'][job['id']]['excerpt_sha256']==rec['excerpt_sha256']
        for field in ('audio','excerpt'):
            req=urllib.request.Request('http://127.0.0.1:7860/studio-mergers-20260907/'+Path(rec[field]).name,
                                        headers={'Range':'bytes=0-43'})
            with urllib.request.urlopen(req,timeout=10) as response:
                assert response.status==206 and len(response.read())==44
    assert reused==6
    for g in assignments(spec):
        assert {j['method'] for j in g['mixes']}=={'linear','ties','knots_ties'}
        for field in ('lyrics','caption','seed','requested_duration'):
            assert len({j[field] for j in g['mixes']+[g['reference']]})==1
    with urllib.request.urlopen('http://127.0.0.1:7860/studio-mergers-20260907/',timeout=10) as response:
        page=response.read().decode()
    assert '24/24 rendered; 0 errors. Status: complete.' in page
    assert page.count('Reveal methods')==6 and page.count('<audio ')==24
    result=dict(status='passed',clips=24,new_renders=18,reused=6,measurements=24,
                checks=['frozen inference sources and checkpoint hashes','dense artifact hashes',
                        '144 actual runtime hosts for each dense render','exact first-20-second excerpts',
                        'matched captions, lyrics and seeds within groups','baseline reuse provenance',
                        'all 48 audio URLs support range requests','six complete blind groups'])
    write(WORK/'audit.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
