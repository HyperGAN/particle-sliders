"""Render the fixed-MMD endpoint, reusing byte-identical existing controls."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=ROOT/'eval/listen/gan-objective-20260905'
    out=ROOT/'eval/listen/gan-objective-mmd-20260905'
    spec=json.loads((previous/'render_spec.json').read_text())
    source=Path(spec['checkpoints'][0]['path'])
    candidate=ROOT/'models/conditional-mmd-cfg-research-20260905/conditional-mmd-cfg-research-20260905_step720.safetensors'
    completion=json.loads((candidate.parent/'completion.json').read_text())
    assert completion['step']==720
    assert sha(source)==spec['checkpoints'][0]['sha256']
    assert sha(Path(spec['prompts']))==spec['prompts_sha256']
    assert spec['seeds']==[7,23] and spec['duration']==20 and spec['row']==0
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    records=[]
    for seed in spec['seeds']:
        original=previous/f'{source.stem}-s{seed}'
        meta=json.loads((original/'checkpoint.json').read_text())
        assert meta['sha256']==sha(source) and meta['seed']==seed
        for weights in (source,candidate):
            dest=out/f'{weights.stem}-s{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            for audio in sorted(original.glob('*.wav')):
                if weights==candidate and audio.name=='02_slider_Female_plus1.wav':continue
                target=dest/audio.name
                if target.exists():assert sha(target)==sha(audio)
                else:shutil.copyfile(audio,target)
                records.append(dict(source=str(audio),destination=str(target),sha256=sha(target)))
    (out/'reuse.json').write_text(json.dumps(dict(source_render_spec=spec,
        physical_gpu=1,candidate_sha256=sha(candidate),copies=records),indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'analysis/gan_bcap/render_steps.py'),
        '--weights',str(source),str(candidate),'--out',str(out),
        '--prompts',spec['prompts'],'--row','0','--seeds','7','23','--duration','20'],check=True,cwd=ROOT)


if __name__=='__main__':main()
