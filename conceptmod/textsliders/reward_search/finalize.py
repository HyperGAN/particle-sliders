"""Audit completed outputs and publish a local comparison page automatically."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

from ..reward_sliders.specs import WORKSPACE,RewardSpec,digest,sha,read_json,write_json
from ..reward_sliders.experiment import verify
from ..reward_sliders.directions import paired_statistics
from .setup import DEFAULT_RUN
from .report import report


def audit(run):
    m=read_json(run/'manifest.json');verify(m)
    for source,expected in read_json(run/'controller-sources.json').items():assert sha(source)==expected
    valid_manifests={digest(m),*m.get('compatible_manifest_sha256',[])}
    count=0;stages=defaultdict(list)
    with sqlite3.connect(run/'queue.sqlite') as db:
        jobs=db.execute('SELECT id,stage,payload,status,gpu FROM jobs ORDER BY rowid').fetchall()
    for ident,stage,payload,state,gpu in jobs:
        assert state=='complete',ident
        job=json.loads(payload);family=job['family']
        for arm in job['arms']:
            name=f"{family['family']}-s{job['seed']}-{arm['name']}"
            row=read_json(run/'stages'/stage/'observations'/f'{name}.json')
            assert row['status']=='complete' and row['reward']['valid']
            assert row['arm']==arm and row['seed']==job['seed'] and row['family']==family['family']
            assert row['provenance']['physical_gpu']==gpu
            assert row['provenance']['manifest_sha256'] in valid_manifests
            assert row['provenance']['family_sha256']==digest(family)
            assert sha(row['audio'])==row['audio_sha256']==row['reward']['audio_sha256']
            assert row['reward']['reward_spec_sha256']==digest(asdict(RewardSpec(**m['reward_spec'])))
            if row.get('trajectory'):assert sha(row['trajectory'])==row['trajectory_sha256']
            windows=row['reward']['windows']
            assert [(w['start_s'],w['end_s']) for w in windows]==[(0.,10.),(10.,20.)]
            assert abs(sum(w['axes']['CE'] for w in windows)/2-row['reward']['scalar'])<1e-9
            info=sf.info(row['audio']);assert info.frames/info.samplerate>=20
            data,rate=sf.read(row['audio'],frames=20*info.samplerate,dtype='float32',always_2d=True)
            assert np.isfinite(data).all()
            rms=float(np.sqrt(np.mean(data.astype('float64')**2)))
            assert abs(rms-row['reward']['normalization_rms'])<1e-10
            stages[stage].append(row);count+=1
    result=read_json(run/'results.json')
    assert result['original_v1_vs_off']==paired_statistics(stages['final'],'v1-original')
    for name,group in result['comparisons'].items():
        assert group['vs_off']==paired_statistics(stages['final'],name)
        assert group['vs_original_v1']==paired_statistics(stages['final'],name,'v1-original')
    candidate=read_json(run/'final-candidate.json')
    assert candidate['candidate']==result['candidate'] and candidate['selected_before_final_audio']
    from safetensors.torch import load_file
    import torch
    torch.set_num_threads(4)
    checkpoints=[]
    for path in sorted((run/'students').glob('*/*.safetensors')):
        metadata=read_json(path.with_suffix('.json'));tensors=load_file(str(path))
        assert sha(path)==metadata['weights_sha256']
        assert metadata['rank']==metadata['alpha']==8 and len(tensors)==432
        alphas=[v for k,v in tensors.items() if k.endswith('.alpha')]
        assert len(alphas)==144 and all(float(v)==8 for v in alphas)
        assert all(torch.isfinite(v).all() for v in tensors.values())
        assert all(v.shape[0]==8 for k,v in tensors.items() if k.endswith('.lora_down.weight'))
        assert all(v.shape[1]==8 for k,v in tensors.items() if k.endswith('.lora_up.weight'))
        checkpoints.append(dict(path=str(path),sha256=sha(path),tensors=432,rank=8,alpha=8))
    for path in (run/'audit').glob('gpu*-off-restoration.json'):assert read_json(path)['exact']
    for path in (run/'students').glob('*/off-restoration.json'):assert read_json(path)['exact']
    write_json(run/'audit/final-integrity.json',dict(passed=True,observations=count,
        stages={name:len(rows) for name,rows in stages.items()},matched_gpu_assignment=True,
        audio_and_trajectory_hashes_verified=True,score_aggregation_verified=True,
        checkpoints=checkpoints,manifest_sha256=sha(run/'manifest.json'),
        result_sha256=sha(run/'results.json'),script_sha256=sha(__file__)))


def listening_page(run):
    result=read_json(run/'results.json');m=read_json(run/'manifest.json')
    candidate=result['candidate_name'];destination=WORKSPACE/'app/static/reward-ce-v2'
    (destination/'audio').mkdir(parents=True,exist_ok=True)
    pairs=[]
    for family in (f for f in m['families'] if f.get('group')=='final'):
        for seed in m['search']['final_seeds']:
            rows={label:read_json(run/'stages/final/observations'/f"{family['family']}-s{seed}-{name}.json")
                  for label,name in [('Off','off'),('v1','v1-original'),('Candidate',candidate)]}
            target=min(.1,*(r['reward']['normalization_rms'] for r in rows.values()))
            pair=dict(family=family['family'],seed=seed,caption=family['caption'],lyrics=family['lyrics'],tracks={})
            for label,row in rows.items():
                source=Path(row['audio']);assert sha(source)==row['audio_sha256']
                stem=row['id'];raw=destination/'audio'/f'{stem}.wav'
                if not raw.exists():os.link(source,raw)
                playback=destination/'audio'/f'{stem}-matched.mp3'
                if not playback.exists():
                    gain=target/row['reward']['normalization_rms']
                    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(source),
                        '-af',f'volume={gain:.12g}','-c:a','libmp3lame','-q:a','2','-threads','1',str(playback)],check=True)
                pair['tracks'][label]=dict(ce=row['reward']['scalar'],playback='audio/'+playback.name,
                                           original='audio/'+raw.name,source_sha256=sha(source))
            pairs.append(pair)
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reward slider comparisons</title><style>
:root{color-scheme:dark}body{margin:0;background:#101419;color:#eef2f6;font:16px/1.55 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:32px 20px}h1{font-size:38px}p{max-width:800px}#cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}
article{padding:20px;border:1px solid #34404e;background:#1a2028;border-radius:12px}h2{font-size:19px}button,select{font:inherit;padding:8px 12px;margin:5px 5px 5px 0;background:#25313d;color:inherit;border:1px solid #657484;border-radius:7px}
button[aria-pressed=true]{background:#b4edbd;color:#132218}audio{width:100%;margin:12px 0}a{color:#aad9ff}pre{white-space:pre-wrap;font:14px/1.5 system-ui,sans-serif}details{margin-top:12px}
@media(max-width:700px){#cards{grid-template-columns:1fr}}
</style><main><h1>Reward slider comparisons</h1><p>Compare Off, the original v1 LoRA, and the candidate selected before these new prompts were rendered. All 32 matched final pairs are included. Playback levels are matched across each set; original WAV downloads are available.</p><p>Recorded outcome: <strong>__OUTCOME__</strong>. These examples evaluate the first 20 seconds.</p><button id="pause">Pause all</button><div id="cards"></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const pairs=JSON.parse(document.querySelector('#data').textContent),players=[];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const family of [...new Set(pairs.map(p=>p.family))]){
 const choices=pairs.filter(p=>p.family===family),card=document.createElement('article');
 card.innerHTML=`<h2>${esc(family)}</h2><label>Seed <select>${choices.map((p,i)=>`<option value="${i}">${p.seed}</option>`).join('')}</select></label><div>${['Off','v1','Candidate'].map(a=>`<button data-arm="${a}" aria-pressed="${a==='Off'}">${a}</button>`).join('')}</div><audio controls preload="none"></audio><p class="links"></p><details><summary>Scores</summary><p class="scores"></p></details><details><summary>Prompt and lyrics</summary><pre></pre></details>`;
 document.querySelector('#cards').append(card);const audio=card.querySelector('audio');let index=0,arm='Off',revision=0;
 function load(carry=true){const time=carry?audio.currentTime:0,playing=!audio.paused,token=++revision,pair=choices[index],track=pair.tracks[arm];audio.pause();audio.src=track.playback;
  card.querySelectorAll('[data-arm]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.arm===arm)));
  card.querySelector('.links').innerHTML=Object.entries(pair.tracks).map(([label,t])=>`<a href="${t.original}" download>${label} WAV</a>`).join(' · ');
  card.querySelector('.scores').textContent=Object.entries(pair.tracks).map(([label,t])=>`${label}: ${t.ce.toFixed(3)}`).join(' · ');
  card.querySelector('pre').textContent=pair.caption+'\\n\\n'+pair.lyrics;
  audio.onloadedmetadata=()=>{if(token!==revision)return;audio.currentTime=Math.min(time,Math.max(0,audio.duration-.01));if(playing)audio.play().catch(()=>{});};if(playing||time>0)audio.load();
 }
 card.querySelectorAll('[data-arm]').forEach(b=>b.onclick=()=>{arm=b.dataset.arm;load();});card.querySelector('select').onchange=e=>{index=Number(e.target.value);load(false);};
 audio.onplay=()=>players.forEach(p=>{if(p!==audio)p.pause();});players.push(audio);load(false);
}
document.querySelector('#pause').onclick=()=>players.forEach(p=>p.pause());
</script></html>'''
    page=template.replace('__OUTCOME__',html.escape(result['decision'].replace('_',' '))).replace('__DATA__',
        json.dumps(pairs,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026'))
    sys.path.insert(0,str(WORKSPACE));from app.rewriter import _artist_name_hit
    assert not _artist_name_hit('',page)
    (destination/'index.html').write_text(page)
    write_json(destination/'pairs.json',dict(pairs=pairs,candidate=result['candidate'],outcome=result['decision']))
    write_json(run/'audit/listening-page.json',dict(url='http://192.168.1.90:7860/reward-ce-v2/',
        pairs=32,page_sha256=sha(destination/'index.html'),original_audio_unchanged=True,level_matched=True))


def finalize(run,wait=False):
    run=Path(run)
    if wait:
        print('Waiting for the scored campaign to finish.',flush=True)
        while not (run/'experiment-done').exists():time.sleep(15)
    state=read_json(run/'status.json')
    if state['stage']!='complete':
        report(run);write_json(run/'audit/finalization-error.json',dict(error='Campaign did not complete',status=state))
        raise RuntimeError('Campaign did not complete; see retained status and service logs')
    audit(run)
    result=read_json(run/'results.json')
    for path in (run/'students').glob('*/*.safetensors'):
        sidecar=path.with_suffix('.json');metadata=read_json(sidecar)
        archive=run/'audit/pre-evaluation-sidecars'/sidecar.name
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes(sidecar.read_bytes())
        metadata['reward']['interpretation']='research CE student; development and fixed-duration final evaluation recorded'
        selected=str(path)==result['candidate'].get('checkpoint')
        metadata['reward']['evaluation']=dict(selected_for_final=selected,
            selected_multiplier=result['candidate'].get('coefficient') if selected else None,
            decision=result['decision'],results=str(run/'results.json'),default='off')
        assert sha(path)==metadata['weights_sha256'];write_json(sidecar,metadata)
    listening_page(run);report(run)
    write_json(run/'audit/finalization.json',dict(complete=True,time_unix=time.time(),
        script_sha256=sha(__file__),report_sha256=sha(run/'README.md'),
        decision=read_json(run/'results.json')['decision']))
    print('Final evidence audit and comparison page completed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    try:finalize(args.run_dir,args.wait)
    except Exception as error:
        write_json(args.run_dir/'audit/finalization-error.json',dict(error=repr(error),time_unix=time.time()))
        raise
