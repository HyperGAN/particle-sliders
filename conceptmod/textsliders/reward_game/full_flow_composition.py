"""Fresh composition under independent, fixed per-host energy settings."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import signal
import time
from .core import read,write,immutable,sha,digest,locked,IntegrityError,Store
from .full_flow_confirmation import verify_batch
from .acoustic_artifact import checkpoint,component
from ..reward_sliders.specs import digest as renderer_digest


def freeze(home,confirmation,name):
    from ..reward_sliders.render import resolve_styles
    from ..reward_sliders.evaluate import extra_style_hashes
    from ..reward_sliders.specs import validate_families
    home=Path(home).resolve();source=home/'confirmation'/confirmation;p,manifest=verify_batch(home,source)
    if not read(source/'scorecard.json')['batch_pass']:raise IntegrityError('A passed fresh batch is required before the composition check')
    if p['candidate']['multiplier']!=1.:raise IntegrityError('This separate composition version declares unit reward energy')
    # Fixed fixture indices and style additions are declared by source order,
    # without using individual candidate CE or diagnostic values to select them.
    selected=[deepcopy(manifest['families'][i]) for i in (1,2,6,7)]
    original_styles=[dict(f['style_multipliers']) for f in selected]
    original_styles[2]={'house':2.};original_styles[3]={'acoustic-folk':2.}
    for f,styles in zip(selected,original_styles):
        total=sum(abs(v) for v in styles.values())
        if abs(total-2.)>1e-8:raise IntegrityError('Composition starts with two units of style energy')
        f['family']=f"{name}-{f['family']}";f['style_multipliers']=dict(styles)
    validate_families(selected);m=deepcopy(manifest);m['families']=selected
    m['style_components']={f['family']:resolve_styles(f['style_multipliers']) for f in selected}
    for comps in m['style_components'].values():extra_style_hashes(m,comps)
    cases=[dict(id=f"{f['family']}-s{s}",family=f['family'],seed=s,physical_gpu=1) for f in selected for s in p['seeds']]
    c=p['candidate'];arms=[dict(name='style-reduced',checkpoint=None,multiplier=0.),dict(name='fixed-energy',checkpoint=c['path'],multiplier=1.)]
    protocol=dict(name=name,source_confirmation=confirmation,source_protocol_sha256=digest(p),candidate=c,cases=cases,arms=arms,
        manifest_sha256=digest(m),sources={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in ('full_flow_composition.py','acoustic_artifact.py','acoustic_renderer.py')},physical_gpu=1,new_clip_budget=16,
        version='full-flow-per-host-v1',
        host_energy_by_kind=dict(language_model=2.,transformer=1.),
        control_effective_energy=dict(language_model=2.,transformer=0.),
        candidate_effective_energy=dict(language_model=2.,transformer=1.),
        allocation='All deployed styles occupy the LM host. The acoustic candidate alone occupies the transformer host. Turning it on leaves the LM allocation unchanged; there is no transformer style to reduce.',
        original_style_energy=2.,reduced_style_energy=2.,reward_energy=1.,
        original_style_multipliers=original_styles,reduced_style_multipliers=[f['style_multipliers'] for f in selected],
        fixture_selection='fixed source family indices 1,2,6,7; two vocal and two instrumental; house and acoustic-folk instrumental style assignments declared before composition audio',
        gates=dict(wins=6,mean=.05,worst_floor=-.5,all_valid=True),research_complete=False)
    folder=home/'composition'/name
    if folder.exists():raise IntegrityError('Composition name already used')
    immutable(folder/'manifest.json',m);immutable(folder/'protocol.json',protocol)
    Store(home).event('composition_frozen',batch=name,protocol_sha256=digest(protocol),new_clip_budget=16)
    return protocol


def verify(home,folder):
    from .setup import verify as verify_game
    import json
    verify_game(home);p=read(folder/'protocol.json');m=read(folder/'manifest.json')
    entries=[json.loads(l) for l in (Path(home)/'ledger.jsonl').read_text().splitlines()]
    entries=[r for r in entries if r['kind']=='composition_frozen' and r['batch']==p['name']]
    if len(entries)!=1 or entries[0]['protocol_sha256']!=digest(p) or digest(m)!=p['manifest_sha256']:raise IntegrityError('Composition identity changed')
    for file,h in {**p['sources'],**m['style_hashes']}.items():
        if sha(file)!=h:raise IntegrityError('Composition source/style changed')
    c=p['candidate']
    if checkpoint(c['path'],c['multiplier'])!=c:raise IntegrityError('Composition checkpoint changed')
    return p,m


def score(home,name):
    import html
    home=Path(home).resolve();folder=home/'composition'/name;p,m=verify(home,folder);rows=[];invalid=[];audio=[]
    for c in p['cases']:
        values={};paths={}
        for arm in p['arms']:
            file=folder/'observations'/f"{c['id']}-{arm['name']}.json"
            if not file.exists():continue
            o=read(file)
            if o['arm']!=arm or o['provenance']['manifest_sha256']!=renderer_digest(m):raise IntegrityError('Composition observation changed')
            if o.get('audio'):
                if sha(o['audio'])!=o['audio_sha256']:raise IntegrityError('Composition audio changed')
                paths[arm['name']]=o['audio']
            if o['status']=='complete' and o['reward']['valid']:values[arm['name']]=o['reward']['scalar']
        if len(values)!=2:invalid.append(c['id'])
        else:rows.append(dict(c,delta=values['fixed-energy']-values['style-reduced'],ce=values,audio=paths))
        audio.append(dict(case=c['id'],audio=paths))
    deltas=[r['delta'] for r in rows];wins=sum(v>0 for v in deltas);mean=sum(deltas)/8 if len(deltas)==8 else None;worst=min(deltas) if deltas else None
    passed=not invalid and wins>=6 and mean>=.05 and worst>=-.5
    result=dict(name=name,passed=passed,valid_cases=len(rows),wins=wins,meaningful_wins=sum(v>=.02 for v in deltas),equal_family_gain=mean,
        worst_delta=worst,invalid=invalid,rows=rows,new_clips=sum(len(r['audio']) for r in audio),protocol_sha256=digest(p),research_complete=False)
    write(folder/'scorecard.json',result)
    summary=f'Fixed per-host-energy composition: {len(rows)}/8 valid, {wins}/8 wins, mean {mean}, worst {worst}, pass {passed}.\n'
    (folder/'scorecard.md').write_text(summary)
    body=''.join('<h3>'+html.escape(row['case'])+'</h3>'+''.join('<p>'+html.escape(arm)+'<audio controls preload="none" src="'+html.escape(os.path.relpath(path,folder))+'"></audio></p>' for arm,path in row['audio'].items()) for row in audio)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Composition check</title><h1>Fixed per-host-energy composition</h1><p>'+html.escape(summary)+'</p>'+body)
    return result


def run(home,name):
    import torch
    from .resources import gpu_lease
    from .acoustic_renderer import AcousticRenderer
    from ..reward_sliders.reward import CEReward
    from ..reward_sliders.specs import RewardSpec
    home=Path(home).resolve();folder=home/'composition'/name;p,m=verify(home,folder)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise IntegrityError('Composition owns physical GPU 1')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel);renderer=None
    with locked(folder/'batch.lock'),gpu_lease(home,1):
        scorer=CEReward(RewardSpec(**m['reward_spec']))
        try:
            for case in p['cases']:
                family=next(f for f in m['families'] if f['family']==case['family'])
                for arm in p['arms']:
                    file=folder/'observations'/f"{case['id']}-{arm['name']}.json"
                    if file.exists():
                        old=read(file)
                        if old['status']=='running':
                            old.update(status='interrupted',error='Interrupted output retained without reroll')
                            audio=folder/'audio'/f"{case['id']}-{arm['name']}.wav"
                            if audio.exists():old.update(audio=str(audio),audio_sha256=sha(audio))
                            write(file,old)
                        continue
                    if renderer is None:renderer=AcousticRenderer(folder,m,1)
                    extra=component(arm['checkpoint'],arm['multiplier']) if arm['checkpoint'] else None
                    began=time.time()
                    try:
                        row=renderer.observe(family,case['seed'],arm,scorer,extra=extra)
                        row.update(started_unix=began,finished_unix=time.time());write(file,row)
                    except BaseException as exc:
                        if file.exists():
                            row=read(file);path=folder/'audio'/f"{case['id']}-{arm['name']}.wav"
                            row.update(status='interrupted',error=repr(exc),started_unix=began,finished_unix=time.time())
                            if path.exists():row.update(audio=str(path),audio_sha256=sha(path))
                            write(file,row)
                        raise
                    Store(home).event('composition_case_finished',batch=name,case=case['id'],arm=arm['name'],status=row['status'])
        finally:
            if renderer is not None:
                renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
                exact=all(torch.equal(mod.weight.detach().cpu(),pristine) for mod,pristine in renderer.host._merge_state(renderer.device).pristine.items())
                write(folder/'off-restoration.json',dict(exact=exact,physical_gpu=1))
                if not exact:raise IntegrityError('Composition Off restoration failed')
        result=score(home,name);Store(home).event('composition_finished',batch=name,passed=result['passed'],new_clips=result['new_clips']);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--name',required=True)
    p.add_argument('action',choices=('freeze','run','score'));p.add_argument('--confirmation')
    a=p.parse_args();result=freeze(a.home,a.confirmation,a.name) if a.action=='freeze' else globals()[a.action](a.home,a.name)
    print(__import__('json').dumps(result,indent=2),flush=True)
