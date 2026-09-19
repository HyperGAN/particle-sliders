"""Separately frozen acoustic-host fresh confirmation. A passing batch still requires replication."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import signal
import time
import json
from .core import read, write, immutable, sha, digest, locked, IntegrityError, Store
from .setup import verify
from .acoustic_artifact import checkpoint
from .core import checkpoint as lm_checkpoint
from .acoustic_confirmation_support import component
from .confirmation_amendments import effective_sources
from ..reward_sliders.specs import digest as renderer_digest
from .confirmation_stats import summarize


def freeze(home, evaluation, fixtures, name, replication_of=None):
    from .acoustic_evaluate import inspect
    from ..reward_sliders.specs import validate_families
    from ..reward_sliders.render import resolve_styles
    from ..reward_sliders.evaluate import extra_style_hashes
    home=Path(home).resolve();game,base=verify(home)
    card=inspect(home,evaluation)
    if card['stage']!=16 or card['valid_cases']!=16 or not card['advance']:
        raise IntegrityError('Fresh rendering requires a strict completed development pass')
    candidate=checkpoint(card['candidate_provenance']['path'],card['candidate_provenance']['multiplier'])
    if candidate!=card['candidate_provenance']:raise IntegrityError('Evaluated candidate changed')
    original=lm_checkpoint(game['original_reference']['checkpoint'],1.)
    draft=read(fixtures);families=draft['families'];seeds=draft['seeds']
    validate_families(families)
    if len(families)!=8 or len(set(f['family'] for f in families))!=8 or len(seeds)!=2 or len(set(seeds))!=2:
        raise IntegrityError('Need eight unique fresh families and two unique seeds')
    voices={v:sum(f['voice']==v for f in families) for v in ('female','male','unspecified','instrumental')}
    if set(voices.values())!={2}:raise IntegrityError('Unbalanced confirmation voices')
    previous=list(base['families']);used_seeds={c['seed'] for c in game['cases']}
    parent=Path(read(home/'game.json')['parent_home'])
    for old in list((home/'confirmation').glob('*/protocol.json'))+list((parent/'confirmation').glob('*/protocol.json')):
        p=read(old);previous+=read(old.parent/'manifest.json')['families'];used_seeds.update(p['seeds'])
    for family in families:
        if any(family['caption']==p['caption'] or family['lyrics']==p['lyrics'] or family['family']==p['family'] for p in previous):
            raise IntegrityError('Confirmation fixture already exposed')
    if set(seeds)&used_seeds:raise IntegrityError('Confirmation seed already exposed')
    template=read(parent/'recipes/confirmation-template-acoustic-v1.json')
    if replication_of:
        first=home/'confirmation'/replication_of
        if not read(first/'scorecard.json')['batch_pass'] or read(first/'protocol.json')['candidate']!=candidate:
            raise IntegrityError('Replication requires the unchanged candidate from a passing first batch')
    manifest=deepcopy(base);manifest['families']=families
    manifest['host_energy_by_kind']=dict(game['host_energy_by_kind'])
    manifest.pop('compatible_manifest_sha256',None)
    manifest['style_components']={f['family']:resolve_styles(f['style_multipliers']) for f in families}
    manifest['pilot']['render_cap_seconds']=game['duration_seconds']
    for comps in manifest['style_components'].values():extra_style_hashes(manifest,comps)
    cases=[dict(id=f"{f['family']}-s{s}",family=f['family'],seed=s,physical_gpu=1) for f in families for s in seeds]
    arms=[dict(name='off',checkpoint=None,multiplier=0.),dict(name='original',checkpoint=original['path'],multiplier=1.),
          dict(name='candidate',checkpoint=candidate['path'],multiplier=candidate['multiplier'])]
    source_names=('acoustic_confirmation.py','acoustic_confirmation_support.py','confirmation_amendments.py','acoustic_artifact.py','acoustic_renderer.py','confirmation_stats.py','acoustic_composition.py','acoustic_complete_confirmation.py')
    sources={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in source_names}
    diagnostic_files=[Path(__file__).with_name('acoustic_confirmation_intent.py'),Path(__file__).parent.parent/'reward_sliders/diagnostics.py',
                      Path(__file__).parent.parent/'reward_sliders/evaluate.py']
    diagnostics=dict(sources={str(f.resolve()):sha(f) for f in diagnostic_files},calibration='Off only before original/candidate diagnostics',
        rule='existing preservation_tolerances and preservation_report, every pair valid',comparisons=['off','original'])
    protocol=dict(version='acoustic-v1.1',name=name,evaluation=evaluation,candidate=candidate,original=original,fixtures_sha256=sha(fixtures),
        cases=cases,seeds=seeds,physical_gpu=1,arms=arms,template=template,replication_of=replication_of,
        manifest_sha256=digest(manifest),sources=sources,diagnostics=diagnostics,new_clip_budget=48,research_complete=False)
    folder=home/'confirmation'/name
    if folder.exists():raise IntegrityError('Use an unused batch name; resume existing batches with run')
    immutable(folder/'development-scorecard.json',card);immutable(folder/'manifest.json',manifest)
    immutable(folder/'protocol.json',protocol)
    Store(home).event('confirmation_frozen',batch=name,protocol_sha256=digest(protocol),candidate_sha256=candidate['weights_sha256'],new_clip_budget=48)
    return protocol


def verify_batch(home, folder):
    verify(home);folder=Path(folder);p=read(folder/'protocol.json');m=read(folder/'manifest.json')
    freezes=[json.loads(line) for line in (Path(home)/'ledger.jsonl').read_text().splitlines()
             if line and json.loads(line).get('kind')=='confirmation_frozen' and json.loads(line).get('batch')==p['name']]
    if len(freezes)!=1 or freezes[0]['protocol_sha256']!=digest(p):raise IntegrityError('Fresh protocol differs from its frozen ledger entry')
    if digest(m)!=p['manifest_sha256']:raise IntegrityError('Fresh manifest changed')
    for file,expected in {**effective_sources(home,folder,p),**m['style_hashes']}.items():
        if sha(file)!=expected:raise IntegrityError('Fresh source or style changed: '+file)
    for label in ('candidate','original'):
        c=p[label]
        validator=checkpoint if label=='candidate' else lm_checkpoint
        if validator(c['path'],c['multiplier'])!=c:raise IntegrityError('Frozen checkpoint changed')
    return p,m


def score(home, name):
    import html
    home=Path(home).resolve();folder=home/'confirmation'/name;p,m=verify_batch(home,folder)
    rows=[];audio_rows=[];new_clips=0;render_seconds=0.
    for case in p['cases']:
        row=dict(id=case['id'],family=case['family'],seed=case['seed'],valid=True,audio={})
        for arm in p['arms']:
            file=folder/'observations'/f"{case['id']}-{arm['name']}.json"
            if not file.exists():row['valid']=False;continue
            o=read(file)
            if o['arm']!=arm or o['provenance']['manifest_sha256']!=renderer_digest(m) or o['family']!=case['family'] or o['seed']!=case['seed']:
                raise IntegrityError('Fresh observation identity changed')
            if o.get('audio'):
                if sha(o['audio'])!=o['audio_sha256']:raise IntegrityError('Fresh audio changed')
                row['audio'][arm['name']]=o['audio'];new_clips+=1
                render_seconds+=o.get('timing',{}).get('total_seconds',0.)
            valid=o['status']=='complete' and (o.get('reward') or {}).get('valid',False)
            row['valid'] &= valid
            if valid:row[arm['name']]=o['reward']['scalar']
        rows.append(row)
    card=summarize(rows,p['cases']);card.update(batch=name,protocol_sha256=digest(p),new_clips=new_clips,render_seconds=render_seconds,
        replication_of=p['replication_of'],candidate_sha256=p['candidate']['weights_sha256'])
    write(folder/'scorecard.json',card)
    summary=f"Fresh batch {name}: {card['valid_cases']}/16 valid; practical pass={card['practical_pass']}; uncertainty pass={card['uncertainty_pass']}. Research complete: false.\n"
    for label,c in card['comparisons'].items():summary+=f"\nVs {label}: {c['wins']}/16 wins; equal-family gain {c['equal_family_gain']}; worst {c['worst_delta']}; family interval {c['interval']}.\n"
    (folder/'scorecard.md').write_text(summary)
    for row in sorted(rows,key=lambda r:(r.get('candidate',0)-r.get('off',0),r['id'])):
        players=''.join(f'<td>{html.escape(label)}<audio controls preload="none" src="{html.escape(os.path.relpath(path,folder))}"></audio></td>' for label,path in row['audio'].items())
        values={k:row.get(k) for k in ('off','original','candidate')}
        audio_rows.append(f'<tr><td>{html.escape(row["id"])}<pre>{html.escape(str(values))}</pre></td>{players}</tr>')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fresh confirmation listening</title><style>body{font:16px sans-serif;margin:2em}td{padding:1em;border-bottom:1px solid #aaa}audio{display:block;width:230px}</style><h1>Fresh confirmation</h1><pre>'+html.escape(summary)+'</pre><p>Raw audio, ordered with losses first. Completion of a batch does not establish final success.</p><table>'+''.join(audio_rows)+'</table>')
    return card


def run(home,name):
    import torch
    from .resources import gpu_lease
    from .acoustic_renderer import AcousticRenderer
    from ..reward_sliders.reward import CEReward
    from ..reward_sliders.specs import RewardSpec
    home=Path(home).resolve();folder=home/'confirmation'/name;p,m=verify_batch(home,folder)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise IntegrityError('Fresh rendering owns physical GPU 1')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    renderer=None;began=time.monotonic();store=Store(home)
    with locked(folder/'batch.lock'),gpu_lease(home,1):
        store.status('fresh_confirmation_running',batch=name,research_complete=False,protocol_sha256=digest(p))
        scorer=CEReward(RewardSpec(**m['reward_spec']))
        try:
            for case in p['cases']:
                family=next(f for f in m['families'] if f['family']==case['family'])
                for arm in p['arms']:
                    file=folder/'observations'/f"{case['id']}-{arm['name']}.json"
                    if file.exists():
                        old=read(file)
                        if old['status']=='running':
                            old.update(status='failed',error='Interrupted output retained without reroll')
                            audio=folder/'audio'/f"{case['id']}-{arm['name']}.wav"
                            if audio.exists():old.update(audio=str(audio),audio_sha256=sha(audio))
                            write(file,old)
                        continue
                    if renderer is None:renderer=AcousticRenderer(folder,m,1)
                    extra=component(arm['checkpoint'],arm['multiplier']) if arm['checkpoint'] else None
                    started=time.time()
                    try:
                        row=renderer.observe(family,case['seed'],arm,scorer,extra=extra)
                        row.update(started_unix=started,finished_unix=time.time());write(file,row)
                    except BaseException as exc:
                        if file.exists():
                            row=read(file);audio=folder/'audio'/f"{case['id']}-{arm['name']}.wav"
                            row.update(status='interrupted',error=repr(exc),started_unix=started,finished_unix=time.time())
                            if audio.exists():row.update(audio=str(audio),audio_sha256=sha(audio))
                            write(file,row)
                        raise
                    store.event('confirmation_case_finished',batch=name,case=case['id'],arm=arm['name'],status=row['status'],audio=row.get('audio'))
                score(home,name)
        finally:
            if renderer is not None:
                renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
                exact=all(torch.equal(mod.weight.detach().cpu(),saved) for mod,saved in renderer.host._merge_state(renderer.device).pristine.items())
                write(folder/'audit/off-restoration.json',dict(exact=exact,physical_gpu=1,elapsed_seconds=time.monotonic()-began))
                if not exact:raise IntegrityError('Fresh renderer failed exact Off restoration')
        result=score(home,name)
        store.event('confirmation_batch_finished',batch=name,batch_pass=result['batch_pass'],new_clips=result['new_clips'])
        store.status('confirmation_review_required',batch=name,research_complete=False)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--name',required=True)
    p.add_argument('action',choices=('freeze','run','score'));p.add_argument('--evaluation');p.add_argument('--fixtures');p.add_argument('--replication-of')
    a=p.parse_args()
    result=freeze(a.home,a.evaluation,a.fixtures,a.name,a.replication_of) if a.action=='freeze' else globals()[a.action](a.home,a.name)
    print(__import__('json').dumps(result,indent=2),flush=True)
