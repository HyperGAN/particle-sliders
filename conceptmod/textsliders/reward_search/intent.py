"""Observe final audio for voice, lyric, style and diversity changes, separately from CE."""
from copy import deepcopy
from pathlib import Path
import time

from ..reward_sliders.specs import sha,digest,read_json,write_json
from ..reward_sliders.diagnostics import IntentDiagnostics
from ..reward_sliders.evaluate import preservation_report
from .setup import DEFAULT_RUN,SOURCE
from .report import report


def run_diagnostics(run):
    run=Path(run);m=read_json(run/'manifest.json');folder=run/'intent';folder.mkdir(exist_ok=True)
    tolerances=read_json(SOURCE/'transfer/manifest.json')['transfer']['tolerances']
    protocol=dict(scope='post-selection diagnostics on all final 20-second comparison clips',
                  selection_effect='none; CE alone selects the checkpoint and strength',
                  tolerance_reference=tolerances,source_manifest_sha256=sha(run/'manifest.json'),
                  implementation_sha256=sha(__file__))
    write_json(folder/'protocol.json',protocol)
    model=IntentDiagnostics(folder/'cache')
    while not (run/'final-candidate.json').exists():
        if (run/'experiment-done').exists():
            write_json(folder/'status.json',dict(stage='unavailable',reason='Campaign ended before final selection'));return
        time.sleep(15)
    selected=read_json(run/'final-candidate.json')['comparison_name']
    names={'off','v1-original',selected}
    families={f['family']:f for f in m['families'] if f.get('group')=='final'}
    expected={f'{family}-s{seed}-{name}' for family in families for seed in m['search']['final_seeds'] for name in names}
    done={}
    while len(done)<len(expected):
        for ident in sorted(expected-set(done)):
            path=run/'stages/final/observations'/f'{ident}.json'
            if not path.exists():continue
            original=read_json(path)
            if original['status'] not in ('complete','failed'):continue
            row=dict(original,voice=families[original['family']]['voice'])
            if row['status']=='complete':
                assert sha(row['audio'])==row['audio_sha256']
                row['intent']=model.measure(row['audio'],families[row['family']])
            write_json(folder/'observations'/f'{ident}.json',row);done[ident]=row
            write_json(folder/'status.json',dict(stage='measuring',completed=len(done),total=len(expected),updated_unix=time.time()))
            print(f'INTENT {len(done)}/{len(expected)} {ident}',flush=True)
        if len(done)==len(expected):break
        if (run/'experiment-done').exists() and read_json(run/'status.json')['stage']!='complete':
            write_json(folder/'status.json',dict(stage='incomplete',completed=len(done),total=len(expected),reason='Campaign ended early'));return
        time.sleep(10)
    comparisons={}
    for label,treatment,control in [('v1_vs_off','v1-original','off'),('candidate_vs_off',selected,'off'),
                                    ('candidate_vs_v1',selected,'v1-original')]:
        rows=[]
        for family in families:
            for seed in m['search']['final_seeds']:
                for name,role in [(control,'off'),(treatment,'lora')]:
                    row=deepcopy(done[f'{family}-s{seed}-{name}']);row['arm']['name']=role;rows.append(row)
        comparisons[label]=preservation_report(rows,tolerances)
    write_json(folder/'results.json',dict(comparisons=comparisons,
        valid_diagnostics=sum(r.get('intent',{}).get('valid',False) for r in done.values()),
        total=len(done),candidate=selected,protocol_sha256=digest(protocol),
        limitation='Short-clip diagnostics with reused v1 variability tolerances; no full-song preservation claim'))
    write_json(folder/'status.json',dict(stage='complete',completed=len(done),total=len(expected),updated_unix=time.time()))
    report(run)
    print('Final intent diagnostics completed; CE selection unchanged.',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    try:run_diagnostics(args.run_dir)
    except Exception as error:
        write_json(args.run_dir/'intent/status.json',dict(stage='error',error=repr(error),updated_unix=time.time()))
        raise
