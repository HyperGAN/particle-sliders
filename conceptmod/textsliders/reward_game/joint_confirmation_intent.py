"""Independent preservation checks, with baseline-only tolerance calibration."""
import argparse
from copy import deepcopy
from pathlib import Path
from .core import read,write,immutable,sha,digest,locked,IntegrityError,Store
from .joint_confirmation import verify_batch,score


def run(home,name):
    from ..reward_sliders.diagnostics import IntentDiagnostics,preservation_tolerances
    from ..reward_sliders.evaluate import preservation_report
    home=Path(home).resolve();folder=home/'confirmation'/name;p,m=verify_batch(home,folder)
    if not score(home,name)['batch_pass']:raise IntegrityError('Complete and pass fresh CE evidence before expensive preservation diagnostics')
    declared=p['diagnostics']
    for path,expected in declared['sources'].items():
        if sha(path)!=expected:raise IntegrityError('Frozen independent diagnostics changed')
    diagnostics=folder/'intent';measurer=IntentDiagnostics(diagnostics/'cache');families={f['family']:f for f in m['families']}
    with locked(diagnostics/'diagnostics.lock'):
        records=[]
        # Candidate diagnostics are not computed until all Off observations have
        # fixed the tolerance values under the previously declared rule.
        for arm in ('off','original','candidate'):
            if arm!='off' and not (diagnostics/'tolerances.json').exists():raise IntegrityError('Baseline calibration missing')
            for case in p['cases']:
                ident=f"{case['id']}-{arm}";source=folder/'observations'/f'{ident}.json';raw=read(source)
                if raw['status']!='complete' or sha(raw['audio'])!=raw['audio_sha256']:raise IntegrityError('Invalid diagnostic source')
                path=diagnostics/'observations'/f'{ident}.json'
                if path.exists():
                    row=read(path)
                    if row['source_observation_sha256']!=sha(source):raise IntegrityError('Diagnostic observation changed')
                else:
                    row=dict(raw,voice=families[case['family']]['voice'],source_observation_sha256=sha(source),
                             intent=measurer.measure(raw['audio'],families[case['family']]))
                    write(path,row)
                records.append(row)
                print('INTENT',ident,row['intent'].get('valid'),flush=True)
                if not row['intent'].get('valid'):raise IntegrityError('Missing preservation evidence: '+str(row['intent'].get('error')))
            if arm=='off':
                tolerance=preservation_tolerances(records)
                if any(v is None for v in tolerance['values'].values()):raise IntegrityError('Incomplete baseline tolerance calibration')
                immutable(diagnostics/'tolerances.json',dict(tolerance,source_audio_hashes=[r['audio_sha256'] for r in records]))
        tolerances=read(diagnostics/'tolerances.json');comparisons={}
        for control in ('off','original'):
            paired=[]
            for row in records:
                if row['arm']['name'] not in (control,'candidate'):continue
                changed=deepcopy(row);changed['arm']['name']='lora' if row['arm']['name']=='candidate' else 'off';paired.append(changed)
            comparisons[control]=preservation_report(paired,tolerances)
        result=dict(comparisons=comparisons,passed=all(c['passed'] for c in comparisons.values()),
            tolerance_sha256=sha(diagnostics/'tolerances.json'),protocol_sha256=digest(declared),
            observations_sha256={str(f):sha(f) for f in (diagnostics/'observations').glob('*.json')},
            new_audio_generated=0,research_complete=False,
            natural_completion='20.4-second capped excerpts; raw observed durations retained, natural full-song completion not established')
        write(diagnostics/'preservation.json',result)
        Store(home).event('confirmation_preservation_finished',batch=name,passed=result['passed'],result_sha256=sha(diagnostics/'preservation.json'))
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--name',required=True)
    a=p.parse_args();print(__import__('json').dumps(run(a.home,a.name),indent=2))
