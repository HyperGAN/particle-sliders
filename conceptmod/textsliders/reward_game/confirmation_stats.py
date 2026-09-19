"""Whole-family uncertainty and frozen practical gates for fresh evidence."""
from collections import defaultdict
import math


def family_interval(family_means, *, replicates=50000, seed=36931):
    import numpy as np
    values=np.array([family_means[k] for k in sorted(family_means)],dtype=np.float64)
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('Need finite independent family means')
    generator=np.random.default_rng(seed)
    sampled=values[generator.integers(0,len(values),size=(replicates,len(values)))].mean(1)
    low,high=np.quantile(sampled,[.025,.975])
    return dict(low=float(low),high=float(high),level=.95,unit='whole family',replicates=replicates,seed=seed)


def summarize(rows, expected_cases):
    """Rows hold three valid CE values per case, never individual audio windows."""
    expected={r['id']:r for r in expected_cases}
    if len(expected)!=16 or len(expected_cases)!=16:raise ValueError('Fresh batch requires sixteen unique cases')
    families=defaultdict(list)
    for case in expected_cases:families[case['family']].append(case['seed'])
    if len(families)!=8 or any(len(v)!=2 or len(set(v))!=2 for v in families.values()):
        raise ValueError('Fresh batch requires eight families with two distinct seeds each')
    invalid=[];valid=[];seen=set()
    for row in rows:
        ident=row['id']
        if ident not in expected or ident in seen:raise ValueError('Unexpected or duplicate confirmation case')
        seen.add(ident);case=expected[ident]
        if row['family']!=case['family'] or row['seed']!=case['seed']:raise ValueError('Confirmation case identity changed')
        if not row.get('valid') or any(not isinstance(row.get(k),(float,int)) or not math.isfinite(row[k]) for k in ('candidate','off','original')):
            invalid.append(ident);continue
        valid.append(dict(row,deltas={label:row['candidate']-row[label] for label in ('off','original')}))
    invalid.extend(sorted(set(expected)-seen));comparisons={}
    for label in ('off','original'):
        groups=defaultdict(list)
        for row in valid:groups[row['family']].append(row['deltas'][label])
        means={key:sum(values)/len(values) for key,values in groups.items()}
        deltas=[r['deltas'][label] for r in valid]
        comparisons[label]=dict(wins=sum(v>0 for v in deltas),meaningful_wins=sum(v>=.02 for v in deltas),
            ties=sum(v==0 for v in deltas),equal_family_gain=sum(means.values())/len(means) if means else None,
            worst_delta=min(deltas) if deltas else None,family_deltas=means,
            interval=family_interval(means) if not invalid else None)
    off= comparisons['off'];original=comparisons['original']
    practical=bool(not invalid and off['wins']>=13 and off['equal_family_gain']>=.10 and off['worst_delta']>=-.50 and
                   original['wins']>=10 and original['equal_family_gain']>=.05)
    uncertainty=bool(not invalid and all(c['interval']['low']>0 for c in comparisons.values()))
    return dict(scheduled=16,valid_cases=len(valid),invalid=invalid,comparisons=comparisons,rows=valid,
        practical_pass=practical,uncertainty_pass=uncertainty,batch_pass=practical and uncertainty,
        research_complete=False,next_action='Independent replication and preservation/composition review remain required')
