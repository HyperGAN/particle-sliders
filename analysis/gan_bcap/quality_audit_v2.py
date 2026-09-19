"""Final audio audit refinements, independent of frozen training source files.

Adds partial near-duplicate occupancy loss and requires candidate-level endpoint
evidence before a calibrated judge can pass a quality gate. The preference score
and rendered fixtures are unchanged. These are test-driven audit refinements,
not coefficient fits to the new candidates' scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders.gan_v2 import metrics as original
from conceptmod.textsliders.gan_v2.data import sha


def duplicate_groups(values,threshold):
    values=original._embeddings(values)
    distance=np.maximum(0.,1-values@values.T)
    parent=list(range(len(values)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(values)):
        for j in range(i):
            if distance[i,j]<=threshold:parent[root(i)]=root(j)
    counts={}
    for i in range(len(values)):
        key=root(i);counts[key]=counts.get(key,0)+1
    probabilities=np.asarray(list(counts.values()))/len(values)
    return dict(count=len(counts),sizes=sorted(counts.values(),reverse=True),
        effective_count=float(np.exp(-(probabilities*np.log(probabilities)).sum())),
        interpretation='Connected near-duplicate groups in this representation; not known musical modes.')


def compare_diversity(candidate,neutral,positive,**kwargs):
    result=original.compare_diversity(candidate,neutral,positive,**kwargs)
    if result['status']=='insufficient_evidence':return result
    threshold=result['candidate']['duplicate_distance']
    groups={key:duplicate_groups(values,threshold) for key,values in
            [('candidate',candidate),('neutral',neutral),('positive',positive)]}
    reference=min(groups['neutral']['effective_count'],groups['positive']['effective_count'])
    ratio=groups['candidate']['effective_count']/reference
    result.update(near_duplicate_groups=groups,effective_group_ratio=ratio)
    if ratio<=.5 and result['near_duplicate_excess']>=.25:
        result.update(status='collapse_suspected',reason='Excess repeated samples and reduced near-duplicate group occupancy')
    return result


def quality_decision(clips,diversity,*,judge_calibration=None,**kwargs):
    result=original.quality_decision(clips,diversity,judge_calibration=None,**kwargs)
    if result['decision'] in ('reject_candidate','evaluate_more'):return result
    if any(r['diagnostics']['lyric_status']=='unmeasurable' for r in clips):return result
    calibration=judge_calibration or {}
    required=('artifact_sha256','description_threshold','minimum_endpoint_rate')
    if (not calibration.get('validated_on_reserved_labels') or any(k not in calibration for k in required)
            or not re.fullmatch('[0-9a-f]{64}',str(calibration.get('artifact_sha256','')))):
        return dict(result,decision='quality_unvalidated',musical_quality_validated=False,
                    reason='Need independently validated judge thresholds and candidate endpoint evidence')
    threshold=float(calibration['description_threshold']);minimum=float(calibration['minimum_endpoint_rate'])
    if not np.isfinite(threshold) or not 0<minimum<=1:raise ValueError('Invalid calibrated endpoint thresholds')
    rate=sum(r['diagnostics']['absolute_description_margin']>=threshold for r in clips)/len(clips)
    if rate<minimum:
        return dict(result,decision='reject_candidate',musical_quality_validated=False,
                    calibrated_endpoint_rate=rate,reason='Candidate fails the calibrated condition endpoint on the declared fixtures')
    return dict(result,decision='pass_declared_quality_gates',musical_quality_validated=True,
        calibrated_endpoint_rate=rate,calibration=calibration,
        interpretation='Passed declared checks on these fixtures; not proof of universal musical quality.')


def evaluate(scores,output,cache):
    from analysis.gan_bcap import audio_v2
    previous=audio_v2.compare_diversity,audio_v2.quality_decision
    audio_v2.compare_diversity=compare_diversity;audio_v2.quality_decision=quality_decision
    try:
        result=audio_v2.evaluate(scores,output,cache)
    finally:
        audio_v2.compare_diversity,audio_v2.quality_decision=previous
    result['audit_revision']=dict(version=2,source_sha256=sha(__file__),
        base_metrics_sha256=sha(original.__file__),base_audio_scorer_sha256=sha(audio_v2.__file__),
        changes=['Partial near-duplicate occupancy loss','Candidate-specific calibrated endpoint gate'],
        unchanged=['Preference score','Audio fixtures','Candidate weights'],judge_calibration=None)
    temporary=Path(output).with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(output)
    return result


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'analysis/gan_bcap/v2_20260905/audio_embeddings')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    evaluate(args.scores,args.output,args.cache)
