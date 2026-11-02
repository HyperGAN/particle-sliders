"""Separate quality, condition and diversity evidence; never fill missing with zero."""
from __future__ import annotations

import math
import numpy as np


def _embeddings(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Need finite audio embeddings for distinct seeds')
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if (norms<1e-12).any():raise ValueError('Zero audio embedding')
    return values/norms


def diversity_summary(values,*,duplicate_distance=.002):
    if not 0<=duplicate_distance<2:raise ValueError('Invalid cosine-distance tolerance')
    values=_embeddings(values)
    distances=np.maximum(0.,1-values@values.T)
    pairwise=distances[np.triu_indices(len(values),1)]
    medoid=int(np.argmin(distances.sum(1)))
    radii=distances[medoid]
    return dict(seeds=len(values),median_pair_distance=float(np.median(pairwise)),
        central_radius=float(np.median(radii)),tail_radius_90=float(np.quantile(radii,.9)),
        near_duplicate_fraction=float(np.mean(pairwise<=duplicate_distance)),
        duplicate_distance=duplicate_distance,
        interpretation='Within-prompt frozen audio representation diversity; not known musical mode count.')


def compare_diversity(candidate,neutral,positive,*,minimum_seeds=4,minimum_ratio=.35):
    if len(candidate)<2 and len(candidate)==len(neutral)==len(positive):
        return dict(status='insufficient_evidence',reason='Need at least two seeds to measure diversity',
                    central_radius_ratio=None,seeds=len(candidate))
    c,n,p=map(_embeddings,(candidate,neutral,positive))
    if c.shape!=n.shape or c.shape!=p.shape:raise ValueError('Diversity controls must use the same seeds and representation')
    if minimum_seeds<4 or not 0<minimum_ratio<1:raise ValueError('Invalid diversity screen')
    n0,p0=diversity_summary(n),diversity_summary(p)
    # A tight positive condition can legitimately reduce voice variation.
    reference=min(n0['central_radius'],p0['central_radius'])
    duplicate=max(1e-4,.05*min(n0['median_pair_distance'],p0['median_pair_distance']))
    summaries=[diversity_summary(v,duplicate_distance=duplicate) for v in (c,n,p)]
    result=dict(candidate=summaries[0],neutral=summaries[1],positive=summaries[2],
        central_radius_ratio=None,status='insufficient_evidence',
        calibration='Declared gross-collapse screen, not a calibrated perceptual diversity threshold')
    if len(c)<minimum_seeds:
        result['reason']='Too few matched seeds';return result
    if reference<1e-5:
        result['reason']='Reference diversity is itself too small';return result
    ratio=summaries[0]['central_radius']/reference
    duplicate_excess=summaries[0]['near_duplicate_fraction']-max(summaries[1]['near_duplicate_fraction'],summaries[2]['near_duplicate_fraction'])
    result.update(central_radius_ratio=ratio,near_duplicate_excess=duplicate_excess)
    if ratio<minimum_ratio and duplicate_excess>=.25:
        result.update(status='collapse_suspected',reason='Central variation compressed with excess near-duplicates')
    else:
        result.update(status='no_collapse_detected',reason='No gross collapse under this representation and seed budget')
    return result


def clip_diagnostics(candidate,neutral,positive):
    required=('rms','duration','lyrics','concept','hf14k_fraction','clipped_fraction')
    for row in (candidate,neutral,positive):
        if any(key not in row or not math.isfinite(row[key]) for key in required):
            raise ValueError('Missing or nonfinite quality component')
    failures=[]
    if candidate['rms']<.02*max(neutral['rms'],1e-12):failures.append('near_silence')
    if candidate['duration']<.5*min(neutral['duration'],positive['duration']):failures.append('short_output')
    if candidate['clipped_fraction']>max(neutral['clipped_fraction'],positive['clipped_fraction'])+.01:
        failures.append('excess_clipping')
    if candidate['hf14k_fraction']>max(neutral['hf14k_fraction'],positive['hf14k_fraction'])+.02:
        failures.append('excess_high_frequency_energy')
    lyric_reference=min(neutral['lyrics'],positive['lyrics'])
    lyric_status='unmeasurable' if lyric_reference<.2 else 'measurable_proxy'
    if lyric_reference>=.2 and candidate['lyrics']<lyric_reference-.15:failures.append('lyric_proxy_regression')
    return dict(failures=failures,technical_screen_passed=not failures,
        lyric_status=lyric_status,absolute_description_margin=candidate['concept'],
        relative_description_gain=candidate['concept']-neutral['concept'],
        positive_reference_margin=positive['concept'],
        reaches_positive_description_reference=candidate['concept']>=positive['concept']-.02,
        condition_interpretation='Description similarity, not a calibrated voice classifier')


def quality_decision(clips,diversity,*,minimum_prompts=8,maximum_failure_rate=.1,
                     judge_calibration=None):
    """Fail closed on missing coverage/calibration; ranking cannot waive failures."""
    if not clips or not diversity:return dict(decision='evaluate_more',reason='Missing clips or diversity controls')
    groups={row['prompt'] for row in clips}
    failures=sum(bool(row['diagnostics']['failures']) for row in clips)
    failure_rate=failures/len(clips)
    per_prompt={group:sum(bool(r['diagnostics']['failures']) for r in clips if r['prompt']==group)/
                sum(r['prompt']==group for r in clips) for group in sorted(groups)}
    result=dict(failure_rate=failure_rate,per_prompt_failure_rate=per_prompt,
        description_reference_reach_rate=sum(r['diagnostics']['reaches_positive_description_reference'] for r in clips)/len(clips),
        absolute_positive_description_rate=sum(r['diagnostics']['absolute_description_margin']>0 for r in clips)/len(clips),
        musical_quality_validated=False)
    severe=any('near_silence' in row['diagnostics']['failures'] for row in clips)
    if severe or any(r['status']=='collapse_suspected' for r in diversity.values()):
        return dict(result,decision='reject_candidate',reason='Near-silence or consensus suspected diversity collapse')
    if len(groups)<minimum_prompts or set(diversity)!=groups or any(r['status']=='insufficient_evidence' for r in diversity.values()):
        return dict(result,decision='evaluate_more',reason='Insufficient matched prompt/seed coverage')
    if failure_rate>maximum_failure_rate:
        return dict(result,decision='reject_candidate',reason='Declared technical-regression screen exceeded; not a musical-quality verdict')
    if any(r['diagnostics']['lyric_status']=='unmeasurable' for r in clips):
        return dict(result,decision='quality_unvalidated',reason='Lyrics could not be assessed reliably against references')
    if not judge_calibration or not judge_calibration.get('validated_on_reserved_labels'):
        return dict(result,decision='quality_unvalidated',reason='Technical screens passed; preference/condition judge is not calibrated')
    return dict(result,decision='pass_declared_quality_gates',musical_quality_validated=True,
                calibration=judge_calibration)
