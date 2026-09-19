"""Retrospective agreement of existing metrics with one listener's rankings.

No fitting, new audio, GPU work, score-direction search, or compound objective.
Pairwise preferences within a ranking are dependent observations.
"""
from itertools import combinations
from pathlib import Path
import json
import math
from common import WORK, OLD, sha, write

METRICS = {
    'CE': 'Content Enjoyment', 'PQ': 'Production Quality',
    'PC': 'Production Complexity (exploratory: higher)',
    'CU': 'Content Usefulness',
    'phrase_accuracy': 'ASR phrase match', 'recall': 'ASR lyric recall',
    'precision': 'ASR lyric precision',
    'concept_a': 'First concept CLAP margin',
    'concept_b': 'Second concept CLAP margin',
    'concept_mean': 'Mean of two CLAP margins (exploratory)',
    'concept_min': 'Minimum of two CLAP margins (exploratory)',
    'cosine_to_linear': 'Update cosine to ordinary (static method order)',
}


def groups():
    feedback = json.loads((WORK/'listening-feedback.json').read_text())
    assert feedback['screen_sha256'] == sha(WORK/'screen.json')
    assert feedback['blind_key_sha256'] == sha(WORK/'blind-key.json')
    measured = json.loads((WORK/'measurements.json').read_text())
    rendered = json.loads((WORK/'renders.json').read_text())
    geometry = {(g['pair'], g['method']):g['cosine_to_linear']
                for g in json.loads((WORK/'geometry-summary.json').read_text())}
    old_feedback = json.loads((OLD/'listening-feedback.json').read_text())['entries'][-1]
    assert old_feedback['screen_sha256'] == sha(OLD/'screen.json')
    old_measured = json.loads((OLD/'measurements.json').read_text())
    old_rendered = json.loads((OLD/'renders.json').read_text())
    old_spec = json.loads((OLD/'screen.json').read_text())
    result = []

    def add(dataset, fixture, pair, entries, measurements, renders, uncertain=False, close_edge=False):
        ranked = []
        for entry in entries:
            item = measurements['records'][entry['id']]
            rec = renders['records'][entry['id']]
            assert item['status'] == rec['status'] == 'complete'
            assert item['excerpt_sha256'] == rec['excerpt_sha256'] == entry['excerpt_sha256']
            a,b = (item['concepts'][p] for p in pair)
            values = {**item['aesthetics'], **item['lyric_diagnostics'],
                      'concept_a':a, 'concept_b':b, 'concept_mean':(a+b)/2, 'concept_min':min(a,b)}
            if dataset == 'mergers':
                values['cosine_to_linear'] = (1. if entry['method'] == 'linear' else
                                               geometry[('+'.join(pair),entry['method'])])
            assert all(math.isfinite(v) for v in values.values())
            ranked.append(dict(id=entry['id'],method=entry['method'],values=values))
        result.append(dict(dataset=dataset,fixture=fixture,pair=pair,ranked=ranked,
                           uncertain_ranking=uncertain,close_top_edge=close_edge))

    for g in feedback['records']:
        add('mergers',g['fixture'],g['pair'],g['ranking_best_to_worst'],measured,rendered,
            uncertain=g['fixture']=='confirmation' and g['pair']==['female','pop'],
            close_edge=g['fixture']=='confirmation' and g['pair']==['country','indie-rock'])
    for g in old_feedback['rankings']:
        entries=[]
        for r in g['best_to_worst']:
            job=next(j for j in old_spec['jobs'] if j['id']==r['job_id'])
            assert job['energy']['language_model']==r['energy']
            entries.append(dict(id=r['job_id'],method=f"ordinary_E{r['energy']:g}",excerpt_sha256=r['excerpt_sha256']))
        add('earlier_energy','familiar',g['sliders'],entries,old_measured,old_rendered,
            close_edge=g['top_two_close'] is True)
    return result


def score(selected, metric, omit_close_edges=False):
    details=[]
    for g in selected:
        if any(metric not in r['values'] for r in g['ranked']):continue
        vals=[r['values'][metric] for r in g['ranked']]
        pairs=[]
        for a,b in combinations(range(3),2):
            if omit_close_edges and g['close_top_edge'] and (a,b)==(0,1):continue
            credit=1. if vals[a]>vals[b] else .5 if vals[a]==vals[b] else 0.
            pairs.append(dict(preferred=g['ranked'][a]['id'],other=g['ranked'][b]['id'],credit=credit))
        best=max(vals); tied=sum(v==best for v in vals)
        details.append(dict(fixture=g['fixture'],pair=g['pair'],metric_values_in_human_order=vals,
                            agreements=sum(p['credit'] for p in pairs),comparisons=len(pairs),
                            top1_credit=float(vals[0]==best)/tied,pairs=pairs))
    return dict(agreements=sum(d['agreements'] for d in details),
                comparisons=sum(d['comparisons'] for d in details),
                top1_credit=sum(d['top1_credit'] for d in details),groups=len(details),details=details)


def main():
    all_groups=groups()
    conditions={
        'mergers_all': ([g for g in all_groups if g['dataset']=='mergers'],False),
        'mergers_without_uncertain': ([g for g in all_groups if g['dataset']=='mergers' and not g['uncertain_ranking']],False),
        'mergers_without_uncertain_or_close_edge': ([g for g in all_groups if g['dataset']=='mergers' and not g['uncertain_ranking']],True),
        'earlier_energy': ([g for g in all_groups if g['dataset']=='earlier_energy'],False),
    }
    scores={name:{metric:score(gs,metric,omit) for metric in METRICS} for name,(gs,omit) in conditions.items()}
    result=dict(schema=1,status='complete',interpretation='Retrospective descriptive agreement, not predictive validation or causal explanation.',
        source_sha256={str(p):sha(p) for p in [WORK/'listening-feedback.json',WORK/'measurements.json',
            WORK/'renders.json',WORK/'screen.json',WORK/'blind-key.json',WORK/'geometry-summary.json',
            OLD/'listening-feedback.json',OLD/'measurements.json',OLD/'screen.json',OLD/'renders.json',
            OLD/'geometry.json',Path(__file__)]},
        metrics=METRICS,score_direction='Higher for every listed metric; directions are not optimized against these labels.',
        pairwise_ties='Half credit; tied maximum scores split top-choice credit.',scores=scores,
        limitations=['Six new ranking groups from two songs and one listener; their 18 pairwise comparisons are dependent.',
            'No fitted weights or combined scoring rule. All inspected candidate metrics are reported; choosing the highest observed agreement is retrospective selection.',
            'The old energy study shares the first song and three E2.8 audio files, so it is a setting-transfer check, not an independent holdout.',
            'Mean/minimum CLAP margins combine differently scaled concepts and are exploratory, not calibrated joint-style scores.',
            'Measurements cover first 20 seconds; the user may also have judged full recordings.',
            'Static update geometry cannot by itself predict preferences that reverse between songs with identical merged weights.'])
    write(WORK/'metric-agreement.json',result)
    lines=['# Do the saved metrics track the listener?','',
        'Partly, but none is validated as a reliable selection rule or causal explanation. This audit uses saved measurements only.','',
        '| Metric | Merger pairwise agreement | Preferred clip picked | Earlier energy pairwise agreement |',
        '|---|---:|---:|---:|']
    for metric,label in METRICS.items():
        a=scores['mergers_all'][metric];b=scores['earlier_energy'][metric]
        old=f"{b['agreements']:g}/{b['comparisons']}" if b['comparisons'] else 'Not applicable'
        lines.append(f"| {label} | {a['agreements']:g}/{a['comparisons']} | {a['top1_credit']:g}/{a['groups']} | {old} |")
    lines+=['', 'Tied predictions receive half pairwise credit and split top-choice credit. A ranking of three clips creates three comparisons; six rankings are not eighteen independent listening tests.','',
        'Content Enjoyment (CE) gives the largest aggregate agreement among these saved single metrics: 13/18, but selects the listener’s favorite in only 3/6 groups. A fixed ordinary > KnOTS-TIES > TIES order gives 12/18 and also selects 3/6 favorites. Update cosine produces exactly that fixed order here; it contributes no song-dependent selection.','',
        'Excluding the uncertain female + pop confirmation group gives CE 11/15 and the fixed method order 12/15. Excluding the close country confirmation top comparison as well gives CE 10/14 and the fixed method order 11/14. CE gets only 4/9 comparisons and 1/3 favorites right in the earlier energy study. Production Complexity gets 8/9 in that energy study and 12/18 here, making it another candidate to evaluate prospectively. Complexity is not inherently better, and these two studies share clips. No fitted blend of metrics is justified by these few labels.','',
        '## What matches, and what fails','',
        '- Female + pop: CE and PQ both put KnOTS-TIES above ordinary on both songs, agreeing with the listener on that comparison. Neither chooses TIES as the confirmation favorite, where the listener was uncertain.',
        '- Country + indie rock: both concept margins recover the entire first-song ranking. On confirmation, they put ordinary first but incorrectly put KnOTS-TIES above TIES. CE recovers the full confirmation ranking and its small top-score gap, but puts ordinary last on the first song.',
        '- House + acoustic folk: CE and PQ recover the first-song order. On confirmation, both make errors below or above ordinary; neither recovers the full ranking.','',
        '## Weight geometry','',
        'Source-update cosines are approximately 0.0127 for female + pop, 0.0612 for country + indie rock, and 0.0434 for house + acoustic folk. That is a possible association to investigate, not a validated explanation from only three pairs. These values are fixed across songs, while the house preference reverses. All merger conditions also match each projection’s update norm, so Frobenius energy cannot distinguish them. Raw weight cosine is not the same as activation alignment.','',
        '## Implication','',
        'Use CE/PQ, complexity and each requested concept margin as separate diagnostics. A predictive rule needs new preferences on songs and seeds not used to select or tune it, and must beat a simple fixed-method baseline. A mechanistic explanation would additionally need measurements of the model’s behavior on the actual song/seed context; this audit does not establish one.','',
        'Definitions: [Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) separates enjoyment from technical production quality. [KnOTS](https://arxiv.org/abs/2410.19735) studies alignment for LoRA merging; its motivation is not proof that raw weight cosine predicts these listening choices.','',
        'The earlier energy check shares the first-song fixture and three recordings with this study. It is useful evidence of a failure to transfer across tested settings, not an independent test set. All metrics were measured on the first 20 seconds.']
    (WORK/'metric-agreement.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:len(METRICS)+6]))


if __name__=='__main__':
    main()
