"""Select tested slider settings by mean Content Enjoyment across shared seeds.

This decision rule was adopted after the user clarified that the sweep should
make metric-guided choices. It is not a preregistered or held-out validation.
"""
import html
import json
import statistics
from pathlib import Path
from common import WORK, OUTPUT, PUBLIC, PAIRS, sha, write, verify


def main():
    spec = json.loads((WORK / 'screen.json').read_text())
    renders = json.loads((WORK / 'renders.json').read_text())
    measured = json.loads((WORK / 'measurements.json').read_text())
    audit = json.loads((WORK / 'audit.json').read_text())
    assert renders['status'] == measured['status'] == 'complete'
    assert audit['status'] == 'passed'
    assert renders['screen_sha256'] == measured['screen_sha256'] == audit['screen_sha256'] == sha(WORK / 'screen.json')
    assert {j['id'] for j in spec['jobs']} == set(renders['records']) == set(measured['records'])
    verify(spec)
    groups = []
    presets = []
    for pair in PAIRS:
        candidates = []
        centers = {j['seed']: j for j in spec['jobs'] if j['pair'] == list(pair) and j['condition'] == 'center'}
        for condition in spec['conditions']:
            jobs = sorted([j for j in spec['jobs'] if j['pair'] == list(pair)
                           and j['condition'] == condition['key']], key=lambda j: j['seed'])
            assert [j['seed'] for j in jobs] == sorted(spec['seeds'])
            samples = []
            for job in jobs:
                rec = renders['records'][job['id']]
                m = measured['records'][job['id']]
                base = measured['records'][centers[job['seed']]['id']]
                assert rec['status'] == m['status'] == 'complete'
                assert sha(rec['excerpt']) == m['excerpt_sha256'] == rec['excerpt_sha256']
                assert sha(rec['audio']) == rec['audio_sha256']
                samples.append(dict(seed=job['seed'], job_id=job['id'], baseline_id=centers[job['seed']]['id'],
                    CE=m['aesthetics']['CE'], PQ=m['aesthetics']['PQ'],
                    CE_delta=m['aesthetics']['CE']-base['aesthetics']['CE'],
                    PQ_delta=m['aesthetics']['PQ']-base['aesthetics']['PQ'],
                    concepts={name: m['concepts'][name] for name in pair},
                    concept_deltas={name: m['concepts'][name]-base['concepts'][name] for name in pair},
                    lyric_phrase=m['lyric_diagnostics']['phrase_accuracy']))
            candidates.append(dict(condition=condition['key'], energy=jobs[0]['energy'],
                sliders=jobs[0]['sliders'], share=condition['share'], samples=samples,
                mean_CE=statistics.mean(x['CE'] for x in samples),
                mean_PQ=statistics.mean(x['PQ'] for x in samples),
                mean_CE_delta=statistics.mean(x['CE_delta'] for x in samples),
                mean_PQ_delta=statistics.mean(x['PQ_delta'] for x in samples)))
        # No fitted blend, pruning threshold, listening gate, or implicit style veto.
        # PQ breaks exact CE ties only. Condition id makes any remaining tie stable.
        candidates.sort(key=lambda c: (-c['mean_CE'], -c['mean_PQ'], c['condition']))
        winner = candidates[0]
        warnings = []
        for sample in winner['samples']:
            for metric in ('CE', 'PQ'):
                if sample[metric + '_delta'] < 0:
                    warnings.append(f'Seed {sample["seed"]}: {metric} is below the E2.8 equal-balance reference.')
            for concept, delta in sample['concept_deltas'].items():
                if delta < 0:
                    warnings.append(f'Seed {sample["seed"]}: {concept} CLAP margin is below the reference.')
        groups.append(dict(pair=list(pair), selected_condition=winner['condition'],
                           candidates=candidates, diagnostics=warnings,
                           CE_margin_over_runner_up=winner['mean_CE']-candidates[1]['mean_CE']))
        presets.append(dict(pair=list(pair), selection_metric='mean Content Enjoyment across seeds 707 and 909',
            energy=winner['energy'], sliders=winner['sliders'], mean_CE=winner['mean_CE'],
            mean_CE_gain=winner['mean_CE_delta'], mean_PQ_gain=winner['mean_PQ_delta'],
            selected_condition=winner['condition'], source_job_ids=[s['job_id'] for s in winner['samples']],
            diagnostics=warnings))
    result = dict(schema=1, selected_by='mean_CE',
        rule='Highest mean Content Enjoyment over both shared seeds; exact ties use mean PQ, then condition id.',
        objective_timing='Adopted after the completed sweep, following user clarification.',
        secondary_metrics='PQ and each requested concept margin are reported as diagnostics, not hidden gates or weighted terms.',
        human_listening_required=False, validation='Selection on the measured search samples; fresh-seed generalization is untested.',
        source_sha256={str(WORK / name): sha(WORK / name) for name in
                       ('screen.json', 'renders.json', 'measurements.json', 'audit.json', 'select_by_metric.py')},
        groups=groups)
    write(WORK / 'metric-selection.json', result)
    write(WORK / 'selected-presets.json', dict(schema=1, selection_sha256=sha(WORK / 'metric-selection.json'), presets=presets))
    lines = ['# Settings selected by Content Enjoyment', '',
        'The user clarified that this experiment should use the metric to choose settings.',
        'Selection now maximizes mean Content Enjoyment (CE) across the two shared seeds for each pair.',
        'Production Quality (PQ) and the two style margins are visible diagnostics. They do not silently change the objective.',
        'No listening ranking is required to make these selections.', '',
        '| Pair | Energy | Balance in pair order | Mean CE | CE gain vs reference | PQ gain vs reference |',
        '|---|---:|---:|---:|---:|---:|']
    sections = []
    def player(jid, label):
        rec = renders['records'][jid]
        return f'<article><h4>{html.escape(label)}</h4><audio controls preload="none" src="{Path(rec["excerpt"]).name}"></audio><a href="{Path(rec["audio"]).name}">Full recording</a></article>'
    for g, preset in zip(groups, presets):
        winner = g['candidates'][0]
        pair_label = ' + '.join(g['pair'])
        balance = f'{100*winner["share"]:.0f}/{100*(1-winner["share"]):.0f}'
        lines.append(f'| {pair_label} | {winner["energy"]["language_model"]:.1f} | {balance} | {winner["mean_CE"]:.3f} | {winner["mean_CE_delta"]:+.3f} | {winner["mean_PQ_delta"]:+.3f} |')
        parts = [f'<section><h2>{html.escape(pair_label)}</h2><p class="winner">Selected: energy {winner["energy"]["language_model"]:.1f}, balance {balance}</p>',
                 f'<p>Mean CE {winner["mean_CE"]:.3f} · gain {winner["mean_CE_delta"]:+.3f} over energy 2.8 at 50/50. Mean PQ change {winner["mean_PQ_delta"]:+.3f}.</p>']
        if g['diagnostics']:
            parts.append('<details><summary>Quality and style checks</summary><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in g['diagnostics']) + '</ul></details>')
        for sample in winner['samples']:
            parts.append(f'<h3>Seed {sample["seed"]}</h3><div class="grid">'
                         + player(sample['job_id'], f'Selected setting · CE {sample["CE"]:.3f}')
                         + player(sample['baseline_id'], 'Reference · E2.8 at 50/50') + '</div>')
        parts.append('<details><summary>All five settings, ranked by mean CE</summary><table><tr><th>Energy</th><th>Balance</th><th>CE</th><th>PQ</th></tr>')
        for candidate in g['candidates']:
            parts.append(f'<tr><td>{candidate["energy"]["language_model"]:.1f}</td><td>{candidate["share"]*100:.0f}/{(1-candidate["share"])*100:.0f}</td><td>{candidate["mean_CE"]:.3f}</td><td>{candidate["mean_PQ"]:.3f}</td></tr>')
        parts.append('</table></details></section>')
        sections.append(''.join(parts))
    lines += ['', 'Reference: energy 2.8 at 50/50. CE and PQ gains are score points, not percentages or measured human preference.', '',
        'Female + pop and country + indie rock improve CE and PQ over the reference on both seeds.',
        'The female-voice CLAP margin drops on one selected female + pop sample, so quality improvement is not uniform style improvement.',
        'House + acoustic folk at E3.2 has the highest mean CE, only about 0.024 ahead of E2.4. Its mean PQ falls by 0.058 versus the reference,',
        'and its CE improvement reverses on seed 909. E2.4 is the stronger alternative for quality and consistency, but E3.2 is the winner of the stated CE objective.', '',
        'These are the best tested settings on this search sample, selected after the sweep. They have not been validated on fresh seeds or songs.',
        'No automatic production routing or global slider default has been installed. The presets are saved in `selected-presets.json`.', '',
        '[Metric-selected results](http://192.168.1.90:7860/studio-combo-energy-20260907/selected.html).', '']
    (WORK / 'metric-selection.md').write_text('\n'.join(lines))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Metric-selected slider settings</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;background:#141a22;color:#e8edf4;font:16px/1.5 system-ui}a{color:#9ed3ff}section{margin:24px 0;padding:24px;border:1px solid #3a4455;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}audio{width:100%}summary{cursor:pointer;padding:10px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #3a4455}.winner{font-size:20px;color:#b8e7bd}</style><h1>Settings selected by Content Enjoyment</h1><p>The highest average CE across seeds 707 and 909 selects one setting per pair. Production quality and style scores remain separate checks. Listening is optional; the metric has made the selection.</p><p>These are selections from the 30 measured clips. Generalization to fresh seeds and songs has not yet been tested. Audio previews contain the first 20 seconds at their generated levels.</p><p><a href="selected-presets.json">Selected presets</a> · <a href="metric-selection.json">Selection data</a> · <a href="index.html">Full comparison</a></p>'''
    page += ''.join(sections) + '<script>document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'
    for folder in (OUTPUT, PUBLIC):
        (folder / 'selected.html').write_text(page)
        for filename in ('metric-selection.json', 'selected-presets.json'):
            (folder / filename).write_bytes((WORK / filename).read_bytes())
    print('\n'.join(lines[8:13]))


if __name__ == '__main__':
    main()
