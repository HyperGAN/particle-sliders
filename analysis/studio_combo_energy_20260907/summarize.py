"""Report paired changes without fitting or declaring a preference winner."""
import csv
import io
import json
import statistics
from common import WORK, PAIRS, sha, write


def main():
    spec = json.loads((WORK / 'screen.json').read_text())
    renders = json.loads((WORK / 'renders.json').read_text())
    measured = json.loads((WORK / 'measurements.json').read_text())
    assert renders['status'] == measured['status'] == 'complete'
    assert renders['screen_sha256'] == measured['screen_sha256'] == sha(WORK / 'screen.json')
    rows = []
    for job in spec['jobs']:
        rec = renders['records'][job['id']]
        m = measured['records'][job['id']]
        assert m['status'] == 'complete' and m['excerpt_sha256'] == rec['excerpt_sha256']
        rows.append(dict(id=job['id'], pair='+'.join(job['pair']), seed=job['seed'],
            condition=job['condition'], energy=job['energy']['language_model'], share=job['share'],
            device=rec['device'], seconds=rec['inspection']['seconds'],
            **m['aesthetics'], concept_a=m['concepts'][job['pair'][0]],
            concept_b=m['concepts'][job['pair'][1]],
            lyric_phrase=m['lyric_diagnostics']['phrase_accuracy'],
            lyric_recall=m['lyric_diagnostics']['recall'],
            excerpt_rms=m['rms'],
            peak=rec['inspection']['peak'], clipped_fraction=rec['inspection']['clipped_fraction']))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (WORK / 'results.csv').write_text(buf.getvalue())
    axes = ['CE', 'PQ', 'PC', 'CU', 'concept_a', 'concept_b', 'lyric_phrase', 'lyric_recall']
    changes = []
    for pair in PAIRS:
        key = '+'.join(pair)
        for condition in ('e24', 'e32', 'balance40', 'balance60'):
            deltas = []
            for seed in spec['seeds']:
                c = next(r for r in rows if r['pair'] == key and r['seed'] == seed and r['condition'] == condition)
                b = next(r for r in rows if r['pair'] == key and r['seed'] == seed and r['condition'] == 'center')
                deltas.append(dict(seed=seed, **{k: c[k]-b[k] for k in axes}))
            changes.append(dict(pair=key, condition=condition, per_seed=deltas,
                                mean={k: statistics.mean(d[k] for d in deltas) for k in axes}))
    write(WORK / 'paired-changes.json', dict(screen_sha256=sha(WORK / 'screen.json'),
        baseline='center: E2.8, 50/50', changes=changes,
        limitation='Two seeds on one existing song; diagnostics only, no human preferences yet.'))
    lines = ['# Combination and energy diagnostics', '',
        'All 30 studio renders and CPU measurements completed. These results describe the first 20 seconds.',
        'Aesthetics and CLAP use RMS-normalized copies; listening previews retain the generated levels.',
        'Following the user clarification, mean Content Enjoyment selects the best tested settings.',
        '[Selected presets and tradeoffs](metric-selection.md). Listening is optional, not a prerequisite for selection.', '',
        '[Blind listening page](http://192.168.1.90:7860/studio-combo-energy-20260907/).', '',
        'Each table averages two fresh seeds on the same existing lyric sheet and caption. Differences across',
        'seeds are retained in `results.csv` and `paired-changes.json`; two samples are not a confidence interval.',
        'Both cards rendered production recuts. Energy splits between faders: 50/50 gives 1.2/1.2 at E2.4,',
        '1.4/1.4 at E2.8 and 1.6/1.6 at E3.2. At E2.8 the unequal balances give 1.12/1.68 or 1.68/1.12.', '',
        'CE = content enjoyment; PQ = production quality; PC = production complexity. Higher complexity is',
        'not necessarily desirable. Each CLAP margin measures one requested concept separately. ASR is a',
        'diagnostic: an instrumental intro can score poorly against the full lyric sheet.', '',
        '## Metric observations', '',
        '- At equal balance, E2.4 scores above E2.8 on PQ in all six matched comparisons and on CE in five.',
        '  These are six comparisons from only two seeds on one song, and some gaps are small.',
        '- The highest energy-triplet CE score is at E3.2 for seed 707 and E2.4 for seed 909, for every pair.',
        '  The female + pop seed-707 CE lead over E2.8 is only about 0.005.',
        '- At E2.8, 60/40 improves CE and PQ over 50/50 on both seeds for female + pop and country + indie rock.',
        '  The country seed-707 PQ increase is only about 0.002. For female + pop, the feminine-voice CLAP margin',
        '  decreases on seed 909 despite the quality-score gains; this is not uniform improvement on all goals.',
        '- House + acoustic folk is sensitive to the seed: 60/40 improves CE/PQ on 707 and reduces both on 909.',
        '  Both unequal balances also reduce the acoustic-folk margin on seed 909.', '',
        'The numerical selection is reported separately from these diagnostics. Combining E2.4 with 60/40',
        'would be an untested interaction; the screen has not rendered that combination.', '']
    for pair in PAIRS:
        key = '+'.join(pair)
        lines += ['## ' + ' + '.join(pair), '',
                  '| Setting | CE | PQ | PC | First concept | Second concept | Lyric phrase |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for condition in spec['conditions']:
            chosen = [r for r in rows if r['pair'] == key and r['condition'] == condition['key']]
            values = [statistics.mean(r[k] for r in chosen) for k in ('CE','PQ','PC','concept_a','concept_b','lyric_phrase')]
            label = f'E{condition["energy"]:.1f}, {condition["share"]*100:.0f}/{(1-condition["share"])*100:.0f}'
            lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
        lines += ['', 'Changes from E2.8 at 50/50, shown separately for each seed:', '',
                  '| Setting | CE Δ seed 707 | CE Δ seed 909 | PQ Δ seed 707 | PQ Δ seed 909 |',
                  '|---|---:|---:|---:|---:|']
        for entry in [c for c in changes if c['pair'] == key]:
            a, b = entry['per_seed']
            lines.append(f'| {entry["condition"]} | {a["CE"]:+.3f} | {b["CE"]:+.3f} | {a["PQ"]:+.3f} | {b["PQ"]:+.3f} |')
        lines += ['']
    lines += ['## Scope', '',
        'This is a local cross-shaped sweep. Energy comparisons use equal balance; balance comparisons use E2.8.',
        'It does not establish the best balance at other energies, a universal optimum, a solo energy default,',
        'or a result for other songs or three-slider combinations. The repeated center recording in the energy',
        'and balance listening sections is one observation, not two independent renders.', '',
        'The selected presets maximize mean CE on these search samples; fresh-seed generalization remains untested.',
        'The production generator, weights and default energy are unchanged.', '']
    (WORK / 'results.md').write_text('\n'.join(lines))
    print('Wrote results.csv, results.md and paired-changes.json', flush=True)


if __name__ == '__main__':
    main()
