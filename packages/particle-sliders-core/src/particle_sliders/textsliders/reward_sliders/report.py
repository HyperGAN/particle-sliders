"""Readable decision record without changing collection or training inputs."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from .specs import DEFAULT_RUN, read_json


def comparison_table(comparisons):
    lines = ['| Comparison | Valid pairs | CE change | Seed wins | Family bootstrap 95% interval |',
             '|---|---:|---:|---:|---|']
    for label, row in comparisons.items():
        change = 'missing' if row['mean_delta'] is None else f"{row['mean_delta']:+.4f}"
        wins = 'missing' if row['seed_win_rate'] is None else f"{row['seed_win_rate']:.1%}"
        interval = 'missing' if row['family_bootstrap_95'] is None else ', '.join(f'{v:+.4f}' for v in row['family_bootstrap_95'])
        lines.append(f"| {label} | {row['valid_pairs']} | {change} | {wins} | {interval} |")
    return lines+['']


def report(run):
    run=Path(run)
    manifest=read_json(run/'manifest.json')
    current=read_json(run/'status.json')
    rows=[read_json(p) for p in sorted((run/'observations').glob('*.json'))]
    transfer_rows=[read_json(p) for p in sorted((run/'transfer/observations').glob('*.json'))]
    complete=sum(r['status']=='complete' for r in rows+transfer_rows)
    failed=[r for r in rows+transfer_rows if r['status']=='failed']
    lines=['# reward-ce-v1','',
           'Content Enjoyment activation experiment with gated, ordinary rank-8/alpha-8 Music 3 LoRA training.','',
           f"Current stage: **{current['stage']}**. Completed observations: {complete}. Failed observations: {len(failed)}.",'',
           '[Frozen protocol](manifest.json) · [Status](status.json) · [Capture audit](audit/parity.json) · [Prompt audit](prompt-audit.json)','',
           'The pilot fixes eight training, four development and four test families before scoring. '
           'Each exact caption/sheet/style cell has four seeds. CE is the only selection reward; other axes remain diagnostics. '
           'Both CFG branches are captured and steered after the complete prompt prefill. '
           'The captured frame feedback supplies shared histories for teacher/student residuals.','',
           'The live GPU audit compares decoded PCM, because FLOAT WAV files contain a timestamp in their PEAK chunk. '
           'The original file hashes remain part of each observation. The first engineering audit and its correction '
           'are retained under [amendments](amendments/0001-pcm-parity/record.json).','']
    if 'completed' in current:
        lines += [f"Stage progress: {current['completed']}" + (f"/{current['total']}" if 'total' in current else '') + '.','']
    if (run/'audit/listening-page.json').exists():
        listen = read_json(run/'audit/listening-page.json')
        lines += [f"[Listen to all eight matched development A/B pairs]({listen['url']}). "
                  'Playback includes optional level matching and original WAV downloads.','']
    if (run/'fit.json').exists():
        lines += ['## Direction fit','','| Layer | Training rank agreement | Dev rank agreement | Median residual L2 |',
                  '|---|---:|---:|---:|']
        for layer,row in read_json(run/'fit.json').items():
            lines.append(f"| {layer} | {row['training']['mean']:.3f} | {row['development']['mean']:.3f} | {row['residual_median_l2']:.3f} |")
        lines += ['','Association alone does not establish that steering improves audio.','']
    if (run/'dev-selection.json').exists():
        selection=read_json(run/'dev-selection.json')
        lines += [f"Development selection: **{selection['selected']['name']}**.",'',
                  '| Arm | Mean normalized CE |','|---|---:|']
        for name,value in selection['mean_ce'].items():
            lines.append(f"| {name} | {'failed' if value is None else f'{value:.4f}'} |")
        lines.append('')
    if (run/'causal-result.json').exists():
        gate=read_json(run/'causal-result.json')
        lines += ['## Causal result','',f"LoRA training gate: **{'passed' if gate['passed'] else 'closed'}**.",'',
                  '| Comparison | Paired CE change | Seed wins | Family bootstrap 95% interval |','|---|---:|---:|---|']
        for name in ('positive_vs_off','positive_vs_random','reversed_vs_off'):
            row=gate[name]
            change = 'missing' if row['mean_delta'] is None else f"{row['mean_delta']:+.4f}"
            wins = 'missing' if row['seed_win_rate'] is None else f"{row['seed_win_rate']:.1%}"
            interval = 'missing' if row['family_bootstrap_95'] is None else ', '.join(f'{v:+.4f}' for v in row['family_bootstrap_95'])
            lines.append(f"| {name.replace('_',' ')} | {change} | {wins} | {interval} |")
        lines += ['','The pilot gate requires at least +0.02 CE against both Off and the fixed random direction, '
                  'more than half of seed pairs won against Off, and no unresolved arm failures. '
                  'Only a fresh full-song transfer study can validate the whole selected method.','']
        if gate['passed'] and any(gate[k]['family_bootstrap_95'][0]<=0 for k in ('positive_vs_off','positive_vs_random')):
            lines += ['The pilot meets its declared continuation rule, but the family bootstrap intervals '
                      'still overlap zero. Proceeding to a student does not establish a reliable or broad quality gain.','']
        if not gate['passed'] and not (run/'student/latest.json').exists():
            lines += ['No LoRA was trained: the causal result did not pass the frozen gate. '
                      'The generation-span trainer, standard export and transfer evaluation are implemented, '
                      'but have no trained Music 3 reward artifact from this failed probe.','']
        lines += ['### Matched family effects','','| Family | Positive − Off CE | Positive − random CE | Reversed − Off CE |',
                  '|---|---:|---:|---:|']
        comparisons = [gate[k]['family_deltas'] for k in ('positive_vs_off','positive_vs_random','reversed_vs_off')]
        for family in sorted(set().union(*(c.keys() for c in comparisons))):
            values = [f'{c[family]:+.4f}' if family in c else 'missing' for c in comparisons]
            lines.append(f"| {family} | {' | '.join(values)} |")
        lines += ['', 'These four families, not their individual windows or seeds, are the independent pilot units.','']
        paired = defaultdict(dict)
        for row in rows:
            if (row['split']=='test' and row['arm']['name'] in ('positive','off')
                and row['status']=='complete' and row.get('reward',{}).get('valid')):
                paired[(row['family'],row['cell_hash'],row['seed'])][row['arm']['name']] = row
        axes = defaultdict(lambda: defaultdict(list))
        for (family, _, _), pair in paired.items():
            if set(pair) != {'positive','off'}:
                continue
            for axis in ('CE','PQ','PC','CU'):
                axes[axis][family].append(pair['positive']['reward']['diagnostics']['axes'][axis]
                                          - pair['off']['reward']['diagnostics']['axes'][axis])
        if axes:
            lines += ['### Other scorer axes','','| Axis | Positive − Off, equal family weight |','|---|---:|']
            for axis, families in axes.items():
                mean = sum(sum(v)/len(v) for v in families.values())/len(families)
                lines.append(f'| {axis} | {mean:+.4f} |')
            lines += ['', 'Only CE selected the teacher. These diagnostics are not a combined reward. '
                      'Every raw and normalized window score remains in the [observations](observations/).','']
    intent_path = run/'evaluation/causal-intent/summary.json'
    if intent_path.exists():
        intent = read_json(intent_path)
        lines += ['## Pilot intent diagnostics','',
                  f"Valid matched pairs: {intent['valid_pairs']}/16. Invalid diagnostic observations: {len(intent['invalid'])}.",'',
                  '| Measurement | Positive − Off, equal family weight |','|---|---:|']
        for metric, value in intent['mean_changes'].items():
            lines.append(f"| {metric.replace('_',' ')} | {value:+.5f} |")
        lines += ['', 'These post-selection measurements describe the short pilot audio; they do not establish '
                  'full-song preservation and do not change the causal gate. '
                  '[Pair and family details](evaluation/causal-intent/summary.json) include output embedding diversity.','']
    if (run/'student/latest.json').exists():
        latest=read_json(run/'student/latest.json')
        lines += ['## LoRA checkpoints','',
                  f"Latest exported step: {latest['step']}. [{Path(latest['weights']).name}]({Path(latest['weights']).relative_to(run)}).",'',
                  'Full training states remain separate from inference safetensors. Checkpoints and effective strengths '
                  'are selected with development audio CE, followed by matched composition controls and fresh transfer songs.','']
        lines += ['| Step | Prompt relative RMS range | Generation residual relative error range | Teacher residual cosine range |',
                  '|---|---:|---:|---:|']
        for path in sorted((run/'student').glob('diagnostics-step*.json')):
            step = path.stem.removeprefix('diagnostics-step')
            diagnostics = read_json(path)
            ranges = []
            for metric in ('prompt_relative_rms','generation_relative_error','residual_cosine'):
                values = [r[metric] for r in diagnostics]
                ranges.append(f'{min(values):.3f}–{max(values):.3f}')
            lines.append(f"| [{step}]({path.relative_to(run)}) | {' | '.join(ranges)} |")
        lines += ['', 'These are conditional-branch, same-history diagnostics on one row per training family at multiplier 1. '
                  'Passing the gross-divergence bounds does not establish preserved words or improved free-running audio.','']
    if (run/'evaluation/selection.json').exists():
        selection = read_json(run/'evaluation/selection.json')
        lines += ['## LoRA development selection','',f"Selected arm: **{selection['selected']['name']}**.",'',
                  '| Arm | Mean normalized CE |','|---|---:|']
        for name, value in selection['mean_ce'].items():
            lines.append(f"| {name} | {'failed' if value is None else f'{value:.4f}'} |")
        lines += ['', '[Frozen evaluation recipe](evaluation/manifest.json). '
                  'Solo effective strength is controlled through host energy; a lone fader does not sweep strength.','']
    if (run/'evaluation/composition-results.json').exists():
        result = read_json(run/'evaluation/composition-results.json')
        lines += ['## Composition controls','']+comparison_table({
            'Added LoRA, original style multipliers fixed':result['isolated'],
            'Added LoRA, total energy fixed':result['fixed_energy'],
            'Added LoRA versus matched reduced styles':result['matched_reduced_style']})
        lines += ['[Exact resolved multipliers and observations](evaluation/composition-results.json).','']
    if (run/'evaluation/pilot-test-lora.json').exists():
        result = read_json(run/'evaluation/pilot-test-lora.json')
        lines += ['## Pilot LoRA comparison','']+comparison_table({'LoRA versus Off':result['statistics']})
        lines += [result['limitation']+'.','']
    if (run/'evaluation/runtime.json').exists():
        timing = read_json(run/'evaluation/runtime.json')
        lines += ['## Matched runtime','',
                  '| Treatment | Median end-to-end overhead | 10th–90th percentile | Median overhead excluding setup | Median setup seconds |',
                  '|---|---:|---|---:|---:|']
        for key, label in (('activation','Activation teacher'),('lora','Merged LoRA')):
            interval = ', '.join(f'{v:+.1%}' for v in timing[f'{key}_overhead_p10_p90'])
            core = median((r[key]['total_seconds']-r[key]['setup_merge_seconds']) /
                          (r['off']['total_seconds']-r['off']['setup_merge_seconds'])-1 for r in timing['pairs'])
            setup = median(r[key]['setup_merge_seconds'] for r in timing['pairs'])
            lines.append(f"| {label} | {timing[f'median_{key}_overhead']:+.1%} | {interval} | {core:+.1%} | {setup:.2f} |")
        lines += ['', f"Engineering target: at most {timing['target_fraction']:.0%} median overhead. "+timing['limitation']+'.',
                  'The development grid changes the adapter mix between LoRA/teacher samples; consecutive Off seeds '
                  'can reuse a mix. End-to-end differences therefore include unequal setup-cache patterns. '
                  'The separately reported time excluding setup isolates that component, but does not remove concurrent-load variability.',
                  '[Setup, merge, generation and peak-memory records](evaluation/runtime.json).','']
    if transfer_rows:
        transfer_manifest = read_json(run/'transfer/manifest.json')
        limits = {f['family']:f for f in transfer_manifest['families']}
        lines += ['## Full-song completion','',
                  '| Arm | Valid duration | Cap hits | Too short | Other failures | Running |',
                  '|---|---:|---:|---:|---:|---:|']
        for arm in ('off','lora'):
            counts = dict(valid=0,cap=0,short=0,other=0,running=0)
            for row in (r for r in transfer_rows if r['arm']['name']==arm):
                if row['status']=='running':
                    counts['running'] += 1
                elif row['status']=='complete':
                    counts['valid'] += 1
                else:
                    duration = (row.get('reward') or {}).get('diagnostics',{}).get('duration_s')
                    family = limits[row['family']]
                    kind = ('cap' if duration is not None and duration>=family['render_cap_seconds']-.5 else
                            'short' if duration is not None and duration<family['intended_seconds']*.5 else 'other')
                    counts[kind] += 1
            lines.append(f"| {arm} | {counts['valid']} | {counts['cap']} | {counts['short']} | {counts['other']} | {counts['running']} |")
        lines += ['', 'Each arm has 24 declared outputs. Cap hits and short outputs remain failures even when the '
                  'scorer returns a CE value; their raw scores and audio remain in the [transfer observations](transfer/observations/).','']
    if (run/'transfer/results.json').exists():
        result = read_json(run/'transfer/results.json')
        lines += ['## Fresh full-song transfer','',
                  'Twelve fresh families, two fresh seeds, and one frozen checkpoint/strength rule. '
                  'Primary CE covers every disjoint ten-second window, including the actual tail, weighted by duration.','']
        lines += comparison_table({'LoRA versus Off':result['statistics'],**result['section_changes']})
        if result['statistics']['valid_pairs'] == 0:
            lines += ['There are no valid full-song pairs, so full-song CE improvement and intent preservation '
                      'cannot be estimated from this run. Cap failures in either arm do not establish a '
                      'LoRA-specific completion problem. The short-clip results remain separate evidence.','']
        if result['statistics']['family_deltas']:
            lines += ['| Family | Full-song CE change |','|---|---:|']
            for family, value in result['statistics']['family_deltas'].items():
                lines.append(f'| {family} | {value:+.4f} |')
            lines.append('')
        for attribute, groups in result['subgroups'].items():
            lines += [f'### By {attribute}','']+comparison_table(groups)
        preservation = result['preservation']
        preservation_label = ('not assessable' if not preservation['pairs'] else
                              'passed' if preservation['passed'] else 'failed')
        lines += [f"Preservation checks: **{preservation_label}**. "
                  f"Reported issues: {len(preservation['issues'])}. Generation failures: {len(result['failures'])}.",'',
                  '[Pair-level intent changes, failures, diversity and decision](transfer/results.json) · '
                  '[Frozen transfer fixtures and tolerances](transfer/manifest.json)','',
                  'Short-pilot variability calibrates intent tolerances; duration, silence and diversity bounds are '
                  'declared engineering limits, not estimates of long-song population variability.','']
    if (run/'decision.json').exists():
        decision=read_json(run/'decision.json')
        lines += [f"Decision: **{decision['decision']}**.",'',f"{decision.get('reason','See the machine-readable decision for all statistical results.')}",'',
                  '[Decision details](decision.json)','']
    if failed:
        lines += ['## Failed observations','','| Observation | Error |','|---|---|']
        lines += [f"| {r['id']} | {r['error']} |" for r in failed]
        lines.append('')
    if (run/'audit/tests.json').exists():
        lines += ['[Implementation test record](audit/tests.json) · [Test details](audit/unit-tests.xml)','']
    if (run/'audit/final-integrity.json').exists():
        lines += ['[Final artifact and protocol integrity audit](audit/final-integrity.json)','']
    lines += ['The studio uses GPU 0 and the investigation uses physical GPU 1. '
              'Production slider weights and registry settings are not changed by this research pipeline.','',
              'Run or resume the frozen investigation with:','',
              '```bash',
              'systemctl --user status music-reward-ce-v1-20260907.service',
              'journalctl --user -u music-reward-ce-v1-20260907.service -n 30 --no-pager','```','']
    (run/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    report(parser.parse_args().run_dir)
