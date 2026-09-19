"""Write a local listening comparison and training plot from measured entries."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import statistics

from .game import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    folder = home/'rounds'/args.round
    state = json.loads((home/'state.json').read_text())
    protocol = json.loads((home/'protocol.json').read_text())
    entries = sorted(state['entries'].values(), key=lambda entry: -entry['score'])
    candidates = [entry for entry in entries if entry['round'] == args.round]
    candidate = max(candidates, key=lambda entry: entry['score'])
    card = json.loads((folder/'round.json').read_text())
    previous = card['initial_best_scores']
    outcome = ('The candidate improved the MMD personal best.' if candidate['score'] > previous['mmd']
               else 'The candidate did not improve the MMD personal best on the audio judge.')
    training = Path(candidate['weights']).parent
    completion = json.loads((training/'completion.json').read_text())
    history = [json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    axes = grid.flat
    axes[0].plot([row['step'] for row in history], [row['line_search']['loss'] for row in history], label='Float32 continuation')
    axes[0].set(xlabel='Cumulative attempted updates', ylabel='Fixed paired MMD objective', title='Fixed float32 objective within this run')
    axes[0].legend()
    axes[1].plot([row['step'] for row in history], [row['parameter_step'] for row in history], color='#217c61')
    rejected = [row for row in history if not row['line_search']['accepted']]
    if rejected:
        axes[1].scatter([row['step'] for row in rejected], [0]*len(rejected), marker='x', color='#b74735', label='Rejected')
        axes[1].legend()
    axes[1].set(xlabel='Cumulative attempted updates', ylabel='Parameter movement, L2', title='Actual movement after line search')
    for precision in ('bf16', 'float32'):
        probe = json.loads((training/f'precision-{precision}.json').read_text())
        trials = probe['trials']
        axes[2].plot([trial['length'] for trial in trials],
                     [trial['finite_difference']/trial['predicted_derivative'] for trial in trials],
                     marker='o', label=precision)
    axes[2].axhline(1., color='#888888', linestyle='--', linewidth=1)
    axes[2].set(xscale='log', xlabel='Parameter displacement, L2',
                ylabel='Finite difference / autograd slope', title='Does the gradient predict the slope?')
    axes[2].legend()
    evaluations = [json.loads(path.read_text()) for path in training.glob('evaluation-*.json')]
    evaluations.sort(key=lambda row: row['step'])
    for kind in sorted({kind for row in evaluations for kind in row.get('train_groups', {})}):
        selected = [row for row in evaluations if kind in row.get('train_groups', {})]
        label = 'Active parent histories' if 'scale +1' in kind else 'Original neutral histories'
        axes[3].plot([row['step'] for row in selected], [row['train_groups'][kind] for row in selected], marker='o', label=label)
    axes[3].plot([row['step'] for row in evaluations],
                 [statistics.mean(v['loss'] for v in row['heldout']) for row in evaluations],
                 marker='o', linestyle='--', label='Reserved prompt histories')
    axes[3].set(xlabel='Cumulative attempted updates', ylabel='Fixed paired MMD objective',
                title='Which histories improve?')
    axes[3].legend()
    fig.savefig(folder/'training.png', dpi=160); fig.savefig(folder/'training.svg'); plt.close(fig)

    def escape(value): return html.escape(str(value))
    def link(path): return escape(os.path.relpath(path, folder))
    def audio(path): return f'<audio controls preload="none" src="{link(path)}"></audio>'
    tables = []
    comparison = []
    for row in protocol['rows']:
        for seed in protocol['seeds']:
            fixture = digest([protocol['prompts_sha256'], row, seed, protocol['duration']])
            records = [(entry, next(record for record in entry['records'] if record['fixture'] == fixture)) for entry in entries]
            baseline = records[0][1]['baseline']
            ref = records[0][1]['positive_reference']
            tables.append(f'<section><h2>Prompt {row}, seed {seed}</h2><div class="controls"><div>Slider off{audio(baseline["audio"])}</div><div>Positive caption{audio(ref["audio"])}</div></div>')
            tables.append('<table><thead><tr><th>Checkpoint</th><th>Audio</th><th>Score</th><th>Concept margin</th><th>Enjoyment / production</th><th>ASR phrase accuracy / recall</th></tr></thead><tbody>')
            for entry, record in records:
                sample = record['candidate']; lyric = sample['lyric_diagnostics']
                tables.append(f'<tr><th>{escape(entry["id"])}</th><td>{audio(sample["audio"])}</td><td>{record["heuristic_score"]:.4f}</td><td>{sample["concept"]:.4f}</td><td>{sample["enjoyment"]:.3f} / {sample["production"]:.3f}</td><td>{lyric["phrase_accuracy"]:.3f} / {lyric["recall"]:.3f}</td></tr>')
                comparison.append(dict(row=row, entry=entry['id'], **record))
            tables.append('</tbody></table></section>')
    rows = ''.join(f'<tr><th>{escape(entry["id"])}</th><td>{entry["score"]:.4f}</td><td>{entry["worst_score"]:.4f}</td><td>{escape(entry["eligible"])}</td></tr>' for entry in entries)
    final_loss = history[-1]['line_search']['loss']
    initial_loss = history[0]['loss_before']
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MMD {escape(args.round)}: {escape(candidate['label'])}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;background:#f5f4ef;color:#202b2a;max-width:1220px;margin:36px auto;padding:0 20px}}h1,h2{{line-height:1.2}}section{{background:white;padding:22px;margin:22px 0;border-radius:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px 10px;text-align:left;border-bottom:1px solid #deded7}}audio{{display:block;width:260px;height:36px;margin:10px 0}}img{{max-width:100%}}.controls{{display:flex;gap:30px;flex-wrap:wrap}}a{{color:#176b5d}}</style>
<h1>MMD {escape(args.round)}: {escape(candidate['label'])}</h1><p>{escape(card['hypothesis'])}</p><p>Locked audio comparison: two prompts × two seeds, 20 seconds each. Higher scores are better. This judge is a heuristic; listen to the individual clips.</p>
<section><h2>Round result</h2><p><strong>{escape(outcome)}</strong></p><p>Candidate: {candidate['score']:.4f}. Starting MMD champion: {previous['mmd']:.4f}. Starting overall leader: {previous['overall']:.4f}.</p><p><a href="notes.md">Full findings and next experiment</a> · <a href="training-findings.md">Precision and optimization evidence</a> · <a href="state-and-precision-audit.json">Saved-state audit</a></p></section>
<section><h2>Measured leaderboard</h2><table><thead><tr><th>Checkpoint</th><th>Mean score</th><th>Worst clip</th><th>Eligible</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Training</h2><p>{completion['attempted_updates']} attempted updates; {completion['accepted']} accepted. Float32 objective {initial_loss:.7f} → {final_loss:.7f}. Training took {completion['training_seconds']/60:.1f} minutes. Status: {escape(completion['status'])}.</p><img src="training.svg" alt="Training objective and actual parameter movement"><p><a href="notes.md">Round notes</a> · <a href="comparison.json">Complete per-clip measurements</a> · <a href="confirmation/plan.json">Predeclared confirmation plan</a></p></section>
{''.join(tables)}<p>Float32 teacher targets and calibration differ from the seed: raw losses across these objectives do not rank checkpoints. <a href="confirmation/summary.json">Additional-fixture results, if triggered</a>. Diversity and generalization require separate confirmation. Different files and a decreasing hidden-state loss do not establish either.</p></html>'''
    (folder/'index.html').write_text(page)
    (folder/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(folder/'index.html')


if __name__ == '__main__':
    main()
