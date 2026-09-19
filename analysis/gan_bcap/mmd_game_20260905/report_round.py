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
    parser.add_argument('--round', default='round-0001')
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    folder = home/'rounds'/args.round
    state = json.loads((home/'state.json').read_text())
    protocol = json.loads((home/'protocol.json').read_text())
    entries = sorted(state['entries'].values(), key=lambda entry: -entry['score'])
    candidates = [entry for entry in entries if entry['round'] == args.round]
    candidate = max(candidates, key=lambda entry: entry['score'])
    training = Path(candidate['weights']).parent
    completion = json.loads((training/'completion.json').read_text())
    history = [json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()]
    source_history = [json.loads(line) for line in (Path(json.loads((training/'manifest.json').read_text())['arguments']['source_state']).parent/'train.jsonl').read_text().splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].plot([row['step'] for row in source_history], [row['line_search']['loss'] for row in source_history], label='Seed training')
    axes[0].plot([row['step'] for row in history], [row['line_search']['loss'] for row in history], label='Adaptive continuation')
    axes[0].set(xlabel='Cumulative attempted updates', ylabel='Fixed paired MMD objective', title='Training objective, unchanged units')
    axes[0].legend()
    axes[1].plot([row['step'] for row in history], [row['parameter_step'] for row in history], color='#217c61')
    rejected = [row for row in history if not row['line_search']['accepted']]
    axes[1].scatter([row['step'] for row in rejected], [0]*len(rejected), marker='x', color='#b74735', label='Rejected')
    axes[1].set(xlabel='Cumulative attempted updates', ylabel='Parameter movement, L2', title='Actual movement after line search')
    axes[1].legend()
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
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MMD round 1 — adaptive updates</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;background:#f5f4ef;color:#202b2a;max-width:1220px;margin:36px auto;padding:0 20px}}h1,h2{{line-height:1.2}}section{{background:white;padding:22px;margin:22px 0;border-radius:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px 10px;text-align:left;border-bottom:1px solid #deded7}}audio{{display:block;width:260px;height:36px;margin:10px 0}}img{{max-width:100%}}.controls{{display:flex;gap:30px;flex-wrap:wrap}}a{{color:#176b5d}}</style>
<h1>MMD round 1: adaptive updates</h1><p>Locked audio comparison: two prompts × two seeds, 20 seconds each. Higher scores are better. This judge is a heuristic; listen to the individual clips.</p>
<section><h2>Measured leaderboard</h2><table><thead><tr><th>Checkpoint</th><th>Mean score</th><th>Worst clip</th><th>Eligible</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Training</h2><p>{completion['attempted_updates']} attempted updates; {completion['accepted']} accepted. Fixed objective {initial_loss:.7f} → {final_loss:.7f}. Training took {completion['training_seconds']/60:.1f} minutes. Status: {escape(completion['status'])}.</p><img src="training.svg" alt="Training objective and actual parameter movement"><p><a href="notes.md">Round notes</a> · <a href="comparison.json">Complete per-clip measurements</a> · <a href="confirmation/plan.json">Predeclared confirmation plan</a></p></section>
{''.join(tables)}<p>Diversity and generalization require separate confirmation. Different files and a decreasing hidden-state loss do not establish either.</p></html>'''
    (folder/'index.html').write_text(page)
    (folder/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(folder/'index.html')


if __name__ == '__main__':
    main()
