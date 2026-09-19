"""Build a listening page from completed, separately scored confirmation audio."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    folder = args.directory.resolve()
    summary = json.loads((folder/'summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Confirmation must finish before reporting')
    state = json.loads((Path(__file__).parent/'state.json').read_text())
    labels = {entry['weights']: entry['id'] for entry in state['entries'].values()}
    escape = lambda value: html.escape(str(value), quote=True)

    def audio(sample):
        path = Path(sample['audio'])
        if not path.is_file():
            raise FileNotFoundError(path)
        return f'<audio controls preload="none" src="{escape(os.path.relpath(path, folder))}"></audio>'

    sections = []
    for row in summary['plan']['rows']:
        for seed in summary['plan']['seeds']:
            records = [record for record in summary['records'] if record['row'] == row and record['seed'] == seed]
            if not records:
                raise ValueError('Missing planned fixture')
            sections.append(f'<section><h2>Prompt {row}, seed {seed}</h2><div class="controls"><div>Slider off{audio(records[0]["baseline"])}</div><div>Positive caption{audio(records[0]["positive_reference"])}</div></div>')
            sections.append('<table><thead><tr><th>Checkpoint</th><th>Audio</th><th>Score</th><th>Concept</th><th>Enjoyment / production</th><th>Phrase accuracy / recall</th></tr></thead><tbody>')
            for record in records:
                sample = record['candidate']
                lyric = sample['lyric_diagnostics']
                label = labels.get(record['checkpoint']['path'], Path(record['checkpoint']['path']).stem)
                sections.append(f'<tr><th>{escape(label)}</th><td>{audio(sample)}<details><summary>ASR transcript</summary>{escape(sample["transcript"])}</details></td><td>{record["heuristic_score"]:.4f}</td><td>{sample["concept"]:.4f}</td><td>{sample["enjoyment"]:.3f} / {sample["production"]:.3f}</td><td>{lyric["phrase_accuracy"]:.3f} / {lyric["recall"]:.3f}</td></tr>')
            sections.append('</tbody></table></section>')
    ranking = ''.join(f'<tr><th>{escape(labels.get(item["checkpoint"], Path(item["checkpoint"]).stem))}</th><td>{item["heuristic_score"]:.4f}</td><td>{item["minimum_clip_score"]:.4f}</td></tr>' for item in summary['ranking'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Additional audio confirmation</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#202b2a;background:#f5f4ef;max-width:1250px;margin:36px auto;padding:0 20px}}section{{background:white;border-radius:12px;margin:24px 0;padding:22px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #deded7;vertical-align:top}}audio{{display:block;width:260px;height:36px;margin:10px 0}}.controls{{display:flex;gap:32px;flex-wrap:wrap}}details{{max-width:300px}}a{{color:#176b5d}}</style>
<h1>Additional audio confirmation</h1><p>Two additional prompts, two seeds each, up to {summary['plan']['duration']:g} seconds. Scores use the unchanged heuristic and stay outside the development leaderboard. Higher is better.</p>
<p><a href="plan.json">Predeclared plan</a> · <a href="summary.json">Complete measurements and diversity diagnostics</a></p>
<section><h2>Measured comparison</h2><table><thead><tr><th>Checkpoint</th><th>Mean score</th><th>Worst clip</th></tr></thead><tbody>{ranking}</tbody></table><p>Inspect lyric and quality regressions clip by clip. Two seeds per prompt and full-mix embedding spread cannot establish mode coverage. No human listening verdict has been recorded.</p></section>
{''.join(sections)}</html>'''
    (folder/'index.html').write_text(page)
    print(folder/'index.html')


if __name__ == '__main__':
    main()
