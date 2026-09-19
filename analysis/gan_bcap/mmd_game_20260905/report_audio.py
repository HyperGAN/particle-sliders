"""Create a method-neutral local comparison from registered game measurements."""
import argparse
import html
import json
import os
from pathlib import Path

from .game import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    folder = home/'rounds'/args.round
    state = json.loads((home/'state.json').read_text())
    protocol = json.loads((home/'protocol.json').read_text())
    card = json.loads((folder/'round.json').read_text())
    entries = sorted(state['entries'].values(), key=lambda entry: -entry['score'])
    candidates = [entry for entry in entries if entry['round'] == args.round and entry['family'] != 'reference']
    if not candidates:
        raise ValueError('The round needs a measured candidate')
    candidate = max(candidates, key=lambda entry: entry['score'])
    escape = lambda value: html.escape(str(value), quote=True)
    link = lambda path: escape(os.path.relpath(path, folder))

    def audio(sample):
        path = Path(sample['audio'])
        if not path.is_file():
            raise FileNotFoundError(path)
        return f'<audio controls preload="none" src="{link(path)}"></audio>'

    sections, comparison = [], []
    for row in protocol['rows']:
        for seed in protocol['seeds']:
            fixture = digest([protocol['prompts_sha256'], row, seed, protocol['duration']])
            selected = [(entry, next(record for record in entry['records'] if record['fixture'] == fixture)) for entry in entries]
            sections.append(f'<section><h2>Prompt {row}, seed {seed}</h2><div class="controls"><div>Slider off{audio(selected[0][1]["baseline"])}</div><div>Positive caption{audio(selected[0][1]["positive_reference"])}</div></div>')
            sections.append('<table><thead><tr><th>Entry</th><th>Audio</th><th>Score</th><th>Concept</th><th>Enjoyment / production</th><th>Phrase accuracy / recall</th></tr></thead><tbody>')
            for entry, record in selected:
                sample = record['candidate']; lyrics = sample['lyric_diagnostics']
                sections.append(f'<tr><th>{escape(entry["label"])}</th><td>{audio(sample)}<details><summary>ASR transcript</summary>{escape(sample["transcript"])}</details></td><td>{record["heuristic_score"]:.4f}</td><td>{sample["concept"]:.4f}</td><td>{sample["enjoyment"]:.3f} / {sample["production"]:.3f}</td><td>{lyrics["phrase_accuracy"]:.3f} / {lyrics["recall"]:.3f}</td></tr>')
                comparison.append(dict(row=row, entry=entry['id'], **record))
            sections.append('</tbody></table></section>')
    ranking = ''.join(f'<tr><th>{escape(entry["label"])}</th><td>{escape(entry["family"])}</td><td>{entry["score"]:.4f}</td><td>{entry["worst_score"]:.4f}</td><td>{escape(entry["eligible"])}</td></tr>' for entry in entries)
    previous = card['initial_best_scores']
    gained = candidate['score']-previous['overall']
    manifest = Path(candidate['weights']).parent/'manifest.json'
    method = json.loads(manifest.read_text())
    detail = (f'Effective adapter strength: {method["factor"]:g}× the parent. No optimizer updates; the trained matrices are unchanged.'
              if candidate['family'] == 'calibration' else str(method.get('objective', method.get('method', 'See the recipe'))))
    status = 'New development lead' if candidate['eligible'] and gained > 1e-9 else 'The candidate did not beat the starting overall leader'
    confirmation = ('<a href="confirmation/index.html">Additional confirmation</a>'
                    if (folder/'confirmation/index.html').exists() else
                    '<a href="confirmation/outcome.json">Confirmation trigger outcome</a>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(args.round)} audio comparison</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#202b2a;background:#f5f4ef;max-width:1300px;margin:36px auto;padding:0 20px}}section{{background:white;border-radius:12px;margin:24px 0;padding:22px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #deded7;vertical-align:top}}audio{{display:block;width:260px;height:36px;margin:10px 0}}.controls{{display:flex;gap:32px;flex-wrap:wrap}}details{{max-width:300px}}a{{color:#176b5d}}</style>
<h1>{escape(args.round)}: {escape(candidate['label'])}</h1><p>{escape(card['hypothesis'])}</p><p>{escape(detail)}</p>
<section><h2>{escape(status)}</h2><p>Candidate <strong>{candidate['score']:.4f}</strong>; starting leader {previous['overall']:.4f}; change {gained:+.4f}. Starting MMD record: {previous['mmd']:.4f}. This candidate's family is {escape(candidate['family'])}.</p><p><a href="notes.md">Complete findings</a> · <a href="{link(manifest)}">Exact candidate recipe</a> · <a href="comparison.json">All clip measurements</a> · <a href="confirmation/plan.json">Predeclared confirmation</a></p></section>
<section><h2>Measured development leaderboard</h2><table><thead><tr><th>Entry</th><th>Method family</th><th>Mean score</th><th>Worst clip</th><th>Eligible</th></tr></thead><tbody>{ranking}</tbody></table><p>Locked render-heuristic-v2, two prompts × two seeds, 20 seconds, renderer scale +1. Shared controls are byte-identical. Repeated development scores are not a human listening verdict or evidence of generalization.</p></section>
{''.join(sections)}<p>{confirmation}. Cross-seed diversity requires separate measurements; distinct files do not establish mode coverage.</p></html>'''
    (folder/'index.html').write_text(page)
    (folder/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(folder/'index.html')


if __name__ == '__main__':
    main()
