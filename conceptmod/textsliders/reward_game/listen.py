"""Local listening index with raw controls, candidates and every regression."""
import html
import os
from pathlib import Path

from .core import ROOT,Store,read,write,sha


def build(home,board):
    home=Path(home);game=read(read(home/'game.json')['spec'])
    folder=ROOT/'eval/listen'/home.name;folder.mkdir(parents=True,exist_ok=True)
    audio_dir=folder/'audio';audio_dir.mkdir(exist_ok=True)
    hashes={ref['audio']:ref['audio_sha256'] for case in game['cases'] for ref in case['controls'].values()}
    def player(path):
        if not path or not Path(path).exists():return '<span>Audio unavailable</span>'
        if path not in hashes:hashes[path]=sha(path)
        key=hashes[path];target=audio_dir/(key+'.wav')
        if not target.exists():target.symlink_to(Path(path).resolve())
        return f'<audio controls preload="none" src="audio/{key}.wav"></audio>'
    e=html.escape
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Reward slider development game</title><style>body{font:16px system-ui;max-width:1280px;margin:2rem auto;padding:0 1rem;background:#151a21;color:#eef3f8}a{color:#83c6ff}details{margin:1rem 0;padding:1rem;background:#202936;border-radius:10px}summary{cursor:pointer}table{width:100%;border-collapse:collapse}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #445}audio{width:260px;max-width:100%}.loss{color:#ffaaa0}.gain{color:#9ee1b1}small{color:#aebccc}</style>',
           '<h1>Reward slider development game</h1><p>Fixed development comparisons. Off remains eligible. These exposed cases do not establish fresh confirmation.</p>',
           '<p>Players use the unchanged FLOAT WAVs. CE comparisons use the frozen normalized first 20 seconds. Lower-scoring examples remain visible.</p>']
    cards=[card for rows in board['stages'].values() for card in rows]+board['incomplete']
    for card in cards:
        off=card['comparisons']['off']
        progress=f'{off["wins"]} wins; {card["valid_cases"]}/{card["stage"]} cases scored' if card['decision']=='in_progress' else f'{off["wins"]}/{card["stage"]} wins'
        parts += [f'<details><summary>{e(card["run_id"])} · {progress} · {e(card["decision"])}</summary>',
                  '<table><tr><th>Case</th><th>Candidate</th><th>Off</th><th>Original strength 1</th><th>Original half strength</th></tr>']
        rows={r['case']:r for r in card['rows']}
        for case in game['cases']:
            if case['id'] not in game['stages'][str(card['stage'])]:continue
            row=rows.get(case['id']);parts += [f'<tr><td>{e(case["id"])}<br><small>{e(case["family"]["voice"])}</small></td>']
            if row:
                delta=row['deltas']['off'];color='loss' if delta<0 else 'gain'
                parts += [f'<td><span class="{color}">CE {row["ce"]:.4f}; {delta:+.4f} vs Off</span><br>{player(row.get("audio"))}</td>']
            else:parts += ['<td>Invalid or unattempted; retained in scorecard</td>']
            for label in ('off','v1-original','v1-half'):
                ref=case['controls'][label];parts += [f'<td>CE {ref["ce"]:.4f}<br>{player(ref["audio"])}</td>']
            parts += ['</tr>']
        parts += ['</table></details>']
    selections=list((home/'training').glob('*/selection.json'))
    if selections:
        parts += ['<h2>Training selections</h2><p>These examples selected adapter parameters. They are separate from development and confirmation evidence.</p>']
    store=Store(home)
    for path in selections:
        selection=read(path);proposal=read(path.parent/'proposal.json')
        chosen='Off' if selection['selected_off'] else f'pattern {selection["selected_index"]}'
        parts += [f'<details><summary>{e(path.parent.name)} · selected {e(chosen)}</summary>']
        for result in selection['results']:
            objective=f'{result["objective"]:+.4f}' if result['valid'] else 'invalid'
            parts += [f'<h3>Pattern {result["index"]}: {e(str(result["gains"]))}</h3><p>Training objective: {objective}</p>',
                      '<table><tr><th>Training case</th><th>Candidate</th><th>Matched Off</th></tr>']
            for key,case in zip(result['keys'],proposal['training_cases']):
                row=store.observation(key);ref=case['off_observation'];reward=(row or {}).get('reward') or {}
                value=reward.get('scalar');delta=value-ref['reward']['scalar'] if value is not None else None
                label=f'CE {value:.4f}; {delta:+.4f} vs Off' if delta is not None else e((row or {}).get('status','missing'))
                color='loss' if delta is not None and delta<0 else 'gain'
                hashes[ref['audio']]=ref['audio_sha256']
                parts += [f'<tr><td>{e(case["family"]["family"])} · seed {case["seed"]}</td>',
                          f'<td><span class="{color}">{label}</span><br>{player((row or {}).get("audio"))}</td>',
                          f'<td>CE {ref["reward"]["scalar"]:.4f}<br>{player(ref["audio"])}</td></tr>']
            parts += ['</table>']
        parts += ['</details>']
    matched=list((home/'training').glob('*/collection-recipe.json'))
    if matched:
        parts += ['<h2>Matched policy training pairs</h2><p>Training examples only. Each fixed prompt and seed compares Off with one frozen parent adapter. Higher CE chooses the imitation target; exact ties choose Off.</p>']
    for path in matched:
        root=path.parent;cases=read(root/'cases.json') if (root/'cases.json').exists() else []
        pairs=read(root/'preferences.json') if (root/'preferences.json').exists() else []
        targets={(p['family'],p['seed']):p['chosen'] for p in pairs}
        parts += [f'<details><summary>{e(root.name)} · {len(pairs)}/8 labels frozen</summary>',
                  '<table><tr><th>Training case</th><th>Parent adapter</th><th>Matched Off</th><th>Chosen target</th></tr>']
        for case in cases:
            ref=case['off'];row_path=root/'observations'/f"{case['id']}-matched-parent.json"
            row=read(row_path) if row_path.exists() else {};value=(row.get('reward') or {}).get('scalar')
            delta=value-ref['reward']['scalar'] if value is not None else None
            label=f'CE {value:.4f}; {delta:+.4f} vs Off' if delta is not None else e(row.get('status','unattempted'))
            color='loss' if delta is not None and delta<0 else 'gain';hashes[ref['audio']]=ref['audio_sha256']
            chosen=targets.get((case['family']['family'],case['seed']),'Pending all eight pairs')
            parts += [f'<tr><td>{e(case["id"])}</td><td><span class="{color}">{label}</span><br>{player(row.get("audio"))}</td>',
                      f'<td>CE {ref["reward"]["scalar"]:.4f}<br>{player(ref["audio"])}</td><td>{e(chosen)}</td></tr>']
        parts += ['</table></details>']
    policy_sets=list((home/'training').glob('*/episodes.json'))
    if policy_sets:
        parts += ['<h2>Parent rollout policy training</h2><p>Previously captured parent takes used for a signed code-policy objective. Advantages compare the two parent takes within each training family; these are training data, not fresh evidence.</p>']
    for path in policy_sets:
        episodes=read(path)
        parts += [f'<details><summary>{e(path.parent.name)} · {len(episodes)} recorded parent takes</summary>',
                  '<table><tr><th>Training case</th><th>CE</th><th>Frozen advantage</th><th>Parent audio</th></tr>']
        for episode in episodes:
            row=episode['observation'];hashes[row['audio']]=row['audio_sha256']
            parts += [f'<tr><td>{e(episode["id"])}</td><td>{episode["ce"]:.4f}</td>',
                      f'<td>{episode["advantage"]:+.3f}</td><td>{player(row["audio"])}</td></tr>']
        parts += ['</table></details>']
    parts += ['</html>'];(folder/'index.html').write_text('\n'.join(parts))
    write(home/'listening-page.json',dict(path=str(folder/'index.html'),evaluations=len(cards),training_selections=len(selections),matched_collections=len(matched),parent_policy_sets=len(policy_sets)))
    return str(folder/'index.html')
