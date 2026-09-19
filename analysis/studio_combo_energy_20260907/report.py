"""Two sets of blind triplets, sharing the center audio."""
import html
import json
from pathlib import Path
import random
from common import WORK, PUBLIC, OUTPUT, PAIRS, write


def assignments(spec):
    groups = []
    for axis, keys in [('energy', ('e24', 'center', 'e32')),
                       ('balance', ('balance40', 'center', 'balance60'))]:
        for seed in spec['seeds']:
            for pair in PAIRS:
                jobs = [j for j in spec['jobs'] if j['pair'] == list(pair)
                        and j['seed'] == seed and j['condition'] in keys]
                random.Random(f'combo-energy-20260907-{axis}-{seed}-{"+".join(pair)}').shuffle(jobs)
                assert len(jobs) == 3
                groups.append(dict(axis=axis, seed=seed, pair=list(pair), jobs=jobs))
    return groups


def build(state, output=OUTPUT):
    spec = json.loads((WORK / 'screen.json').read_text())
    groups = assignments(spec)
    write(WORK / 'blind-key.json', [dict(axis=g['axis'], seed=g['seed'], pair=g['pair'],
        letters={chr(65+i): dict(id=j['id'], condition=j['condition']) for i, j in enumerate(g['jobs'])})
        for g in groups])
    complete = sum(r['status'] == 'complete' for r in state['records'].values())
    def player(job, label):
        rec = state['records'].get(job['id'], {})
        if rec.get('status') != 'complete':
            return f'<article><h3>{label}</h3><p>{html.escape(rec.get("status", "pending"))}</p></article>'
        return (f'<article><h3>{label}</h3><audio controls preload="none" src="{Path(rec["excerpt"]).name}"></audio>'
                f'<a href="{Path(rec["audio"]).name}">Full recording ({rec["inspection"]["seconds"]:.1f}s)</a></article>')
    parts = []
    for axis, title in [('energy', 'Energy'), ('balance', 'Balance')]:
        parts.append(f'<h2 id="{axis}">{title}</h2>')
        parts.append('<p>' + ('Equal slider balance; compare three total strengths.' if axis == 'energy'
                     else 'Energy held at 2.8; compare three balances of the two controls.') + '</p>')
        for g in groups:
            if g['axis'] != axis:
                continue
            parts.append('<section><h3>' + html.escape(' + '.join(g['pair']))
                         + f' · seed {g["seed"]}</h3><div class="grid">'
                         + ''.join(player(j, chr(65+i)) for i, j in enumerate(g['jobs'])) + '</div>')
            parts.append('<p>Listen for a satisfying blend, a coherent groove and natural vocals. Letters change between groups.</p>')
            parts.append('<details><summary>Reveal settings</summary><ul>' + ''.join(
                f'<li>{chr(65+i)}: energy {j["energy"]["language_model"]:.1f}; '
                f'{j["share"]*100:.0f}/{(1-j["share"])*100:.0f} in the pair order shown</li>'
                for i, j in enumerate(g['jobs'])) + '</ul></details></section>')
    fixture = spec['jobs'][0]
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio combinations and energy</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;background:#141a22;color:#e8edf4;font:16px/1.5 system-ui}a{color:#9ed3ff}section{margin:24px 0;padding:24px;border:1px solid #3a4455;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}audio{width:100%}summary{cursor:pointer;padding:10px 0}pre{white-space:pre-wrap}h2{margin-top:48px}</style>
<h1>Studio combinations and energy</h1><p>Three pairs, two new seeds, one fixed lyric sheet and caption. Start with the energy triplets, then compare balance. Rank A/B/C within each group before revealing settings.</p>
<p>Every group includes energy 2.8 at equal balance. That same recording appears in both sections. Players contain the first 20 seconds; full recordings are linked underneath.</p>
<p><a href="#energy">Energy</a> · <a href="#balance">Balance</a></p>'''
    if (WORK / 'metric-selection.json').exists():
        page = page.replace('Start with the energy triplets, then compare balance. Rank A/B/C within each group before revealing settings.',
                            'Mean Content Enjoyment has selected the presets. These blind comparisons are available for optional listening.')
        page += '<p><strong><a href="selected.html">See the metric-selected presets, scores and reference audio</a></strong></p>'
    page += f'<p>{complete}/30 recordings complete · {html.escape(state["status"])}</p>'
    if state['status'] != 'complete':
        page += '<p>Refresh this page to load newly completed clips.</p>'
    page += ''.join(parts)
    page += '<details><summary>Fixed lyrics and caption</summary><pre>' + html.escape(fixture['lyrics'])
    page += '</pre><p>' + html.escape(fixture['caption']).replace('\n', '<br>') + '</p></details>'
    page += '<script>document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'
    for folder in (output, PUBLIC):
        temp = folder / 'index.html.tmp'
        temp.write_text(page)
        temp.replace(folder / 'index.html')


if __name__ == '__main__':
    build(json.loads((WORK / 'renders.json').read_text()))
