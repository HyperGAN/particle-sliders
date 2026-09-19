"""Local listening page with concealed, fixed-randomized energy conditions."""
import html
import json
from pathlib import Path
import random

WORK = Path(__file__).resolve().parent


def build(state, output):
    spec = json.loads((WORK/'screen.json').read_text())
    records = state['records']
    completed = sum(r['status'] == 'complete' for r in records.values())
    failed = sum(r['status'] == 'error' for r in records.values())

    def player(job, label=None):
        rec = records.get(job['id'], {})
        title = html.escape(label or job['label'])
        if rec.get('status') == 'complete':
            excerpt = html.escape(Path(rec['excerpt']).name)
            full = html.escape(Path(rec['audio']).name)
            return (f'<article><h3>{title}</h3><audio controls preload="none" src="{excerpt}"></audio>'
                    f'<a href="{full}">Full audio ({rec["inspection"]["seconds"]:.1f}s)</a></article>')
        status = 'failed' if rec.get('status') == 'error' else rec.get('status', 'pending')
        return f'<article><h3>{title}</h3><p>{html.escape(status)}</p></article>'

    cards = []
    for pair in [('female','pop'), ('country','indie-rock'), ('house','acoustic-folk')]:
        title = ' + '.join(pair)
        mixes = [j for j in spec['jobs'] if {s['id'] for s in j['sliders']} == set(pair)]
        random.Random('studio-mix-101-'+title).shuffle(mixes)
        controls = [j for j in spec['jobs'] if (j['label'] == 'Off'
                    or (len(j['sliders']) == 1 and j['sliders'][0]['id'] in pair)
                    or j['label'] == title+' caption reference')]
        cards.append(f'<section><h2>{html.escape(title)}</h2><div class="grid">'
                     + ''.join(player(j, chr(65+i)) for i,j in enumerate(mixes))+'</div>'
                     + '<p>Which keeps both characters, clear words, a coherent groove and a natural voice?</p>'
                     + '<details><summary>Reveal energy settings</summary><ul>'
                     + ''.join(f'<li>{chr(65+i)}: energy {j["energy"]["language_model"]:g}</li>'
                               for i,j in enumerate(mixes))+'</ul></details>'
                     + '<details><summary>Solo and caption references</summary><div class="grid">'
                     + ''.join(player(j) for j in controls)+'</div></details></section>')
    page = ('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Studio blend listening</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;background:#141a22;color:#e8edf4;font:16px/1.5 system-ui}a{color:#9ed3ff}section{margin:32px 0;padding:24px;border:1px solid #3a4455;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}audio{width:100%}h3{font-size:16px}summary{cursor:pointer;padding:10px 0}p{max-width:78ch}</style>
<h1>Studio blend listening</h1><p>Same lyric sheet, base caption and seed. Each mix uses equal shares.
Compare A, B and C before revealing their energy settings. The players contain the first 20 seconds;
full recordings are linked underneath. Set a comfortable playback volume.</p>'''
            + f'<p>{completed}/25 rendered; {failed} errors. Status: {html.escape(state["status"])}. '
            + 'Refresh for new clips.</p>'+''.join(cards)
            + '<details><summary>Lyric sheet</summary><pre>'+html.escape(spec['jobs'][0]['lyrics'])+'</pre></details>'
            + '<p>This is one fixture and seed. Preference here needs confirmation on another song.</p></html>')
    temporary = output/'index.html.tmp'
    temporary.write_text(page)
    temporary.replace(output/'index.html')
