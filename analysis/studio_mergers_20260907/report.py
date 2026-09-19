"""Blind, reproducible listening order for the merger study."""
import html
import json
from pathlib import Path
import random
from common import WORK, PAIRS, write


def assignments(spec):
    groups = []
    for fixture in ('familiar','confirmation'):
        for pair in PAIRS:
            jobs = [j for j in spec['jobs'] if j['fixture']==fixture and j['pair']==list(pair)]
            mixes = [j for j in jobs if j['method']!='linear_e2']
            random.Random('studio-mergers-20260907-'+fixture+'-'.join(pair)).shuffle(mixes)
            groups.append(dict(fixture=fixture,pair=pair,mixes=mixes,
                               reference=next(j for j in jobs if j['method']=='linear_e2')))
    return groups


def build(state, output):
    spec = json.loads((WORK/'screen.json').read_text())
    records=state['records']
    completed=sum(r['status']=='complete' for r in records.values())
    failed=sum(r['status']=='error' for r in records.values())
    groups=assignments(spec)
    write(WORK/'blind-key.json',[dict(fixture=g['fixture'],pair=g['pair'],
          letters={chr(65+i):dict(id=j['id'],method=j['method']) for i,j in enumerate(g['mixes'])}) for g in groups])

    def player(job,label):
        rec=records.get(job['id'],{})
        if rec.get('status')!='complete':
            return f'<article><h3>{label}</h3><p>{html.escape(rec.get("status","pending"))}</p></article>'
        return (f'<article><h3>{label}</h3><audio controls preload="none" src="{Path(rec["excerpt"]).name}"></audio>'
                f'<a href="{Path(rec["audio"]).name}">Full audio ({rec["inspection"]["seconds"]:.1f}s)</a></article>')

    parts=[]
    labels=dict(linear='Ordinary mix, energy 2.8',ties='TIES',knots_ties='KnOTS + TIES')
    for fixture,title in [('familiar','Song 1 — familiar'),('confirmation','Song 2 — confirmation')]:
        parts.append(f'<h2 id="{fixture}">{title}</h2>')
        for g in groups:
            if g['fixture']!=fixture:continue
            parts.append('<section><h3>'+html.escape(' + '.join(g['pair']))+'</h3><div class="grid">'
                +''.join(player(j,chr(65+i)) for i,j in enumerate(g['mixes']))+'</div>'
                +'<p>Judge the blend of both controls, clear words, natural voice and coherent groove. Letters are shuffled separately in each group.</p>'
                +'<details><summary>Energy 2 reference</summary>'+player(g['reference'],'Ordinary mix, energy 2')+'</details>'
                +'<details><summary>Reveal methods</summary><ul>'
                +''.join(f'<li>{chr(65+i)}: {labels[j["method"]]}</li>' for i,j in enumerate(g['mixes']))
                +'</ul><p>TIES retains the largest 30% of entries per projection and resolves sign conflicts. KnOTS first aligns the updates in a shared SVD basis. Both outputs match the ordinary energy-2.8 update norm in every projection. Matching weight norms does not guarantee equal perceived strength.</p></details></section>')
        job=next(j for j in spec['jobs'] if j['fixture']==fixture)
        parts.append('<details><summary>Lyric sheet and base caption</summary><pre>'+html.escape(job['lyrics'])
                     +'</pre><p>'+html.escape(job['caption']).replace('\n','<br>')+'</p></details>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio merger comparison</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;background:#141a22;color:#e8edf4;font:16px/1.5 system-ui}a{color:#9ed3ff}section{margin:24px 0;padding:24px;border:1px solid #3a4455;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}audio{width:100%}h3{font-size:18px}summary{cursor:pointer;padding:10px 0}p{max-width:80ch}pre{white-space:pre-wrap}h2{margin-top:52px}</style>
<h1>Studio merger comparison</h1><p>Compare A, B and C in each group before revealing the methods. The first song matches the previous test; the second uses new lyrics, a new base caption and another seed. Within each group, every method receives the same inputs.</p>
<p>Players contain the first 20 seconds, with full recordings underneath. Your previous favorite, ordinary mixing at energy 2.8, is one of the three choices in every group.</p>
<p><a href="#familiar">Song 1</a> · <a href="#confirmation">Song 2</a> · <a href="/studio-mix-20260907/">Previous energy comparison</a></p>'''
    page+=f'<p>{completed}/{len(spec["jobs"])} rendered; {failed} errors. Status: {html.escape(state["status"])}. Refresh for new clips.</p>'
    page+=''.join(parts)+'<p>This screens one pruning setting on two songs. The studio defaults remain unchanged.</p></html>'
    temp=output/'index.html.tmp';temp.write_text(page);temp.replace(output/'index.html')
