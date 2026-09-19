"""Publish convergence evidence and live prospective-evaluation progress."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'analysis/gan_bcap/convergence_20260905'
OUT=ROOT/'eval/listen/gan-convergence-20260905'


def publish():
    labels={'early300':'Original 300','original600':'Original 600',
            'bounded1350':'Step limit 1350','failed900':'Historical failed 900',
            'original600-lr1':'Original 600 · full LR','bounded1350-lr1':'Step limit 1350 · full LR'}
    probes=[]
    for name,label in labels.items():
        p=WORK/name/'result.json'
        if p.exists():
            r=json.loads(p.read_text())
            probes.append(dict(name=name,label=label,generator=r['generator'],
                discriminator_gain=max(x['available_improvement'] for x in r['discriminator']),
                frozen_critic_exact=r['frozen_critic_exact'],source_state_unchanged=r['source_state_unchanged'],
                trace=[x['loss'] for x in r['generator_trace']]))
    p=WORK/'audio/status.json'
    status=json.loads(p.read_text()) if p.exists() else dict(status='pending',phase='Preparing prospective audio test')
    results=[json.loads(p.read_text()) for p in sorted((WORK/'audio').glob('comparison-*.json'))]
    payload=dict(probes=probes,audio=status,comparisons=results,
                 protocol=json.loads((WORK/'protocol.json').read_text()))
    encoded=json.dumps(payload,allow_nan=False).replace('<','\\u003c')
    version=hashlib.sha256(encoded.encode()).hexdigest()
    rows=''
    for r in probes:
        rows+=f'<tr><td>{html.escape(r["label"])}</td><td>{100*r["generator"]["fractional_improvement"]:.1f}%</td><td>{r["discriminator_gain"]:.6f}</td><td>{r["generator"]["initial"]:.3f} → {r["generator"]["best"]:.3f}</td></tr>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>When should GAN training stop?</title>
<style>*{box-sizing:border-box}body{background:#101115;color:#f2eee6;margin:0;font:16px system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:45px 25px}h1{font-size:clamp(35px,5vw,60px);letter-spacing:-.04em;line-height:1.08}h2{font-size:24px}p,li{line-height:1.6;color:#b8b8c2}a{color:#c0a6ff}.panel{padding:24px;margin:24px 0;border:1px solid #383943;border-radius:16px;background:#1b1c22}.eyebrow{color:#b7f4c3;letter-spacing:.14em;text-transform:uppercase;font-size:12px}.note{border-color:#9b784c}.detail{font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:13px 10px;border-bottom:1px solid #383943;text-align:right}td:first-child,th:first-child{text-align:left}th{color:#b8b8c2}select{background:#30313a;color:#fff;padding:10px;border:1px solid #666;border-radius:8px;font:inherit}svg{width:100%;height:240px}.status{color:#b7f4c3;font-size:20px}</style>
<main><a href="../gan-autonomous-20260905/index.html">← Slider batch · paused</a><p class="eyebrow">Convergence study / GPU 1</p><h1>When should<br>GAN training stop?</h1><p>Separate a settled game from a useful stopping point. Seven catalog controls are complete; the remaining catalog work is paused for this study.</p>
<section class="panel"><h2>What the frozen-opponent tests found</h2><p>The preferred 600 checkpoint can still improve substantially against a fixed critic. Its good listening result does not establish game equilibrium.</p><div class="scroll"><table><tr><th>Source checkpoint</th><th>G loss reduction</th><th>D available improvement</th><th>G initial → best loss</th></tr>'''+rows+'''</table></div><p class="detail">G: complete training objective, 20 measured updates, quarter learning rate unless labeled otherwise, parameter-step limit 2. D: best of four 60-update searches with fresh/restored moments and two learning rates. Each opponent is fixed. These are optimization diagnostics, not audio-quality scores.</p><select id="probe" aria-label="Probe trace"></select><svg id="curve" viewBox="0 0 1000 240" role="img" aria-label="Generator objective against a fixed critic"></svg><p class="detail">Trace shows loss divided by its initial value. Large reductions disprove the working stationarity tolerance. A small reduction can reflect a weak probe or a poorly conditioned failed model.</p></section>
<section class="panel note"><h2>A failed model can look quiet</h2><p>The historical failed 900 critic already separates fake and target spans almost perfectly. Its available improvement is near zero, while the generator loss remains very large. Small changes alone must never mark a slider ready.</p></section>
<section class="panel"><h2>The operational stopping rule</h2><ol><li>Stop immediately on a confirmed training failure; retain the previous usable checkpoint.</li><li>Measure a locked incumbent and later checkpoints on matched prompts and seeds. Average seeds within each prompt before estimating uncertainty.</li><li>Require three checkpoints whose upper gain bound is below a worthwhile improvement of 0.05. An inconclusive interval requests more evaluation.</li><li>Require separate validated concept and preservation checks. A plateau with failed checks is stalled. Passing the statistical test alone does not establish musical quality.</li><li>Confirm the selected checkpoint on reserved examples. Keep the best checkpoint; reaching a resource limit is a budget stop.</li></ol><p class="detail">The current paired interval uses prompt-level t assumptions and a predeclared comparison budget. In 20,000 simulations at the boundary, 4.95% had at least one false no-gain decision across 12 comparisons. The full stopping rule additionally requires three consecutive decisions. This validates the interval calculation under its assumptions, not the audio judge.</p></section>
<section class="panel"><h2>Prospective audio comparison</h2><p id="phase" class="status"></p><p>600 versus 1050, 1200 and 1350. Start with eight new arrangements and lyric sheets × four seeds, giving 32 matched samples per checkpoint. Add eight prompts if any comparison remains inconclusive. All first renders are kept.</p><div id="results"></div><p class="detail">The existing research score stays fixed. Absolute voice-description margins are reported separately and are not treated as a validated classifier. Catalog training remains paused when this study finishes.</p></section><p><a href="data.json">Complete measurements and protocol</a></p></main>'''
    page+='<script>const data='+encoded+';const version='+json.dumps(version)+';'
    page+='''const sel=document.querySelector('#probe'),svg=document.querySelector('#curve'),ns='http://www.w3.org/2000/svg';data.probes.forEach((p,i)=>sel.add(new Option(p.label,i)));function mark(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e)}function draw(){svg.replaceChildren();const p=data.probes[+sel.value];if(!p)return;const v=p.trace.map(x=>x/p.trace[0]),hi=Math.max(1,...v)*1.05;const x=i=>55+i*900/(v.length-1),y=a=>195-a/hi*160;mark('polyline',{points:v.map((a,i)=>x(i)+','+y(a)).join(' '),fill:'none',stroke:'#c0a6ff','stroke-width':3});[0,5,10,15,20].forEach(i=>mark('text',{x:x(i),y:225,fill:'#b8b8c2','text-anchor':'middle'},String(i)));[0,.5,1].forEach(a=>mark('text',{x:38,y:y(a)+5,fill:'#b8b8c2','text-anchor':'end'},a.toFixed(1)))}sel.onchange=draw;draw();document.querySelector('#phase').textContent=data.audio.phase||data.audio.error||data.audio.status;const results=document.querySelector('#results');data.comparisons.forEach(r=>{const p=document.createElement('p');p.textContent=r.prompt_groups+' prompts: '+r.comparisons.map(x=>x.checkpoint.match(/only-(\d+)-/)[1]+' updates: mean gain '+x.mean_gain.toFixed(3)+', upper bound '+(x.upper_gain_bound===null?'pending':x.upper_gain_bound.toFixed(3))+' · '+x.decision.replaceAll('_',' ')).join(' | ');results.append(p)});setInterval(()=>fetch('version.json',{cache:'no-store'}).then(r=>r.json()).then(v=>{if(v.version!==version)location.reload()}).catch(()=>{}),20000);</script></html>'''
    OUT.mkdir(parents=True,exist_ok=True)
    for name,content in [('index.html',page),('data.json',json.dumps(payload,indent=2)),('version.json',json.dumps(dict(version=version)))]:
        p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(content);t.replace(p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        publish()
        if not args.watch:break
        time.sleep(20)
