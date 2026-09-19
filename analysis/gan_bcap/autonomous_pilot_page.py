"""Publish the saved pilot evidence without changing training or selection."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'analysis/gan_bcap/autonomous_campaign_20260905'
OUT = ROOT / 'eval/listen/gan-autonomous-20260905'


def label(path, steps):
    if 'lyrichold-w0p001-stepcap2' in path:
        recipe = 'Lyric hold + step limit'
    elif 'stepcap2-only' in path:
        recipe = 'Step limit only'
    else:
        recipe = 'Original updates'
    return f'{recipe} · {steps:,} updates'


def publish():
    state = json.loads((WORK / 'state.json').read_text())
    reports = [WORK / 'pilot-initial.json', *sorted(WORK.glob('pilot-[0-9]*.json'))]
    records, rankings = {}, {}
    for path in reports:
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report.get('status') != 'complete' or report['rule'] != state['rule']:
            continue
        for row in report['ranking']:
            rankings[row['checkpoint']] = row
        for row in report['records']:
            records[row['checkpoint']['path'], row['fixture']] = row
    data = []
    for key, row in sorted(rankings.items(), key=lambda p: -p[1]['heuristic_score']):
        clips = []
        for (checkpoint, fixture), record in records.items():
            if checkpoint != key:
                continue
            urls = {}
            for field in ['baseline', 'candidate', 'positive_reference']:
                audio = Path(record[field]['audio']).resolve()
                audio.relative_to(ROOT / 'eval/listen')
                urls[field] = os.path.relpath(audio, OUT)
            clips.append(dict(fixture=fixture, seed=record['seed'], score=record['heuristic_score'], group='Search',
                              urls=urls, transcript=record['candidate']['transcript'],
                              concept={field: record[field]['concept'] for field in
                                       ['baseline', 'candidate', 'positive_reference']}))
        data.append(dict(label=label(key, row['steps']), **row, clips=clips))
    confirmation_path = WORK / 'confirmation.json'
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text())
        if confirmation.get('status') == 'complete' and confirmation['rule'] == state['rule']:
            by_checkpoint = {r['checkpoint']: r for r in data}
            for record in confirmation['records']:
                row = by_checkpoint[record['checkpoint']['path']]
                urls = {}
                for field in ['baseline', 'candidate', 'positive_reference']:
                    audio = Path(record[field]['audio']).resolve()
                    audio.relative_to(ROOT / 'eval/listen')
                    urls[field] = os.path.relpath(audio, OUT)
                row['clips'].append(dict(fixture=record['fixture'], seed=record['seed'],
                                         score=record['heuristic_score'], group='Confirmation',
                                         urls=urls, transcript=record['candidate']['transcript'],
                                         concept={field: record[field]['concept'] for field in
                                                  ['baseline', 'candidate', 'positive_reference']}))
    payload = dict(page_revision=2, candidates=data, pilot=state['pilot'], rule=state['rule'],
                   suggestion=state.get('suggestion'), confirmation=state.get('confirmation'))
    encoded = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    version = hashlib.sha256(encoded.encode()).hexdigest()
    suggestion = state.get('suggestion')
    title = (label(suggestion['checkpoint'], suggestion['steps']) if suggestion
             else 'Following the training curve')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GAN training curve / Listen and inspect</title>
<style>*{box-sizing:border-box}body{background:#101115;color:#f2eee6;font:16px system-ui,sans-serif;margin:0}main{max-width:1200px;margin:auto;padding:45px 25px}a{color:#c0a6ff}h1{font-size:clamp(32px,5vw,56px);letter-spacing:-.035em}h2{font-size:24px}p,small{color:#aaaab4;line-height:1.6}.eyebrow{color:#c0a6ff;letter-spacing:.15em;font-size:12px;text-transform:uppercase}.panel{padding:25px;margin:25px 0;border:1px solid #34353b;background:#1b1c22;border-radius:18px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:25px}audio{width:100%;margin-top:15px}select{background:#30313a;color:#fff;border:1px solid #555;border-radius:8px;padding:12px;max-width:100%;margin:8px 12px 15px 0;font:inherit}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;border-bottom:1px solid #34353b;padding:14px 10px}th:first-child,td:first-child{text-align:left;min-width:210px}th{color:#aaaab4;font-weight:400}tbody tr:first-child{background:#27262f}svg{width:100%;height:220px}.scroll{overflow-x:auto}.score{color:#b7f4c3;font-weight:700}.detail{font-size:13px}button{background:none;border:none;color:#c0a6ff;text-align:left;font:inherit;cursor:pointer;padding:0}@media(max-width:650px){.pair{grid-template-columns:1fr}main{padding:25px 15px}.panel{padding:18px}}</style>
<main><a href="index.html">← All 28 controls</a><p class="eyebrow">Repaired GAN / saved evidence</p>'''
    page += f'<h1>{html.escape(title)}</h1>'
    if (OUT.parent / 'gan-convergence-20260905/index.html').exists():
        page += '<p><a href="../gan-convergence-20260905/index.html">Convergence criteria study and live evaluation progress</a></p>'
    if suggestion and state.get('confirmation', {}).get('status') != 'complete':
        page += '<p class="detail">Provisional suggestion; the additional-sample comparison is running.</p>'
    if state.get('confirmation', {}).get('status') == 'complete' and (OUT / 'pilot-winner.safetensors').exists():
        page += '<p><a href="pilot-winner.safetensors">Download the confirmed pilot adapter</a> · <a href="pilot-winner.json">Training and selection metadata</a></p>'
    page += '''<p>Every point below comes from saved audio. Choose a checkpoint and a matched sample to hear how training changes the sound.</p>
<details class="panel"><summary>What warmup, new-lyrics, long and auto-pilot mean</summary><p><strong>Warmup</strong> is the initial GAN training stage, up to 600 updates with the original recipe. The name remains on its 300/450/600-update checkpoints; it does not mean a quiet intro or a learning-rate ramp.</p><p><strong>New-lyrics</strong> uses saved weights with different supplied words and the same caption, with 20-second generation limits and seeds 7 and 23. <strong>Long</strong> uses the selected checkpoint with those new lyrics, seed 101 and a 60-second generation limit. Neither performs additional training; long is a separate generation, not an extension of the short clip.</p><p><strong>Auto-pilot</strong> contains the original Female-slider training-length comparison. A pilot-1350 folder contains both the 600-update reference and the 1,350-update candidate. The checkpoint subfolder identifies the weights.</p><p>In raw folders, <strong>01_slider_neutral_base_zero</strong> is slider off, <strong>02_slider_Female_plus1</strong> is Female slider on, and <strong>03/04_REF_prompt</strong> are caption-only references with the adapter off. The Male caption reference intentionally requests a masculine voice.</p></details>
<section class="panel"><h2>Step-limit training curve</h2><p class="detail">The shared 600-update starting point, followed by step limit only. Higher is better under this fixed heuristic.</p><svg id="curve" viewBox="0 0 1000 220" role="img" aria-label="Measured heuristic by training updates"></svg><p class="detail">Stop after three additional checkpoints fail to improve the running high by more than 0.05. This is an observed plateau on these samples, not a proven musical optimum.</p></section>
<section class="panel"><h2>Listen to the evidence</h2><select id="checkpoint" aria-label="Checkpoint"></select><select id="fixture" aria-label="Prompt and seed"></select><div class="pair"><div><strong>Slider off</strong><audio id="off" controls preload="none"></audio></div><div><strong id="on-label">Slider on</strong><audio id="on" controls preload="none"></audio></div></div><p class="detail" id="clip-score"></p><p class="detail" id="clip-concept"></p><p class="detail">The score rewards improvement relative to slider off. It can be positive even when the intended voice change is incomplete. The description-similarity margin is a diagnostic, not a reliable voice classifier; listening feedback takes precedence.</p><details><summary>Positive-caption reference and transcript diagnostic</summary><audio id="reference" controls preload="none"></audio><p id="transcript"></p><p class="detail">The transcript is automatic and can miss or invent words. It does not establish that a sample is broken.</p></details></section>
<section class="panel"><h2>What contributes to the score</h2><p class="detail">Each row averages the same four prompt/seed samples. These are weighted contributions, so the five component columns add to the total. All changes are relative to the matching slider-off audio. The weakest sample is shown separately because a mean can hide inconsistent results.</p><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>Total</th><th>Concept</th><th>Enjoyment</th><th>Production</th><th>Lyrics</th><th>Noise</th><th>Weakest sample</th></tr></thead><tbody id="scores"></tbody></table></div></section>
<section class="panel"><h2>How the suggestion is checked</h2><p>Concept match has weight 0.4. Predicted enjoyment, production quality and lyric preservation each have weight 0.2. Excess high-frequency noise adds a penalty. Perceptual measurements use separate copies at matched RMS; the listening audio is unchanged.</p><p>The two leading recipes are compared on four additional samples using new arrangements and seeds. The combined result supplies the catalog starting suggestion. Each control then compares its own saved checkpoints.</p><p class="detail">Changing volume by ±3 dB changed the corrected score by less than 0.0001 in the controlled check. Added hiss scored −0.161 and quarter-second repetition −0.829. These checks test particular weaknesses, not overall musical judgment.</p><p id="confirmation"></p></section>
<p class="detail">First renders are retained. The table includes earlier recipe comparisons; the curve isolates the step-limit continuation. This page updates when new evidence arrives and no audio is playing.</p></main>'''
    page += '<script>const data=' + encoded + ';const version=' + json.dumps(version) + ';'
    page += '''const select=document.querySelector('#checkpoint'), fixture=document.querySelector('#fixture');
const audio=[...document.querySelectorAll('audio')];audio.forEach(a=>a.addEventListener('play',()=>audio.forEach(b=>{if(a!==b)b.pause()})));
data.candidates.forEach((r,i)=>select.add(new Option(r.label,i)));if(data.suggestion){const winner=data.candidates.findIndex(r=>r.checkpoint===data.suggestion.checkpoint);if(winner>=0)select.value=winner;}
function chooseClip(){if(!data.candidates.length)return;const r=data.candidates[+select.value],c=r.clips[+fixture.value];audio.forEach(a=>a.pause());document.querySelector('#off').src=c.urls.baseline;document.querySelector('#on').src=c.urls.candidate;document.querySelector('#reference').src=c.urls.positive_reference;document.querySelector('#on-label').textContent='Female slider on (+1) · '+r.label;document.querySelector('#clip-score').textContent='This sample: '+c.score.toFixed(3)+' · search mean: '+r.heuristic_score.toFixed(3);document.querySelector('#clip-concept').textContent='Feminine minus masculine description similarity: off '+c.concept.baseline.toFixed(3)+' → on '+c.concept.candidate.toFixed(3)+'. Positive leans feminine; negative leans masculine.';document.querySelector('#transcript').textContent=c.transcript||'No words detected.';}
function chooseCheckpoint(){fixture.replaceChildren();const r=data.candidates[+select.value];if(!r)return;const counts={};r.clips.forEach((c,i)=>{const n=counts[c.group]||0;counts[c.group]=n+1;fixture.add(new Option(c.group+' arrangement '+(Math.floor(n/2)+1)+' · seed '+c.seed,i))});chooseClip();}
select.onchange=chooseCheckpoint;fixture.onchange=chooseClip;chooseCheckpoint();
const table=document.querySelector('#scores');data.candidates.forEach((r,i)=>{const tr=document.createElement('tr'),td=document.createElement('td'),b=document.createElement('button');b.textContent=r.label;b.onclick=()=>{select.value=i;chooseCheckpoint();select.scrollIntoView({behavior:'smooth',block:'center'})};td.append(b);tr.append(td);[r.heuristic_score,...Object.keys(data.rule.weights).map(k=>r.components[k]*data.rule.weights[k]),r.minimum_clip_score].forEach((v,j)=>{const d=document.createElement('td');d.textContent=v.toFixed(3);if(!j)d.className='score';tr.append(d)});table.append(tr)});
const svg=document.querySelector('#curve'),ns='http://www.w3.org/2000/svg';function mark(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e)}
if(data.pilot.length){const p=data.pilot,low=Math.min(...p.map(r=>r.score))-.12,high=Math.max(...p.map(r=>r.score))+.12,min=p[0].steps,max=p[p.length-1].steps;const x=r=>45+(r.steps-min)/Math.max(1,max-min)*910,y=r=>175-(r.score-low)/(high-low)*145;mark('polyline',{points:p.map(r=>x(r)+','+y(r)).join(' '),fill:'none',stroke:'#c0a6ff','stroke-width':3});p.forEach(r=>{mark('circle',{cx:x(r),cy:y(r),r:5,fill:'#c0a6ff'});mark('text',{x:x(r),y:y(r)-13,fill:'#f2eee6','text-anchor':'middle','font-size':13},r.score.toFixed(3));mark('text',{x:x(r),y:205,fill:'#aaaab4','text-anchor':'middle','font-size':13},String(r.steps))})}
const c=document.querySelector('#confirmation');if(data.confirmation?.status==='complete'){c.textContent='Confirmation complete. '+data.confirmation.ranking.map(r=>data.candidates.find(x=>x.checkpoint===r.checkpoint)?.label+': search '+r.search_score.toFixed(3)+', additional samples '+r.confirmation_score.toFixed(3)+', combined '+r.combined_score.toFixed(3)).join(' · ')}else c.textContent='Additional-sample confirmation is pending.';
setInterval(()=>fetch('pilot-version.json',{cache:'no-store'}).then(r=>r.json()).then(v=>{if(v.version!==version&&audio.every(a=>a.paused))location.reload()}).catch(()=>{}),20000);</script></html>'''
    OUT.mkdir(parents=True, exist_ok=True)
    for path, content in [(OUT / 'pilot.html', page),
                          (OUT / 'pilot-version.json', json.dumps(dict(version=version)))]:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        publish()
        if not args.watch:
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
