#!/usr/bin/env python3
"""Publish the retained, matched CE comparisons to the local listening server."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval/listen/ce-600-vs-660-20260910"
CONTROLS = [("lofi", "Lo-fi"), ("pop", "Pop"), ("afrobeats", "Afrobeats"), ("disco-funk", "Disco / Funk")]

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#121817">
<title>600 / 660 · Listen to the difference</title>
<style>
:root{color-scheme:dark;--bg:#121817;--panel:#1c2522;--line:#34443d;--text:#edf3ea;--muted:#b0bfb3;--accent:#c2ed9d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:40px 24px 80px}
a{color:var(--accent)}.eyebrow{color:var(--accent);font:12px/1.4 ui-monospace,monospace;letter-spacing:.13em;text-transform:uppercase}h1{font-size:clamp(32px,5vw,56px);line-height:1.1;letter-spacing:-.045em;margin:14px 0 18px}h2{font-size:28px;letter-spacing:-.025em;margin:0}h3{font-size:16px;margin:0}.intro{color:var(--muted);max-width:760px;margin:0 0 20px}.toolbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:22px 0}label{cursor:pointer}input{accent-color:var(--accent);width:18px;height:18px;vertical-align:middle;margin-right:8px}
button,.tab{font:inherit;cursor:pointer;min-height:44px;border:1px solid var(--line);border-radius:10px;color:var(--text);background:var(--panel);padding:9px 15px}button:hover,.tab:hover{border-color:var(--accent)}button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
.tabs{display:flex;gap:8px;overflow:auto;padding:4px 0 16px;position:sticky;top:0;background:var(--bg);z-index:2}.tab{white-space:nowrap}.tab[aria-pressed=true]{background:var(--accent);color:#16200f;border-color:var(--accent)}.section-head{margin:22px 0}.section-head p{color:var(--muted);margin:4px 0}
.pair{border:1px solid var(--line);border-radius:16px;background:var(--panel);padding:20px;margin:16px 0}.pair-head{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:18px}.seed{color:var(--muted);font-size:13px;margin:2px 0 0}.actions{display:flex;gap:8px;flex-wrap:wrap}.actions button{font-size:14px;background:var(--bg)}.players{display:grid;grid-template-columns:1fr 1fr;gap:16px}.take{padding:16px;border:1px solid var(--line);border-radius:12px;background:var(--bg);min-width:0}.take.playing{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.take-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}.step{font-size:22px;font-weight:650}.role,.small{font-size:13px;color:var(--muted)}audio{width:100%;display:block;margin:12px 0}.download{font-size:13px}.ce{color:var(--accent);font:14px/1.5 ui-monospace,monospace}.pair-score{margin:12px 0 0}.error{color:#ffb6a6}body:not(.show-scores) .score{display:none!important}[hidden]{display:none!important}footer{border-top:1px solid var(--line);margin-top:30px;padding-top:18px;color:var(--muted);font-size:13px}.notice{padding:14px 16px;border-left:3px solid var(--accent);background:var(--panel);color:var(--muted)}
@media(max-width:620px){main{padding:24px 14px 50px}.players{grid-template-columns:1fr}.pair{padding:14px}.take{padding:12px}.pair-head{align-items:flex-start}.actions{width:100%}.actions button{flex:1;font-size:13px;padding:8px}.toolbar{gap:12px}.tab{font-size:14px}}
</style>
</head>
<body><main>
<div class="eyebrow">Music 3 · checkpoint comparisons</div>
<h1>Listen to the difference.</h1>
<p class="intro">Step <strong>600</strong> is the unpublished alternative. Step <strong>660</strong> is the published version. Each pair uses the same caption, lyrics, seed, and slider strength. These are the original 20-second renders; the arrangements can change.</p>
<div class="toolbar"><label><input type="checkbox" id="scores">Show enjoyment scores</label><button id="stop" type="button">Stop playback</button><span class="small">4 sounds · 16 pairs · original levels</span></div>
<nav class="tabs" aria-label="Choose a sound" id="tabs"></nav>
<div id="content" aria-live="polite"><p>Loading comparisons…</p></div>
<footer>All four measured pairs per sound are included, including examples where 660 scores higher. Content Enjoyment (CE) is a model prediction, not a listening verdict. Scores come from the WAV recordings; the players use MP3 previews without loudness normalization. <a href="manifest.json">Recording details</a></footer>
</main>
<script>
'use strict';
let sequence = null;
let playbackEpoch = 0;
const previews = new Map();
const allAudio = () => [...document.querySelectorAll('audio')];
const status = document.getElementById('content');
function stopAll(reset=false){playbackEpoch++;sequence=null;for(const a of allAudio()){a.pause();if(reset)a.currentTime=0;}}
function report(error, card){const node=card.querySelector('.error');node.textContent='Playback could not start. Try the audio controls or download the WAV.';node.hidden=false;console.error(error);}
function prepare(audio){
 if(audio.ready)return audio.ready;
 const src=audio.dataset.src;
 if(!previews.has(src))previews.set(src,fetch(src).then(r=>{if(!r.ok)throw Error(r.status);return r.blob()}).then(blob=>URL.createObjectURL(blob)));
 audio.ready=previews.get(src).then(url=>new Promise((resolve,reject)=>{audio.addEventListener('loadedmetadata',()=>{audio.closest('.take').querySelector('[data-loading]').hidden=true;resolve()},{once:true});audio.addEventListener('error',()=>reject(audio.error),{once:true});audio.src=url;audio.load();}));
 return audio.ready;
}
async function play(audio,card,at=null){const epoch=playbackEpoch;try{await prepare(audio);if(epoch!==playbackEpoch||!audio.isConnected)return;if(at!==null)audio.currentTime=Math.min(at,Math.max(0,audio.duration-.05));await audio.play();}catch(e){sequence=null;report(e,card);}}
function initPair(card){
 const audios=[...card.querySelectorAll('audio')];
 for(const a of audios){
  a.addEventListener('play',()=>{for(const b of allAudio())if(b!==a)b.pause();for(const take of document.querySelectorAll('.take'))take.classList.remove('playing');a.closest('.take').classList.add('playing');if(sequence&&a!==sequence[0]&&a!==sequence[1])sequence=null;});
  a.addEventListener('pause',()=>a.closest('.take').classList.remove('playing'));
  a.addEventListener('error',()=>report(a.error,card));
  prepare(a).catch(e=>report(e,card));
  a.addEventListener('ended',()=>{a.closest('.take').classList.remove('playing');if(sequence&&sequence[0]===a){const next=sequence[1];sequence=null;next.currentTime=0;play(next,card);}});
 }
 card.querySelector('[data-sequence]').onclick=()=>{stopAll(true);sequence=audios;play(audios[0],card);};
 card.querySelector('[data-switch]').onclick=()=>{const active=audios.find(a=>!a.paused)||audios[0];const other=audios.find(a=>a!==active);const at=active.currentTime;stopAll();play(other,card,at);};
}
function fmt(n){return n.toFixed(3)}
function signed(n){return(n>=0?'+':'')+fmt(n)}
function render(data,id){
 stopAll();const control=data.controls.find(x=>x.id===id)||data.controls[0];
 for(const b of document.querySelectorAll('.tab'))b.setAttribute('aria-pressed',String(b.dataset.id===control.id));
 const average=control.mean_ce;
 status.innerHTML=`<div class="section-head"><h2>${control.label}</h2><p>Two arrangements × two seeds. Listen to both takes, then switch examples.</p><p class="score ce">Mean CE · 600: ${fmt(average['600'])} · 660: ${fmt(average['660'])} · difference ${signed(average['600']-average['660'])}</p></div>`+
 control.pairs.map((pair,i)=>`<article class="pair" id="${pair.id}"><div class="pair-head"><div><h3>Arrangement ${pair.arrangement}</h3><p class="seed">Seed ${pair.seed} · pair ${i+1} of 4</p></div><div class="actions"><button type="button" data-sequence>Play 600 → 660</button><button type="button" data-switch>Switch at same time</button></div></div><div class="players">${pair.takes.map(t=>`<div class="take"><div class="take-head"><span class="step">${t.steps}</span><span class="role">${t.steps===600?'Unpublished alternative':'Published version'}</span></div><audio controls preload="metadata" aria-label="${control.label}, arrangement ${pair.arrangement}, seed ${pair.seed}, step ${t.steps}" data-src="${t.mp3}"></audio><span class="small" data-loading>Loading preview…</span><div class="take-head"><a class="download" href="${t.wav}" download>Original WAV ↓</a><span class="score ce">CE ${fmt(t.ce)}</span></div></div>`).join('')}</div><p class="score pair-score ce">CE difference (600 − 660): ${signed(pair.takes[0].ce-pair.takes[1].ce)}</p><p class="error" hidden></p></article>`).join('');
 for(const card of document.querySelectorAll('.pair'))initPair(card);
}
document.getElementById('scores').onchange=e=>document.body.classList.toggle('show-scores',e.target.checked);
document.getElementById('stop').onclick=()=>stopAll();
fetch('manifest.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(data=>{
 const tabs=document.getElementById('tabs');for(const control of data.controls){const b=document.createElement('button');b.type='button';b.className='tab';b.dataset.id=control.id;b.textContent=control.label;b.onclick=()=>{location.hash=control.id;};tabs.append(b);}
 const update=()=>render(data,location.hash.slice(1));window.addEventListener('hashchange',update);update();
}).catch(e=>{status.innerHTML='<p class="error">Could not load the comparisons. Reload the page to try again.</p>';console.error(e)});
</script></body></html>'''


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def make_audio(job):
    source, wav, mp3, expected = job
    assert digest(source) == expected, f"Source hash changed: {source}"
    if not wav.exists():
        os.link(source, wav)
    assert digest(wav) == expected
    if not mp3.exists():
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(wav),
            "-map_metadata", "-1", "-codec:a", "libmp3lame", "-b:a", "192k",
            "-threads", "1", str(mp3),
        ], check=True)
    return {"wav": wav.name, "wav_sha256": expected, "mp3": mp3.name, "mp3_sha256": digest(mp3)}


def main():
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "metric": "Content Enjoyment", "preview": "MP3 192 kbps; no normalization", "controls": []}
    jobs = []
    for concept, label in CONTROLS:
        report = ROOT / f"analysis/uni16_20260906/scores/{concept}.json"
        records = json.loads(report.read_text())["records"]
        groups = {}
        for record in records:
            assert record["eligible"]
            key = (record["fixture"], record["seed"])
            steps = record["checkpoint"]["steps"]
            assert steps not in groups.setdefault(key, {}), f"Duplicate {key} / {steps}"
            groups[key][steps] = record
        assert len(groups) == 4
        ordered = sorted(groups.items(), key=lambda item: (Path(item[1][600]["candidate"]["audio"]).parts[-3], item[0][1]))
        control = {"id": concept, "label": label, "score_source": str(report.relative_to(ROOT)), "score_sha256": digest(report), "pairs": []}
        for (fixture, seed), takes in ordered:
            assert set(takes) == {600, 660}
            assert takes[600]["baseline"]["sha256"] == takes[660]["baseline"]["sha256"]
            row = int(Path(takes[600]["candidate"]["audio"]).parts[-3].split("-")[1])
            pair = {"id": f"{concept}-row{row}-s{seed}", "arrangement": row + 1, "fixture": fixture, "seed": seed, "takes": []}
            for steps in (600, 660):
                record = takes[steps]
                candidate = record["candidate"]
                source = Path(candidate["audio"])
                stem = f"{concept}-row{row}-s{seed}-step{steps}"
                wav, mp3 = OUT / "audio" / f"{stem}.wav", OUT / "audio" / f"{stem}.mp3"
                pair["takes"].append({"steps": steps, "ce": candidate["enjoyment"], "duration": candidate["duration"], "wav": f"audio/{wav.name}", "mp3": f"audio/{mp3.name}", "source": str(source.relative_to(ROOT)), "source_sha256": candidate["sha256"], "checkpoint_sha256": record["checkpoint"]["sha256"]})
                jobs.append((source, wav, mp3, candidate["sha256"]))
            control["pairs"].append(pair)
        control["mean_ce"] = {str(steps): statistics.mean(p["takes"][i]["ce"] for p in control["pairs"]) for i, steps in enumerate((600, 660))}
        manifest["controls"].append(control)
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest["assets"] = list(pool.map(make_audio, jobs))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "index.html").write_text(HTML)
    print(f"Built {OUT}: {len(jobs)} recordings, 16 matched pairs")


if __name__ == "__main__":
    main()
