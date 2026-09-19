"""Publish matched controls and finish the audit without changing the locked study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'analysis/gan_bcap/v2_20260905'
PAGE = ROOT / 'eval/listen/gan-v2-20260905'
LABELS = {'original': 'Original 600', 'baseline': 'Bounded baseline 660',
          'fm_normalized': 'Normalized FM 660', 'fm_capped': 'FM gradient limit 660',
          'decay': 'LR decay 660', 'repaired': 'Combined repairs 660',
          'ema': 'Combined repairs · EMA 660'}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def relative(path):
    return str(path.relative_to(PAGE)) if path and path.exists() else None


def one(folder, pattern):
    return next(folder.glob(pattern), None)


def collect():
    import yaml
    state = read(WORK / 'study-status.json', {})
    gpu = read(WORK / 'gpu-wait.json', {})
    manifest = read(WORK / 'study-manifest.json', {})
    training = read(WORK / 'training-validation.json', {})
    audit = read(WORK / 'final-audit.json', {})
    if audit.get('status') != 'complete':
        audit = {}
    evaluation = yaml.safe_load((WORK / 'fixtures/evaluation.yaml').read_text())
    if isinstance(evaluation, dict):
        evaluation = evaluation['rows']
    checkpoints = []
    source = Path(manifest['source']['path'])
    checkpoints.append(dict(id='original', label=LABELS['original'],
                            path=str(source.with_name(source.stem.removesuffix('_state') + '_last.safetensors'))))
    for arm in manifest['arms']:
        run = state.get('runs', {}).get(arm, {})
        if run.get('checkpoints', {}).get('live'):
            checkpoints.append(dict(id=arm, label=LABELS[arm], path=run['checkpoints']['live']))
    ema = state.get('runs', {}).get('repaired', {}).get('checkpoints', {}).get('ema')
    if ema:
        checkpoints.append(dict(id='ema', label=LABELS['ema'], path=ema))
    scores = []
    score_files = []
    for row in manifest['evaluation_rows']:
        path = WORK / f'scores-v2-{row:02d}.json'
        score = read(path, {})
        if score.get('status') == 'complete':
            scores.extend(dict(record, prompt_row=row) for record in score['records'])
            score_files.append(dict(path=str(path), sha256=sha(path)))
    by_fixture = {(r['checkpoint']['path'], r['fixture']): r for r in scores}
    original = {r['fixture']: r for r in scores if r['checkpoint']['path'] == checkpoints[0]['path']}
    prompts = []
    completed = 0
    for row in manifest['evaluation_rows']:
        folder = PAGE / f'prompt-{row:02d}'
        prompt = dict(row=row, label=f'Reserved prompt {row + 1}',
                      caption=evaluation[row]['neutral'], lyrics=evaluation[row]['lyrics'], seeds=[])
        for seed in manifest['seeds']:
            first = folder / f'{Path(checkpoints[0]["path"]).stem}-s{seed}'
            item = dict(seed=seed, off=relative(first / '01_slider_neutral_base_zero.wav'),
                        positive=relative(one(first, '03_REF_prompt_*_no_slider.wav')), candidates={})
            reference = next((r for r in scores if r['prompt_row'] == row and r['seed'] == seed), None)
            if reference:
                item['off_metrics'] = clip_metrics(reference['baseline'])
                item['positive_metrics'] = clip_metrics(reference['positive_reference'])
            for checkpoint in checkpoints:
                sub = folder / f'{Path(checkpoint["path"]).stem}-s{seed}'
                clip = one(sub, '02_slider_*_plus1.wav') if (sub / 'checkpoint.json').exists() else None
                item['candidates'][checkpoint['id']] = relative(clip)
                completed += bool(clip)
            prompt['seeds'].append(item)
        prompts.append(prompt)
    summaries = []
    for checkpoint in checkpoints:
        rows = [r for r in scores if r['checkpoint']['path'] == checkpoint['path']]
        measured = audit.get('candidates', {}).get(checkpoint['path'])
        summary = dict(**checkpoint, examples=len(rows), training=training.get(checkpoint['id']))
        if rows:
            summary['clip_metrics'] = {f'{r["prompt_row"]}:{r["seed"]}': clip_metrics(r['candidate']) for r in rows}
            summary.update(score=statistics.mean(r['heuristic_score'] for r in rows),
                components={k: statistics.mean(r['components'][k] for r in rows) for k in rows[0]['components']},
                absolute_description_margin=statistics.mean(r['candidate']['concept'] for r in rows),
                lyric_proxy=statistics.mean(r['candidate']['lyrics'] for r in rows))
            if all(r['fixture'] in original for r in rows):
                summary['paired_delta_600'] = statistics.mean(
                    r['heuristic_score'] - original[r['fixture']]['heuristic_score'] for r in rows)
        if measured:
            summary.update(quality=measured['quality'], mean_score=measured['mean_score'],
                worst_score=measured['worst_score'], score_quantile_10=measured['score_quantile_10'],
                flagged_clips=sum(bool(r['diagnostics']['failures']) for r in measured['clips']),
                flags=[dict(seed=r['seed'], prompt=r['prompt'], failures=r['diagnostics']['failures'])
                       for r in measured['clips'] if r['diagnostics']['failures']],
                consensus_collapse_prompts=sum(d['status'] == 'collapse_suspected' for d in measured['diversity'].values()),
                single_view_alarm_prompts=sum(d['view_disagreement'] for d in measured['diversity'].values()),
                prompt_score_deltas={f'prompt-{by_fixture[(checkpoint["path"], next(r["fixture"] for r in measured["clips"] if r["prompt"] == p))]["prompt_row"] + 1}': statistics.mean(
                    by_fixture[(checkpoint['path'], r['fixture'])]['heuristic_score'] - original[r['fixture']]['heuristic_score']
                    for r in measured['clips'] if r['prompt'] == p)
                    for p in measured['diversity']})
        summaries.append(summary)
    phase = state.get('phase', 'Preparing')
    if state.get('status') == 'running' and gpu.get('status') == 'waiting':
        phase = f'Waiting for GPU {gpu.get("gpu", "1")} capacity; completed audio and measurements are retained.'
    return dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        phase=phase, final_audit_complete=bool(audit),
        candidate_clips_complete=completed,
        candidate_clips_expected=len(checkpoints) * len(prompts) * len(manifest['seeds']),
        prompts=prompts, candidates=summaries, scores=score_files,
        scope='Two reserved prompts × four matched seeds. A screening experiment; no convergence or musical-quality certificate.',
        score_description='Fixed heuristic relative to slider-off; combines description gain, aesthetics, lyric ASR and high-frequency excess. Higher is a proxy preference, not validated musical quality.',
        all_first_samples_retained=True, catalog='paused',
        audit_revision=audit.get('audit_revision'),
        report_source_sha256=sha(Path(__file__)))


def clip_metrics(row):
    lyric = row['lyric_diagnostics']
    return dict(duration=row['duration'], rms=row['rms'], description_margin=row['concept'],
                lyric_word_match=lyric['precision'], lyric_phrase_match=lyric['phrase_accuracy'],
                lyric_coverage=lyric['recall'])


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GAN repair comparison</title>
<style>body{margin:0;background:#141922;color:#e8edf5;font:16px/1.55 system-ui}main{max-width:1080px;margin:auto;padding:30px 22px 80px}h1{font-size:32px;margin-bottom:8px}h2{font-size:22px}p{max-width:850px}.muted{color:#a8b5c8}a{color:#9ccfff}select,button{font:inherit;background:#243044;color:inherit;border:1px solid #63748a;border-radius:6px;padding:8px;margin:4px}label{display:inline-block;margin-right:20px}article,details{background:#1c2533;border:1px solid #3e4c60;border-radius:8px;padding:16px;margin:12px 0}article h3{margin:0 0 8px;font-size:17px}audio{width:100%;height:42px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}.controls article{border-color:#668da4}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #465366;padding:9px;text-align:left}.scroll{overflow-x:auto}pre{white-space:pre-wrap;font:inherit;font-size:14px}.badge{display:inline-block;background:#2a3b4d;border-radius:4px;padding:2px 8px;font-size:13px} @media(max-width:650px){.controls{grid-template-columns:1fr}main{padding:16px}h1{font-size:26px}}</style>
<main><p class="muted">Music GAN · architecture repairs</p><h1>Compare the repairs</h1>
<p id="phase"></p><p class="muted">Every candidate uses the same prompt, lyrics, seed and 20-second duration. All first samples are retained. The combined recipe also changes the training data and critic; the other three interventions each change one mechanism from the bounded baseline.</p>
<p class="muted">The user reassigned this work to GPU 0. Prompt 1 is a complete matched group from GPU 1; prompt 2 runs entirely on GPU 0. The partial earlier attempt is <a href="gpu1-before-reassignment/prompt-01/">retained here</a>.</p>
<p><span class="badge" id="progress"></span> <span class="badge">Catalog paused</span></p>
<div><label>Prompt <select id="prompt"></select></label><label>Seed <select id="seed"></select></label><button id="refresh">Refresh results</button></div>
<details><summary>Prompt and lyrics</summary><p id="caption"></p><pre id="lyrics"></pre></details>
<div class="controls" id="controls"></div><div id="clips"></div>
<h2>Screening measurements</h2><p id="scope" class="muted"></p><div id="scores" class="scroll"></div>
<p class="muted">“Flags” counts samples triggering a technical or lyric-proxy screen. ASR word/phrase match and lyric-sheet coverage are shown separately under each measured clip: fewer words in 20 seconds can lower coverage without making the sung words wrong. “Diversity alarms” counts prompts flagged by both frozen representations. Zero alarms does not prove that all musical modes are preserved. Description similarity is not a calibrated voice classifier.</p>
<details><summary>Training checks and measurement details</summary><p>All five arms completed 60 updates from 600. The two scheduled arms reached their declared learning-rate floor. The effective-weight limit activated in 51 of 60 combined-repair updates; the separate FM gradient limit activated in 21 of 60 updates.</p><p id="scoreDescription"></p><pre id="technical"></pre></details>
<p><a href="comparison-data.json">Comparison data</a> · <a href="index.html">Run details</a> · <a href="../gan-convergence-20260905/index.html">Earlier convergence study</a></p></main>
<script>
let data;
const $=id=>document.getElementById(id), number=x=>typeof x==='number'?x.toFixed(3):'Pending';
function node(tag,text,parent){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(parent)parent.appendChild(x);return x}
function card(title,url,parent,m){const box=node('article',undefined,parent);node('h3',title,box);if(url){const audio=node('audio',undefined,box);audio.controls=true;audio.preload='none';audio.src=url;audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()}))}else node('p','Rendering pending',box);if(m){const percent=x=>(100*x).toFixed(0)+'%';node('p',`${m.duration.toFixed(1)}s · ASR word match ${percent(m.lyric_word_match)} · phrase match ${percent(m.lyric_phrase_match)} · sheet coverage ${percent(m.lyric_coverage)}`,box).className='muted'}}
function clips(){const p=data.prompts[+$('prompt').value],s=p.seeds.find(s=>String(s.seed)===$('seed').value);$('caption').textContent=p.caption;$('lyrics').textContent=p.lyrics;$('controls').replaceChildren();$('clips').replaceChildren();card('Slider off',s.off,$('controls'),s.off_metrics);card('Positive-caption reference · no slider',s.positive,$('controls'),s.positive_metrics);data.candidates.forEach(c=>card(c.label,s.candidates[c.id],$('clips'),c.clip_metrics?.[`${p.row}:${s.seed}`]))}
function seeds(){const old=$('seed').value;$('seed').replaceChildren();data.prompts[+$('prompt').value].seeds.forEach(s=>{const o=node('option',String(s.seed),$('seed'));o.value=s.seed});if([...$('seed').options].some(o=>o.value===old))$('seed').value=old;clips()}
function table(){const host=$('scores');host.replaceChildren();const t=node('table',undefined,host),head=node('tr',undefined,node('thead',undefined,t));['Candidate','Proxy score','Δ vs 600','Worst clip','Flags','Diversity alarms','Decision'].forEach(k=>node('th',k,head));const body=node('tbody',undefined,t);data.candidates.forEach(c=>{const r=node('tr',undefined,body);[c.label,number(c.score),number(c.paired_delta_600),number(c.worst_score),c.flagged_clips===undefined?'Pending':`${c.flagged_clips}/${c.examples}`,c.consensus_collapse_prompts===undefined?'Pending':`${c.consensus_collapse_prompts}/${data.prompts.length}`,c.quality?c.quality.decision.replaceAll('_',' '):'Pending'].forEach(v=>node('td',v,r))})}
async function refresh(){const old=$('prompt').value;data=await(await fetch('comparison-data.json',{cache:'no-store'})).json();$('phase').textContent=data.final_audit_complete?'Matched screening and final audit complete.':data.phase;$('progress').textContent=`${data.candidate_clips_complete}/${data.candidate_clips_expected} candidate clips ready`;$('scope').textContent=data.scope;$('scoreDescription').textContent=data.score_description;$('technical').textContent=data.candidates.map(c=>c.label+'\n'+JSON.stringify({training:c.training,components:c.components,description_margin:c.absolute_description_margin,lyric_proxy:c.lyric_proxy,prompt_score_deltas:c.prompt_score_deltas,flags:c.flags,quality:c.quality},null,2)).join('\n\n');$('prompt').replaceChildren();data.prompts.forEach((p,i)=>{const o=node('option',p.label,$('prompt'));o.value=i});if([...$('prompt').options].some(o=>o.value===old))$('prompt').value=old;seeds();table()}
$('prompt').addEventListener('change',seeds);$('seed').addEventListener('change',clips);$('refresh').addEventListener('click',refresh);refresh().catch(e=>$('phase').textContent='Results temporarily unavailable: '+e.message);
</script></html>'''


def publish():
    result = collect()
    write(PAGE / 'comparison-data.json', result)
    (PAGE / 'compare.html').write_text(HTML)
    if result['final_audit_complete']:
        write(WORK / 'final-results.json', result)
        write(PAGE / 'final-audit.json', read(WORK / 'final-audit.json'))
        lines = ['# Matched audio results for the GAN repairs', '', result['scope'], '',
                 'All five arms completed 60 updates from the original 600 state. '
                 'All 300 updates passed the finite-gradient and declared-update-bound checks. '
                 'The main regression suite has 124 passes, with three additional audio-audit tests.', '',
                 '| Candidate | Proxy score | Paired delta vs 600 | Worst clip | Flagged clips | Consensus diversity alarms |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for c in result['candidates']:
            lines.append(f'| {c["label"]} | {c["score"]:.4f} | {c["paired_delta_600"]:+.4f} | '
                         f'{c["worst_score"]:.4f} | {c["flagged_clips"]}/{c["examples"]} | '
                         f'{c["consensus_collapse_prompts"]}/{len(result["prompts"])} |')
        lines.extend(['', 'The proxy score is the unchanged, predeclared heuristic relative to slider-off. '
            'A higher number does not certify better music. Flags are technical/ASR checks; '
            'word matching and lyric coverage are exposed separately because slower phrasing can lower coverage. '
            'The diversity checks measure near-duplicates in two representations, not known musical modes.', '',
            'No automatic catalog promotion or convergence claim follows from two prompts. '
            'The combined arm changes several safeguards, the critic and the training data together; '
            'it cannot isolate the benefit of each component. EMA still has 74% nominal initial-600 mass before compression.', '',
            '[Listen with matched controls](../../../eval/listen/gan-v2-20260905/compare.html) · '
            '[Full measurements](final-audit.json) · [Implementation and validation](README.md)', ''])
        (WORK / 'results.md').write_text('\n'.join(lines))
    return result


def finish():
    manifest = read(WORK / 'study-manifest.json')
    scores = [WORK / f'scores-v2-{row:02d}.json' for row in manifest['evaluation_rows']]
    while True:
        result = publish()
        if result['final_audit_complete']:
            print('Final audit and comparison page complete', flush=True)
            return
        # The initial screen owns the same embedding cache. Wait for it to
        # finish before the refinement reuses that cache.
        if (read(WORK / 'screening.json', {}).get('status') == 'complete'
                and all(read(path, {}).get('status') == 'complete' for path in scores)):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       HF_HOME='/ml2/music/.cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with (WORK / 'final-audit.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'analysis/gan_bcap/quality_audit_v2.py',
                    '--scores', *map(str, scores), '--output', str(WORK / 'final-audit.json')],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            publish()
            print('Final audit and comparison page complete', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finish', action='store_true', help='Keep the page current and run the final audit once both score files exist')
    args = parser.parse_args()
    finish() if args.finish else publish()
