"""Prospective paired audio stopping study; the catalog stays paused.

Lock the incumbent and comparison candidates before collecting new audio.
Use eight new prompts and four seeds; add eight prompts if a gain interval
remains inconclusive. All first renders are retained. No metric is refitted.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.gan_bcap.autonomous_audio import RULE, summarize
from analysis.gan_bcap.convergence_criteria import paired_gain_bound, stopping_decision
from analysis.gan_bcap.lm_evaluate import vocal_details

WORK = ROOT/'analysis/gan_bcap/convergence_20260905/audio'
SEEDS = [7, 23, 101, 303]
CONTEXTS = [
    (96, 'Nylon guitar, warm fingered bass, cross-stick snare and a dry close room.',
     'A folded map lies on the table\nThe morning train is running slow\nWe tie a ribbon to the handle\nAnd take the path beside the snow'),
    (124, 'Arpeggiated synth, deep rounded bass, tight kick and narrow bright hi-hats.',
     'The ceiling fan turns little circles\nA wooden chair is painted green\nWe draw a line across the paper\nAnd leave a little space between'),
    (72, 'Electric piano, double bass and a brushed kit with a short warm plate reverb.',
     'The pantry door is standing open\nA copper pan reflects the light\nWe put the cups beside the window\nAnd fold the cloth before the night'),
    (148, 'Crunchy electric guitar, picked bass, driving snare and short ringing toms.',
     'A little stone rolls down the hillside\nThe empty basket catches rain\nWe lift the latch above the doorway\nAnd start the wooden wheel again'),
    (104, 'Accordion chords, plucked strings, hand drums and a small bright room.',
     'The market clock has lost a minute\nA yellow scarf hangs on the rail\nWe keep a button in the pocket\nAnd watch the boats beyond the sail'),
    (82, 'Sustained organ, tremolo guitar, soft bass and a loose pocket drum kit.',
     'A silver key rests on the cushion\nThe ladder leans against the wall\nWe hear the water in the garden\nAnd leave the shoes inside the hall'),
    (118, 'Muted brass, syncopated clavinet, round bass and a dry dance groove.',
     'A red balloon is in the doorway\nThe afternoon is warm and clear\nWe stack the plates beside the cupboard\nAnd keep the smallest saucer here'),
    (60, 'Fingerpicked acoustic guitar, low bowed strings and gentle mallet percussion.',
     'A woolen coat hangs by the mirror\nThe evening settles on the stair\nWe place a branch inside the bottle\nAnd turn the little wooden bear'),
    (132, 'Short marimba notes, pulsing synth bass, clipped kick and crisp shaker.',
     'The bicycle rests near the fountain\nA paper ticket slips away\nWe carry apples to the station\nAnd count the windows on the way'),
    (90, 'Resonant upright piano, muted bass, rim clicks and soft room reflections.',
     'A teaspoon rings against the glasses\nThe kettle hums beside the door\nWe brush the crumbs into a corner\nAnd sweep the sunlight from the floor'),
    (110, 'Wah guitar, fingered bass, tight snare and gently syncopated congas.',
     'A little boat is on the mantel\nThe window opens to the square\nWe wind the string around a pencil\nAnd move the table over there'),
    (68, 'Warm harmonium, sparse plucked bass, soft frame drum and intimate room tone.',
     'The lantern waits beside the river\nA pebble rests inside my hand\nWe fold the blanket by the doorway\nAnd trace a circle in the sand'),
    (140, 'Dry strummed electric guitar, octave bass, a lively kit and short cymbal accents.',
     'The painted gate is slightly crooked\nA little bell hangs on a chain\nWe bring the ladder from the kitchen\nAnd fix the handle in the rain'),
    (100, 'Plucked low strings, bell-like keys, restrained drums and a clear spacious mix.',
     'A blue bowl catches all the buttons\nThe empty drawer is smooth and wide\nWe leave a letter on the counter\nAnd push the wooden chair aside'),
    (76, 'Gentle banjo picking, bowed bass, brushes and a warm narrow room.',
     'The garden hose curls by the doorway\nA tiny leaf is on the chair\nWe tie the curtains with a ribbon\nAnd let the morning fill the air'),
    (126, 'Clipped string stabs, rounded synth bass, firm handclaps and a clean steady kick.',
     'The amber lamp lights up the hallway\nA pair of gloves lies on the shelf\nWe turn the pages by the window\nAnd let the clock explain itself'),
]


def write(path, value):
    path = Path(path)
    p = path.with_suffix(path.suffix+'.tmp')
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');p.replace(path)


def setup():
    import yaml
    sys.path.insert(0, str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    WORK.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((ROOT/'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml').read_text())
    original = (raw['rows'] if isinstance(raw, dict) else raw)[0]
    fixtures = []
    for i, (bpm, arrangement, words) in enumerate(CONTEXTS):
        def caption(vocal):
            return f'Global Metadata:\nBPM {bpm}. A complete song with clear lead vocals.\nVocal Details:\n{vocal}\nArrangement:\n{arrangement}'
        neutral = caption(vocal_details(original.get('neutral') or original['target']))
        positive = caption(vocal_details(original['positive']))
        lines = words.splitlines()
        row = dict(target=neutral, neutral=neutral, positive=positive, negative=neutral,
                   lyrics='[verse]\n'+'\n'.join(lines[:2])+'\n[chorus]\n'+'\n'.join(lines[2:]))
        if _artist_name_hit('', json.dumps(row)):
            raise ValueError('Prompt name validation failed')
        p = WORK/f'prompt-{i:02d}.yaml'
        content = yaml.safe_dump([row], sort_keys=False, allow_unicode=True)
        if p.exists() and p.read_text()!=content:
            raise ValueError('Locked prompt changed')
        p.write_text(content)
        fixtures.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    base = ROOT/'models/gan-bcap-repair'
    weights = [base/'smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_last.safetensors']
    for step in [1050,1200,1350]:
        name=f'smoke-stepcap2-only-{step}-s7-20260905'
        weights.append(base/name/f'{name}_last.safetensors')
    manifest = dict(version=1, incumbent=str(weights[0]),
        checkpoints=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in weights],
        fixtures=fixtures, seeds=SEEDS, duration=20, initial_prompts=8, maximum_prompts=16,
        planned_comparisons=12, rule=RULE, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        criteria_sha256=hashlib.sha256((ROOT/'analysis/gan_bcap/convergence_criteria.py').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256((ROOT/'analysis/gan_bcap/render_steps.py').read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256((ROOT/'analysis/gan_bcap/autonomous_audio.py').read_bytes()).hexdigest(),
        expansion='Use prompts 8–15 if any paired gain interval remains inconclusive after prompts 0–7.',
        quality_scope='Report relative and absolute concept diagnostics separately. These are not a calibrated voice classifier. No checkpoint can be marked quality-validated by this study alone.',
        catalog='Remains paused regardless of study result.')
    p = WORK/'manifest.json'
    if p.exists() and json.loads(p.read_text()) != manifest:
        raise ValueError('Locked study manifest changed')
    write(p, manifest)
    return manifest


def run(command, log, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1' if gpu else '', HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with Path(log).open('a') as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def score(folder, output):
    if not output.exists() or json.loads(output.read_text()).get('status')!='complete':
        run([sys.executable,'-u',str(ROOT/'analysis/gan_bcap/autonomous_audio.py'),
             '--folders',str(folder),'--concept','gender','--output',str(output)], output.with_suffix('.log'), False)
    report = json.loads(output.read_text())
    expected = hashlib.sha256((ROOT/'analysis/gan_bcap/autonomous_audio.py').read_bytes()).hexdigest()
    if report['rule']!=RULE or report['source_sha256']!=expected:
        raise ValueError('Cached audio score uses a different protocol')
    return report


def evaluate(manifest, count):
    records, groups = [], {}
    for i in range(count):
        report = json.loads((WORK/f'score-{i:02d}.json').read_text())
        records.extend(report['records'])
        for r in report['records']:
            groups[r['fixture']] = manifest['fixtures'][i]['sha256']
    by_checkpoint = {}
    for r in records:
        by_checkpoint.setdefault(r['checkpoint']['path'], {})[r['fixture']] = r['heuristic_score']
    incumbent = by_checkpoint[manifest['incumbent']]
    comparisons = []
    for checkpoint in manifest['checkpoints'][1:]:
        values = by_checkpoint[checkpoint['path']]
        comparisons.append(dict(checkpoint=checkpoint['path'],
            **paired_gain_bound(values, incumbent, groups,
                                planned_comparisons=manifest['planned_comparisons'])))
    diagnostics = {}
    for checkpoint in by_checkpoint:
        rows = [r for r in records if r['checkpoint']['path']==checkpoint]
        diagnostics[checkpoint] = dict(samples=len(rows),
            positive_description_margin_count=sum(r['candidate']['concept']>0 for r in rows),
            negative_description_margin_count=sum(r['candidate']['concept']<0 for r in rows),
            nonsilent_count=sum(r['eligible'] for r in rows),
            reference_positive_margin_count=sum(r['positive_reference']['concept']>0 for r in rows),
            warning='Description similarity sign is not a calibrated voice label or a success rate.')
    result = dict(status='complete', prompt_groups=count, seeds_per_prompt=len(SEEDS),
        comparisons=comparisons, ranking=summarize(records), diagnostics=diagnostics,
        automatic_decision=stopping_decision(comparisons, quality='unvalidated'),
        interpretation='Prospective confirmation of no worthwhile gain on the fixed research metric. Musical quality and game equilibrium remain separate questions.')
    write(WORK/f'comparison-{count:02d}.json', result)
    return result


def main():
    manifest = setup()
    with (WORK/'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = ROOT/'analysis/gan_bcap/convergence_20260905/status.json'
        while json.loads(pending.read_text()).get('status')=='running':
            write(WORK/'status.json', dict(status='waiting', phase='Waiting for unilateral probes to release GPU 1'))
            time.sleep(10)
        with ThreadPoolExecutor(max_workers=1) as cpu:
            for start, end in [(0,8),(8,16)]:
                futures = []
                for i in range(start,end):
                    out = ROOT/'eval/listen'/f'gan-convergence-prompt-{i:02d}-20260905'
                    weights = [r['path'] for r in manifest['checkpoints']]
                    write(WORK/'status.json', dict(status='running', phase=f'Rendering prompt {i+1}/{end}',
                        updated_utc=datetime.now(timezone.utc).isoformat()))
                    used = int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'],text=True).strip())
                    if used>512:
                        raise RuntimeError(f'GPU 1 unexpectedly occupied: {used} MiB')
                    run([sys.executable,'-u',str(ROOT/'analysis/gan_bcap/render_steps.py'),
                         '--weights',*weights,'--out',str(out),'--prompts',manifest['fixtures'][i]['path'],
                         '--seeds',*map(str,SEEDS),'--duration','20'],WORK/f'render-{i:02d}.log',True)
                    futures.append(cpu.submit(score,out,WORK/f'score-{i:02d}.json'))
                write(WORK/'status.json',dict(status='running',phase=f'Finishing CPU measurements for {end} prompts'))
                for future in futures:
                    future.result()
                result = evaluate(manifest,end)
                if not any(r['decision']=='evaluate_more' for r in result['comparisons']):
                    break
        write(WORK/'status.json',dict(status='complete',phase='Prospective paired comparison complete',
            result=str(WORK/f'comparison-{end:02d}.json'),catalog='paused',
            precision_exhausted=any(r['decision']=='evaluate_more' for r in result['comparisons'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if WORK.exists():
            write(WORK/'status.json', dict(status='failed', error=str(exc), catalog='paused'))
        raise
