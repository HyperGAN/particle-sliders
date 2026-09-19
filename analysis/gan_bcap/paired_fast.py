"""User-requested fast continuation: two clips per recipe, then keep training.

Reuses the exact saved-game trainer. CPU scores follow GPU work asynchronously;
this sparse listening screen does not claim convergence or measure diversity.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.gan_bcap import paired_continuation as base
from analysis.gan_bcap.lm_evaluate import checkpoint_metadata

WORK, PAGE = base.WORK, base.PAGE
SEEDS = (7, 23)
ROW = 0


def setup():
    original = base.setup()
    protocol = dict(version='fast-1', authorization=base.read(WORK / 'fast-sampling-authorization.json'),
        supersedes=base.sha(WORK / 'protocol.json'), checkpoints=list(base.STEPS), gpu=0,
        recipes_unchanged=True, saved_game_resume_unchanged=True,
        prompts=dict(path=str(base.FIXTURES), sha256=base.sha(base.FIXTURES)),
        rows=[ROW], seeds=list(SEEDS), duration=20, clips_per_recipe=2,
        controls='Reuse the existing GPU 0 slider-off and positive-caption audio; no anchor rerendering.',
        scheduling='Two new clips per active recipe, then train the next checkpoint while CPU scoring runs. Stop flags are checked between training and rendering jobs; late audio findings may allow an additional training block.',
        stopping='Nonfinite training or invalid audio stops a branch. Both clips near-silent stop it. Both clips with word precision and phrase accuracy below .35 while both controls are at least .65 on both measures also stop it. One flagged clip is retained as a warning. Low lyric coverage and plateau alone never stop training.',
        inference='Descriptive scores only. No confidence intervals, diversity verdict, convergence claim or automatic larger sweep.',
        rule=original['rule'], original_training_sources=original['sources'],
        source_sha256=base.sha(__file__), frozen_controller_sha256=base.sha(base.__file__),
        renderer_sha256=base.sha(ROOT / 'analysis/gan_bcap/render_v2.py'))
    path = WORK / 'fast-protocol.json'
    if path.exists() and base.read(path) != protocol:
        raise ValueError('Fast continuation protocol changed')
    base.write(path, protocol); base.write(PAGE / 'protocol.json', protocol)


def publish(state):
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    base.write(WORK / 'status.json', state)
    cards = []
    for step in base.STEPS:
        for arm, run in state['stages'].get(str(step), {}).get('runs', {}).items():
            if run.get('complete'):
                p = run['weight']
                cards.append(dict(stage=step, step=step, arm=arm, weight=p, stem=Path(p).stem,
                    label=f'{base.LABELS[arm]} {step}'))
    base.write(PAGE / 'data.json', dict(status=state, cards=cards, seeds=list(SEEDS),
        prompts=[ROW], layout='quick', summaries=[base.read(WORK / f'fast-results-{n}.json')
            for n in base.STEPS if (WORK / f'fast-results-{n}.json').exists()]))


def prepared_audio(step, weights):
    """Copy only exact existing fixtures; return whether GPU work is necessary."""
    from conceptmod.textsliders import generate_listen as G
    folder = PAGE / 'quick' / f'step-{step}' / 'prompt-00'
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(prompts=str(base.FIXTURES.resolve()), prompts_sha256=base.sha(base.FIXTURES),
        row=ROW, seeds=list(SEEDS), duration=20.0, scales=[0., 1.], seed_retries=0,
        renderer_sha256=base.sha(ROOT / 'analysis/gan_bcap/render_v2.py'),
        checkpoints=[dict(path=p, sha256=base.sha(p), steps=checkpoint_metadata(Path(p))[2]) for p in weights])
    manifest = folder / 'render_spec.json'
    if manifest.exists() and base.read(manifest) != spec:
        raise ValueError('Existing quick render has different fixtures or weights')
    source = PAGE / 'step-750/prompt-00'
    old = base.read(source / 'render_spec.json')
    if (old['prompts_sha256'], old['row'], old['duration']) != (spec['prompts_sha256'], ROW, 20):
        raise ValueError('Cached controls use a different prompt')
    known = {p['path']: p['sha256'] for p in old['checkpoints']}
    first = Path(old['checkpoints'][0]['path']).stem
    copied, missing = [], []
    for checkpoint in spec['checkpoints']:
        weight = checkpoint['path']; stem = Path(weight).stem
        for seed in SEEDS:
            if seed not in old['seeds']: raise ValueError('Missing cached control seed')
            dest = folder / f'{stem}-s{seed}'; dest.mkdir(exist_ok=True)
            controls = source / f'{first}-s{seed}'
            files = [controls / '01_slider_neutral_base_zero.wav', controls / '03_REF_prompt_Female_no_slider.wav']
            previous = source / f'{stem}-s{seed}'
            if known.get(weight) == checkpoint['sha256']:
                files += [previous / '02_slider_Female_plus1.wav']
            for p in files:
                if not p.exists(): raise ValueError('An expected preserved sample is missing')
                target = dest / p.name
                if not target.exists(): shutil.copyfile(p, target)
                if base.sha(p) != base.sha(target): raise ValueError('Copied sample hash mismatch')
                copied.append(dict(source=str(p), destination=str(target), sha256=base.sha(target)))
            for name in ('01_slider_neutral_base_zero.wav', '02_slider_Female_plus1.wav', '03_REF_prompt_Female_no_slider.wav'):
                p = dest / name
                if p.exists(): G._inspect_wav(p)
                else: missing.append(str(p))
            base.write(dest / 'checkpoint.json', dict(weights=weight, sha256=checkpoint['sha256'],
                steps=checkpoint['steps'], metadata_source=str(checkpoint_metadata(Path(weight))[1]), seed=seed))
    base.write(manifest, spec)
    base.write(folder / 'reuse.json', dict(source_gpu=0, fixed_seeds=list(SEEDS), samples=copied,
        missing_new_clips=missing, selection='First two declared seeds; no selection by sound or score'))
    return folder, missing


def lyric_alarm(rows):
    if len(rows) != 2 or len({r['seed'] for r in rows}) != 2:
        raise ValueError('Fast lyric check requires two distinct fixed samples')
    def bad(row):
        candidate = row['candidate']['lyric_diagnostics']
        refs = [row[k]['lyric_diagnostics'] for k in ('baseline', 'positive_reference')]
        return (all(candidate[k] < .35 for k in ('precision', 'phrase_accuracy')) and
                all(ref[k] >= .65 for ref in refs for k in ('precision', 'phrase_accuracy')))
    count = sum(bad(r) for r in rows)
    return dict(stop=count == 2, severe_lyric_mismatches=count,
        interpretation='Sparse ASR alarm; low sheet coverage alone is excluded.')


def cheap_audio_alarm(folder, weight):
    import numpy as np
    import soundfile as sf
    flags = []
    for seed in SEEDS:
        sub = folder / f'{Path(weight).stem}-s{seed}'
        wave, rate = sf.read(sub / '02_slider_Female_plus1.wav', always_2d=True)
        control, _ = sf.read(sub / '01_slider_neutral_base_zero.wav', always_2d=True)
        invalid = not wave.size or not np.isfinite(wave).all()
        silent = invalid or float(np.sqrt(np.mean(wave**2))) < .02 * max(float(np.sqrt(np.mean(control**2))), 1e-12)
        flags.append(dict(seed=seed, invalid=invalid, near_silence=silent))
    return dict(stop=any(f['invalid'] for f in flags) or all(f['near_silence'] for f in flags), clips=flags)


def evaluate(step, folder, current):
    from analysis.gan_bcap.autonomous_audio import RULE, summarize
    from conceptmod.textsliders.gan_v2.metrics import clip_diagnostics
    report = WORK / f'fast-scores-{step}.json'
    if not report.exists() or base.read(report).get('status') != 'complete':
        base.run(['analysis/gan_bcap/autonomous_audio.py', '--folders', folder, '--concept', 'gender',
                  '--output', report], WORK / f'fast-score-{step}.log')
    measured = base.read(report); records = measured['records']
    expected = {(w, seed) for w in current.values() for seed in SEEDS}
    observed = [(r['checkpoint']['path'], r['seed']) for r in records]
    if measured['rule'] != RULE or set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Unexpected fast score fixtures')
    if measured['source_sha256'] != base.sha(ROOT / 'analysis/gan_bcap/autonomous_audio.py'):
        raise ValueError('Audio scoring implementation changed')
    ranking = summarize(records); alarms = {}
    for row in ranking:
        examples = [r for r in records if r['checkpoint']['path'] == row['checkpoint']]
        diagnostics = [clip_diagnostics(r['candidate'], r['baseline'], r['positive_reference']) for r in examples]
        row.update(mean_description_margin=statistics.mean(r['candidate']['concept'] for r in examples),
            mean_lyric_proxy=statistics.mean(r['candidate']['lyrics'] for r in examples),
            quality=dict(failure_rate=sum(bool(d['failures']) for d in diagnostics) / len(diagnostics),
                musical_quality_validated=False, decision='sparse_screen'), diversity='not_measured')
        arm = next(a for a, w in current.items() if w == row['checkpoint'])
        alarms[arm] = lyric_alarm(examples)
    comparisons = {}
    scores = {r['checkpoint']: r['heuristic_score'] for r in ranking}
    if all(a in current for a in base.ARMS):
        comparisons['fm_minus_baseline'] = dict(mean_gain=scores[current['fm_capped']] - scores[current['baseline']],
            lower_gain_bound=None, upper_gain_bound=None, decision='descriptive_only')
    result = dict(step=step, status='fast_screen', prompt_count=1, seeds=list(SEEDS), ranking=ranking,
        comparisons=comparisons, alarms=alarms, musical_quality_validated=False,
        interpretation='Two clips per recipe. Descriptive trends and obvious-failure alarms only; no diversity or convergence verdict.')
    base.write(WORK / f'fast-results-{step}.json', result)
    base.write(PAGE / f'fast-results-{step}.json', result)
    return result


def main():
    setup(); base.publish = publish
    with (WORK / 'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = base.read(WORK / 'status.json')
        if state.get('fast_complete'): return
        state.update(status='running', sampling_mode='two_clips_per_recipe', scoring='CPU in background')
        state.pop('error', None); base.phase(state, 'Fast continuation: two clips per recipe, then resume training')
        pending = {}
        def collect(wait=False):
            for step, future in list(pending.items()):
                if wait or future.done():
                    result = future.result()
                    for arm, alarm in result['alarms'].items():
                        if alarm['stop']:
                            state['stopped_arms'][arm] = dict(step=step, reason='Both fixed samples show severe lyric mismatch', **alarm)
                    state['stages'][str(step)]['fast_scored'] = True
                    pending.pop(step); publish(state)
        try:
            with ThreadPoolExecutor(max_workers=1) as cpu:
                for step in base.STEPS:
                    collect()
                    stage = state['stages'].setdefault(str(step), dict(runs={}, complete=False))
                    for arm in base.ARMS:
                        collect()
                        if arm not in state['stopped_arms']: base.continuation(state, stage, arm, step)
                    collect()
                    current = {a: r['weight'] for a, r in stage['runs'].items() if r.get('complete')}
                    if not current: break
                    folder, missing = prepared_audio(step, list(current.values()))
                    if missing:
                        base.phase(state, f'Rendering {step}: two clips per recipe, shared cached controls, GPU 0')
                        base.run(['analysis/gan_bcap/render_v2.py', '--weights', *current.values(), '--out', folder,
                            '--prompts', base.FIXTURES, '--row', ROW, '--seeds', *SEEDS, '--duration', '20'],
                            WORK / f'fast-render-{step}.log', gpu=True)
                    for arm, weight in current.items():
                        alarm = cheap_audio_alarm(folder, weight)
                        if alarm['stop']:
                            state['stopped_arms'][arm] = dict(step=step, reason='Invalid audio or both samples near-silent', **alarm)
                    stage.update(complete=True, fast_rendered=True)
                    pending[step] = cpu.submit(evaluate, step, folder, current)
                    base.phase(state, f'{step} samples ready; CPU scores in background, continuing training')
                base.phase(state, 'Training sweep finished; completing remaining background scores')
                collect(wait=True)
            state.update(status='complete', fast_complete=True,
                conclusion='Reached 2100 or stopped an affected branch for an observed severe failure. Sparse samples do not establish a winner or convergence.')
            base.phase(state, 'Fast continuation complete; samples and all checkpoints retained')
        except Exception as error:
            state.update(status='failed', error=str(error))
            base.phase(state, 'Fast continuation interrupted; full states and samples retained')
            raise


if __name__ == '__main__':
    if '--check' in sys.argv:
        setup(); print('Fast protocol and unchanged training sources verified.')
    else:
        main()
