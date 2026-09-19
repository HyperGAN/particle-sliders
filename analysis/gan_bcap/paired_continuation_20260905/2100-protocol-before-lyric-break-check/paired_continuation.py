"""Continue the two preferred GAN recipes with matched, preserved audio checks.

The protocol is frozen before new updates. Training resumes the complete game;
only the resource endpoint advances, preserving the constant-LR signature.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = ROOT / 'analysis/gan_bcap/paired_continuation_20260905'
PAGE = ROOT / 'eval/listen/gan-paired-continuation-20260905'
FIXTURES = ROOT / 'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml'
TRAIN = ROOT / 'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml'
OLD_PAGE = ROOT / 'eval/listen/gan-v2-20260905'
ARMS = ('baseline', 'fm_capped')
STEPS = (750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100)
SEEDS = (7, 23, 101, 303)
LABELS = {'baseline': 'Bounded baseline', 'fm_capped': 'FM gradient limit'}
OLD_SERVICE = 'music-gan-convergence-audio-20260905.service'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def initial(arm):
    return ROOT / 'models/gan-v2' / f'{arm}-660-20260905'


def anchors():
    original = ROOT / 'models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_last.safetensors'
    return [str(original)] + [str(initial(a) / f'{initial(a).name}_step660.safetensors') for a in ARMS]


def setup():
    import yaml
    from conceptmod.textsliders.gan_v2.data import validate_prompts
    from conceptmod.textsliders.gan_v2.state import code_fingerprints
    from analysis.gan_bcap.autonomous_audio import RULE
    rows = yaml.safe_load(FIXTURES.read_text())['rows']
    training = yaml.safe_load(TRAIN.read_text())
    training = training['rows'] if isinstance(training, dict) else training
    validate_prompts(rows + training)
    if len(rows) != 8 or len({r['lyrics'] for r in rows}) != 8:
        raise ValueError('Need eight distinct reserved lyric sheets')
    if {r['lyrics'] for r in rows} & {r['lyrics'] for r in training}:
        raise ValueError('Training/evaluation lyrics overlap')
    sources = code_fingerprints()
    for arm in ARMS:
        saved = read(initial(arm) / 'manifest.json')
        if saved['sources'] != sources or saved['recipe']['schedule'] != 'constant':
            raise ValueError('Saved trainer sources changed, or continuation is not constant LR')
    protocol = dict(version=1, arms=list(ARMS), checkpoints=list(STEPS), gpu=0,
        starting_states={a: dict(path=str(initial(a) / 'state.pt'), sha256=sha(initial(a) / 'state.pt')) for a in ARMS},
        anchors=[dict(path=p, sha256=sha(p)) for p in anchors()],
        training_prompts=dict(path=str(TRAIN), sha256=sha(TRAIN)),
        evaluation_prompts=dict(path=str(FIXTURES), sha256=sha(FIXTURES)),
        rows=list(range(8)), seeds=list(SEEDS), duration=20, multiplier=1,
        schedule='Original constant LR; signed origin=600 and horizon=660 retained. Only --until advances.',
        state='Restore generator, critic, both optimizers, sampler, RNG and history; no reset or new recipe.',
        rule=RULE, worthwhile_gain=.05, family_alpha=.05, planned_tail_tests=100,
        comparisons='At eight prompts: FM minus baseline, each arm minus its own 660, and each arm minus its previous checkpoint. Five comparisons times ten checkpoints times two interval tails.',
        screening='First two prompts are descriptive only. All eight prompts and all four seeds remain in each completed comparison.',
        stop_rule='Stop an arm on nonfinite training; or at least two consensus diversity alarms out of eight prompts; or at least four near-silent clips out of 32. Lyric regression and metric plateau alone do not stop a GAN branch.',
        budget='2100 is a user-requested resource ceiling, not evidence of convergence. No automatic catalog promotion.',
        limits='Eight repeated selection prompts, one shared training ancestry and one training seed. Proxy intervals do not validate musical quality. Best-over-checkpoints score is exploratory and requires fresh confirmation.',
        sources=sources, analysis_sources={str(p): sha(p) for p in [Path(__file__),
            ROOT / 'analysis/gan_bcap/render_v2.py', ROOT / 'analysis/gan_bcap/autonomous_audio.py',
            ROOT / 'analysis/gan_bcap/quality_audit_v2.py', ROOT / 'analysis/gan_bcap/audio_v2.py',
            ROOT / 'analysis/gan_bcap/convergence_criteria.py']})
    WORK.mkdir(parents=True, exist_ok=True); PAGE.mkdir(parents=True, exist_ok=True)
    path = WORK / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Locked continuation protocol changed')
    write(path, protocol); write(PAGE / 'protocol.json', protocol)
    return protocol


def publish(state):
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write(WORK / 'status.json', state)
    cards = []
    for step in STEPS:
        stage = state['stages'].get(str(step))
        if not stage: continue
        weights = anchors() + [r['weight'] for r in stage['runs'].values() if r.get('complete')]
        for p in weights:
            arm = next((a for a in ARMS if Path(p).stem.startswith(a + '-')), 'original')
            number = 600 if arm == 'original' else (660 if p in anchors() else step)
            cards.append(dict(stage=step, weight=p, stem=Path(p).stem, arm=arm, step=number,
                label=('Original' if arm == 'original' else LABELS[arm]) + f' {number}'))
    write(PAGE / 'data.json', dict(status=state, cards=cards, seeds=list(SEEDS),
        summaries=[read(WORK / f'results-{n}.json') for n in STEPS if (WORK / f'results-{n}.json').exists()]))


def phase(state, message):
    state['phase'] = message; publish(state); print(message, flush=True)


def run(arguments, log, *, gpu=False, training=False):
    if gpu:
        required = 32768 if training else 28672
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= required: break
            print(f'Waiting for GPU 0: {free} MiB free, need {required}', flush=True)
            time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0' if gpu else '', HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with Path(log).open('a') as handle:
        subprocess.run([sys.executable, '-u', *map(str, arguments)], cwd=ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=True)


def continuation(state, stage, arm, target):
    import torch
    entry = stage['runs'].get(arm, {})
    if entry.get('complete'): return
    source = initial(arm) / 'state.pt'
    for previous in STEPS:
        prior = state['stages'].get(str(previous), {}).get('runs', {}).get(arm, {})
        if previous < target and prior.get('complete'): source = Path(prior['directory']) / 'state.pt'
    known = Path(entry['directory']) if entry.get('directory') else None
    if known and (known / 'state.pt').exists():
        source = known / 'state.pt'
        completed = torch.load(source, map_location='cpu', weights_only=True)['completed']
        if completed == target:
            latest = read(known / 'latest.json')
            if latest['step'] != target or not Path(latest['checkpoints']['live']).exists():
                raise ValueError('Completed state has no matching inference checkpoint')
            entry.update(complete=True, weight=latest['checkpoints']['live'])
            publish(state); return
        if completed > target: raise ValueError('Saved state is past the stage target')
    name = f'{arm}-to{target}-20260905'
    directory = ROOT / 'models/gan-v2' / name
    attempt = 1
    while directory.exists():
        attempt += 1; directory = ROOT / 'models/gan-v2' / f'{name}-attempt{attempt}'
    entry = dict(directory=str(directory), resume=str(source), complete=False)
    stage['runs'][arm] = entry
    phase(state, f'Training {LABELS[arm]} to {target} on GPU 0')
    try:
        run(['-m', 'conceptmod.textsliders.gan_v2.train', '--prompts', TRAIN, '--run-dir', directory,
             '--arm', arm, '--resume', source, '--schedule-origin', '600', '--schedule-horizon', '660',
             '--until', target, '--save-every', '30', '--ema-every', '15', '--diagnostics-every', '15',
             '--no-ema'], WORK / f'{directory.name}.log', gpu=True, training=True)
    except subprocess.CalledProcessError:
        log = (WORK / f'{directory.name}.log').read_text()
        if 'FloatingPointError: Nonfinite' not in log: raise
        state['stopped_arms'][arm] = dict(step=target, reason='Nonfinite training; previous checkpoints retained')
        entry['numerical_failure'] = True; publish(state); return
    latest = read(directory / 'latest.json')
    if latest['step'] != target: raise RuntimeError('Training interrupted before endpoint')
    lines = [json.loads(line) for p in directory.glob('train-from-*.jsonl') for line in p.read_text().splitlines()]
    if not lines or any(r['parameter_step'] > 2.00001 for r in lines):
        raise ValueError('Missing updates or parameter bound violation')
    entry.update(complete=True, weight=latest['checkpoints']['live'],
        validation=dict(updates_this_attempt=len(lines), maximum_parameter_step=max(r['parameter_step'] for r in lines),
            fm_limit_activated=sum(r['fm_gradient']['factor'] < 1 for r in lines),
            constant_lr=all(r['lr_scale'] == 1 for r in lines)))
    if not entry['validation']['constant_lr']: raise ValueError('Continuation changed the learning rate')
    publish(state)


def reuse_audio(folder, weights, row, step):
    # The earlier study's prompt 1 was fully rendered on physical GPU 0.
    # Prompt 0 was GPU 1, so its anchors are rendered again on GPU 0 here.
    source = OLD_PAGE / 'prompt-01' if step == STEPS[0] and row == 1 else PAGE / f'step-{STEPS[0]}' / f'prompt-{row:02d}'
    if source == folder or not (source / 'render_spec.json').exists(): return
    spec = read(source / 'render_spec.json')
    if (spec['prompts_sha256'], spec['row'], spec['seeds'], spec['duration']) != (sha(FIXTURES), row, list(SEEDS), 20):
        raise ValueError('Cached controls use a different fixture')
    known = {p['path']: p['sha256'] for p in spec['checkpoints']}
    first = Path(spec['checkpoints'][0]['path']).stem
    copied = []
    for weight in weights:
        for seed in SEEDS:
            dest = folder / f'{Path(weight).stem}-s{seed}'; dest.mkdir(parents=True, exist_ok=True)
            controls = source / f'{first}-s{seed}'
            files = list(controls.glob('01_*.wav')) + list(controls.glob('03_*.wav'))
            if known.get(weight) == sha(weight):
                files += list((source / f'{Path(weight).stem}-s{seed}').glob('02_*.wav'))
            for p in files:
                target = dest / p.name
                if not target.exists(): shutil.copyfile(p, target)
                if sha(p) != sha(target): raise ValueError('Reused audio hash mismatch')
                copied.append(dict(source=str(p), destination=str(target), sha256=sha(target)))
    write(folder / 'audio-reuse.json', dict(reason='Exact fixture and checkpoint reuse on GPU 0', files=copied))


def score(folder, output):
    from analysis.gan_bcap.autonomous_audio import RULE
    if not output.exists() or read(output).get('status') != 'complete':
        run(['analysis/gan_bcap/autonomous_audio.py', '--folders', folder, '--concept', 'gender', '--output', output], output.with_suffix('.log'))
    report = read(output)
    if report['rule'] != RULE or report['source_sha256'] != sha(ROOT / 'analysis/gan_bcap/autonomous_audio.py'):
        raise ValueError('Audio scorer changed')
    spec = read(folder / 'render_spec.json')
    expected = {(p['path'], seed) for p in spec['checkpoints'] for seed in SEEDS}
    observed = [(r['checkpoint']['path'], r['seed']) for r in report['records']]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Missing or duplicate matched evaluation clips')
    return report


def paired(a, b, records, *, count):
    from analysis.gan_bcap.convergence_criteria import paired_gain_bound
    values = {p: {r['fixture']: r['heuristic_score'] for r in records if r['checkpoint']['path'] == p} for p in (a, b)}
    groups = {}
    for r in records:
        if r['checkpoint']['path'] == a:
            spec = read(Path(r['baseline']['audio']).parent.parent / 'render_spec.json')
            groups[r['fixture']] = f"{spec['prompts_sha256']}:{spec['row']}"
    result = paired_gain_bound(values[a], values[b], groups, planned_comparisons=100)
    return dict(challenger=a, incumbent=b, **result)


def severe_audio_alarm(details):
    if len(details['diversity']) != 8 or len(details['clips']) != 32:
        raise ValueError('Severe audio stopping rule requires all eight prompts and four seeds')
    collapse = sum(v['status'] == 'collapse_suspected' for v in details['diversity'].values())
    silent = sum('near_silence' in r['diagnostics']['failures'] for r in details['clips'])
    return dict(stop=collapse >= 2 or silent >= 4,
        consensus_collapse_prompts=collapse, near_silent_clips=silent)


def assess(state, step, reports, *, full):
    from analysis.gan_bcap.autonomous_audio import summarize
    records = [r for p in reports for r in read(p)['records']]
    count = len(reports)
    ranking = summarize(records)
    comparisons = {}
    runs = state['stages'][str(step)]['runs']
    current = {a: r['weight'] for a, r in runs.items() if r.get('complete')}
    if all(a in current for a in ARMS):
        comparisons['fm_minus_baseline'] = paired(current['fm_capped'], current['baseline'], records, count=count)
    for arm, weight in current.items():
        comparisons[f'{arm}_minus_660'] = paired(weight, anchors()[ARMS.index(arm) + 1], records, count=count)
        index = STEPS.index(step)
        if full and index:
            previous = STEPS[index - 1]
            prior = state['stages'].get(str(previous), {}).get('runs', {}).get(arm, {})
            if prior.get('complete'):
                old = [r for i in range(8) for r in read(WORK / f'scores-{previous}-{i:02d}.json')['records']
                       if r['checkpoint']['path'] == prior['weight']]
                comparisons[f'{arm}_minus_previous'] = paired(weight, prior['weight'], records + old, count=count)
        else:
            comparisons[f'{arm}_minus_previous'] = comparisons[f'{arm}_minus_660']
    audit = None
    if full:
        output = WORK / f'audit-{step}.json'
        if not output.exists() or read(output).get('status') != 'complete':
            run(['analysis/gan_bcap/quality_audit_v2.py', '--scores', *reports, '--output', output], WORK / f'audit-{step}.log')
        audit = read(output)
        for arm, weight in current.items():
            details = audit['candidates'][weight]
            alarm = severe_audio_alarm(details)
            if alarm['stop']:
                state['stopped_arms'][arm] = dict(step=step, reason='Declared severe audio alarm',
                    **alarm)
    # Candidate records retain words/phrases/coverage separately from the composite.
    for row in ranking:
        examples = [r for r in records if r['checkpoint']['path'] == row['checkpoint']]
        row['mean_description_margin'] = statistics.mean(r['candidate']['concept'] for r in examples)
        row['mean_lyric_proxy'] = statistics.mean(r['candidate']['lyrics'] for r in examples)
        if audit:
            details = audit['candidates'][row['checkpoint']]
            row['quality'] = details['quality']; row['score_quantile_10'] = details['score_quantile_10']
            row['consensus_collapse_prompts'] = sum(v['status'] == 'collapse_suspected' for v in details['diversity'].values())
    result = dict(step=step, status='complete' if full else 'descriptive_screen', prompt_count=count,
        seeds=list(SEEDS), ranking=ranking, comparisons=comparisons, musical_quality_validated=False,
        interpretation='Best average proxy and persistence are separate. Budget completion is not convergence. Individual failures and matched listening remain visible.')
    write(WORK / f'results-{step}.json', result); write(PAGE / f'results-{step}.json', result)
    if full: shutil.copyfile(WORK / f'audit-{step}.json', PAGE / f'audit-{step}.json')
    publish(state)


def main():
    setup()
    with (WORK / 'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(WORK / 'status.json') if (WORK / 'status.json').exists() else dict(stages={}, stopped_arms={}, gpu=0, catalog='paused')
        if state.get('status') == 'complete': return
        state['status'] = 'running'; state.pop('error', None)
        phase(state, 'Continuing both recipes from their complete 660 states')
        try:
            for step in STEPS:
                stage = state['stages'].setdefault(str(step), dict(runs={}, complete=False))
                if stage['complete']: continue
                for arm in ARMS:
                    if arm not in state['stopped_arms']: continuation(state, stage, arm, step)
                weights = anchors() + [r['weight'] for r in stage['runs'].values() if r.get('complete')]
                if len(weights) == len(anchors()): break
                reports = []
                with ThreadPoolExecutor(max_workers=1) as cpu:
                    pending = []
                    for row in range(8):
                        folder = PAGE / f'step-{step}' / f'prompt-{row:02d}'
                        folder.mkdir(parents=True, exist_ok=True)
                        report = WORK / f'scores-{step}-{row:02d}.json'; reports.append(report)
                        if not (report.exists() and read(report).get('status') == 'complete'):
                            phase(state, f'Rendering {step}: prompt {row + 1}/8, four matched seeds, GPU 0')
                            reuse_audio(folder, weights, row, step)
                            run(['analysis/gan_bcap/render_v2.py', '--weights', *weights, '--out', folder,
                                 '--prompts', FIXTURES, '--row', row, '--seeds', *SEEDS, '--duration', '20'],
                                WORK / f'render-{step}-{row:02d}.log', gpu=True)
                        pending.append(cpu.submit(score, folder, report))
                        if row == 1:
                            phase(state, f'Finishing the first two-prompt comparison at {step}')
                            for future in pending: future.result()
                            assess(state, step, reports, full=False)
                    phase(state, f'Finishing scores and diversity checks at {step}, eight prompts')
                    for future in pending: future.result()
                assess(state, step, reports, full=True)
                stage['complete'] = True; publish(state)
            state.update(status='complete', conclusion='Declared continuation budget or severe-alarm stop reached; convergence and a universal recipe winner are not established.')
            phase(state, 'Paired continuation complete; all checkpoints and audio retained')
            subprocess.run(['systemctl', '--user', 'start', OLD_SERVICE], check=True)
        except Exception as error:
            state.update(status='failed', error=str(error))
            phase(state, 'Continuation interrupted; saved states and first samples retained for recovery')
            raise


if __name__ == '__main__':
    if '--check' in sys.argv:
        setup(); print('Protocol, source signatures, prompts and starting artifacts verified.')
    else:
        main()
