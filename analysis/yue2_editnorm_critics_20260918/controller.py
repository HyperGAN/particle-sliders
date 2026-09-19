#!/usr/bin/env python3
"""Dual-GPU: paired-edit norm + sn_mlp vs gmix (ParticleGAN, no R1/R2)."""
from __future__ import annotations
import json, os, shutil, signal, subprocess, time
from pathlib import Path

ROOT = Path('/ml2/music/sliders-conceptmod')
EXP = ROOT / 'analysis/yue2_editnorm_critics_20260918'
PYBIN = '/ml2/music/.cache/yue2-test-env/bin/python'
STEPS = 1600
SAVE_EVERY = 100
WARMUP = 200
COS_ABORT = 0.5
DLOSS_ABORT_LOW = 0.15
GADV_ABORT = 4.0
HORIZON_SEC = 8 * 3600


def load_exp():
    return json.loads((EXP / 'experiment.json').read_text())


def write_state(**kw):
    path = EXP / 'state.json'
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw)
    cur['updated'] = time.time()
    path.write_text(json.dumps(cur, indent=2) + '\n')


def argv_for(cand):
    run = EXP / 'runs' / cand['name']
    run.mkdir(parents=True, exist_ok=True)
    prepared = run / 'prepared.pt'
    src = Path(cand['prepared_from'])
    if not prepared.exists() and src.exists():
        shutil.copy2(src, prepared)
    cmd = [
        PYBIN, '-u', str(ROOT / 'conceptmod/textsliders/train_lora_yue2_arm_b.py'),
        '--recipe', 'particle_bridge',
        '--name', f"{cand['slider']}-yue2-{cand['critic']}-editnorm",
        '--prompts_file', cand['prompts'],
        '--save_dir', str(run),
        '--steps', str(STEPS), '--until', str(STEPS),
        '--save_every', str(SAVE_EVERY), '--seed', '7', '--device', 'cuda:0',
        '--sample_seeds', '128', '--history_tokens', '32', '--adv_batch', '8',
        '--critic', cand['critic'],
    ]
    for flag, key in [
        ('--critic_tokens', 'tokens'), ('--critic_width', 'width'),
        ('--critic_layers', 'layers'), ('--critic_heads', 'heads'),
        ('--critic_score_bound', 'score_bound'),
    ]:
        if key in cand and cand[key] is not None:
            cmd.extend([flag, str(cand[key])])
    return cmd, run


def collapsed(run: Path):
    p = run / 'progress.json'
    if not p.exists():
        return False, ''
    j = json.loads(p.read_text())
    step = int(j.get('step') or 0)
    if step < WARMUP:
        return False, ''
    cos = float(j.get('cos_pos') or 0)
    d_loss = float(j.get('d_loss') or 0)
    g_adv = float(j.get('g_adv') or 0)
    if cos < COS_ABORT:
        return True, f'cos_pos={cos:.3f}<{COS_ABORT} @ {step}'
    if d_loss < DLOSS_ABORT_LOW and g_adv > 2.0:
        return True, f'd_loss={d_loss:.3f} g_adv={g_adv:.3f} @ {step}'
    if g_adv > GADV_ABORT:
        return True, f'g_adv={g_adv:.3f}>{GADV_ABORT} @ {step}'
    return False, ''


def summarize(run: Path):
    st = run / 'status.json'
    pr = run / 'progress.json'
    out = {}
    if st.exists():
        j = json.loads(st.read_text())
        out.update({k: j.get(k) for k in ('status', 'completed', 'error')})
    if pr.exists():
        j = json.loads(pr.read_text())
        out.update({k: j.get(k) for k in ('step', 'cos_pos', 'd_loss', 'g_adv', 'grad_norm', 'noise_std')})
    sp = run / 'state.pt'
    if sp.exists():
        try:
            import torch
            hist = torch.load(sp, map_location='cpu', weights_only=True)['history']
            healthy = [h for h in hist if h['step'] >= WARMUP and h['cos_pos'] >= COS_ABORT]
            if healthy:
                best = max(healthy, key=lambda h: h['cos_pos'])
                out['best_healthy_cos'] = best['cos_pos']
                out['best_healthy_step'] = best['step']
            for h in hist:
                if h['step'] >= WARMUP and h['cos_pos'] < COS_ABORT:
                    out['first_collapse_step'] = h['step']
                    out['first_collapse_cos'] = h['cos_pos']
                    break
            else:
                out['survived'] = True
            if hist:
                out['noise_start'] = hist[0].get('noise_std')
        except Exception as e:
            out['hist_error'] = str(e)
    audit = run / 'teacher-audit.json'
    if audit.exists():
        try:
            norm = json.loads(audit.read_text()).get('normalization') or {}
            out['edit_rms'] = norm.get('edit_rms')
            out['noise_start_audit'] = norm.get('noise_start')
            out['norm_mode'] = norm.get('mode')
        except Exception:
            pass
    return out


def main():
    start = time.time()
    deadline = start + HORIZON_SEC
    exp = load_exp()
    pending = list(exp['candidates'])
    results = []
    still = []
    for c in pending:
        run = EXP / 'runs' / c['name']
        st = run / 'status.json'
        if st.exists():
            j = json.loads(st.read_text())
            if j.get('status') in ('complete', 'collapsed_abort', 'checkpoint_ready') and int(j.get('completed') or 0) >= 100:
                results.append(dict(c, result=summarize(run)))
                continue
        still.append(c)
    pending = still
    procs = {0: None, 1: None}

    write_state(phase='running', pending=[c['name'] for c in pending], results=results,
                deadline=deadline, started=start,
                active={str(g): None for g in procs})

    stop = False

    def on_sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    def launch(gpu, cand):
        cmd, run = argv_for(cand)
        log = EXP / 'logs' / f"{cand['name']}.log"
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': str(ROOT),
            'CUDA_VISIBLE_DEVICES': str(gpu),
            'HF_HUB_OFFLINE': '1',
            'HF_HOME': '/ml2/music/.cache/huggingface',
        })
        lf = open(log, 'a', buffering=1)
        lf.write(f'\n===== launch gpu={gpu} {time.ctime()} =====\n')
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        return dict(proc=proc, cand=cand, run=run, log=log, logf=lf, gpu=gpu, launched=time.time())

    def finish(slot, reason):
        info = procs[slot]
        if not info:
            return
        proc = info['proc']
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=60)
        run = info['run']
        st_path = run / 'status.json'
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        if reason.startswith('collapse'):
            st.update(status='collapsed_abort', abort_reason=reason, completed=st.get('completed') or 0)
            st_path.write_text(json.dumps(st, indent=2) + '\n')
        summary = dict(info['cand'], result=summarize(run), abort=reason, gpu=info['gpu'])
        results.append(summary)
        try:
            info['logf'].close()
        except Exception:
            pass
        procs[slot] = None
        write_state(phase='running', pending=[c['name'] for c in pending],
                    active={str(g): (procs[g]['cand']['name'] if procs[g] else None) for g in procs},
                    results=results, deadline=deadline)

    while not stop and time.time() < deadline:
        for gpu in (0, 1):
            if procs[gpu] is None and pending and time.time() < deadline:
                cand = pending.pop(0)
                procs[gpu] = launch(gpu, cand)
                write_state(phase='running', pending=[c['name'] for c in pending],
                            active={str(g): (procs[g]['cand']['name'] if procs[g] else None) for g in procs},
                            results=results, deadline=deadline,
                            message=f'launched {cand["name"]} on gpu {gpu}')

        if all(v is None for v in procs.values()) and not pending:
            break

        for gpu, info in list(procs.items()):
            if not info:
                continue
            code = info['proc'].poll()
            dead, why = collapsed(info['run'])
            if dead:
                finish(gpu, f'collapse:{why}')
                continue
            if code is not None:
                finish(gpu, 'complete' if code == 0 else f'exit_{code}')
                continue
        write_state(phase='running', pending=[c['name'] for c in pending],
                    active={str(g): (procs[g]['cand']['name'] if procs[g] else None) for g in procs},
                    results=results, deadline=deadline, heartbeat=time.time())
        time.sleep(15)

    for gpu in (0, 1):
        if procs[gpu]:
            finish(gpu, 'horizon_stop' if time.time() >= deadline else 'stop')

    board = []
    for row in results:
        r = row.get('result') or {}
        board.append(dict(
            name=row.get('name'), slider=row.get('slider'), critic=row.get('critic'),
            survived=r.get('survived'), first_collapse_step=r.get('first_collapse_step'),
            best_healthy_cos=r.get('best_healthy_cos'), best_healthy_step=r.get('best_healthy_step'),
            final_cos=r.get('cos_pos'), final_step=r.get('step') or r.get('completed'),
            edit_rms=r.get('edit_rms'), noise_start=r.get('noise_start_audit') or r.get('noise_start'),
            abort=row.get('abort'),
        ))
    board.sort(key=lambda x: (
        0 if x.get('survived') else 1,
        -(x.get('best_healthy_cos') or 0),
    ))
    (EXP / 'board.json').write_text(json.dumps(board, indent=2) + '\n')
    lines = ['# Edit-norm critic compare (sn_mlp vs gmix)\n', f'updated: {time.ctime()}\n\n']
    for b in board:
        lines.append(
            f"- **{b['name']}**: survived={b.get('survived')} best_cos={b.get('best_healthy_cos')} "
            f"noise_start={b.get('noise_start')} edit_rms={b.get('edit_rms')} abort={b.get('abort')}\n"
        )
    (EXP / 'board.md').write_text(''.join(lines))
    write_state(phase='done', results=results, board=board, ended=time.time())
    print('done', json.dumps(board, indent=2))


if __name__ == '__main__':
    main()
