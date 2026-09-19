"""Held-out distribution probe for YuE2 particle-bridge checkpoints.

Runs during training, immediately after each EMA milestone, while the base
model is already loaded. Writes ``prepared-eval.pt``, ``probe-fixtures.pt``,
``probe.jsonl`` and ``probe-summary.json``.

The independent two-sample classifier is intentionally not trained here.
``evaluator_auc`` stays null until that pass exists. Do not rank a run as
finished on these metrics alone.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conceptmod.textsliders.particle_bridge_gan import register_paired_error_norm

STRENGTHS = (0.25, 0.5, 1.0)
EVAL_SIGMAS = (0.5, 1.0, 2.0)
PROJECTION_COUNT = 256
PROJECTION_SEED = 20260918
NOISE_SEED = 20260919
PROBE_SEED_OFFSET = 770_001
LISTEN_SEED_OFFSET = 880_001
FLAG_RATIO = 1.5
PLATEAU_RMS_TOL = 0.01
PLATEAU_SWD_TOL = 0.02
SUBGROUP_WORSE = 0.10
GAIN_ABS_TOL = 0.05
GAIN_REL_TOL = 0.10
SELECTION_RMS_TOL = 0.01
SELECTION_SWD_TOL = 0.02
SELECTION_ABS = 1e-4
SIGMA_FIELDS = {
    0.5: 'game_swd_sigma_0_5',
    1.0: 'game_swd_sigma_1',
    2.0: 'game_swd_sigma_2',
}

_BPM = re.compile(r'\bBPM\s+(\d+)\b', re.I)
_VOCAL = re.compile(r'\b(female|male)\s+lead\b', re.I)


def independent_evaluator_auc(real, fake, groups):
    """Placeholder for a fresh row-held-out classifier.

    When implemented: split by probe-row identity (template, continuation
    seed), not by noise draw; train a new classifier on one set of rows;
    score ROC AUC and a bootstrap interval on the other rows. Do not reuse
    the training discriminator. This pass does not train that classifier.
    """
    del real, fake, groups
    return None


def prompt_subgroup(neutral):
    """Vocal configuration and tempo bucket parsed from a neutral caption."""
    text = neutral or ''
    bpm_match = _BPM.search(text)
    vocal_match = _VOCAL.search(text)
    bpm = int(bpm_match.group(1)) if bpm_match else None
    if bpm is None:
        tempo = 'unknown'
    elif bpm < 110:
        tempo = 'under_110'
    elif bpm <= 130:
        tempo = '110_to_130'
    else:
        tempo = 'over_130'
    vocal = vocal_match.group(1).lower() if vocal_match else 'unknown'
    return dict(vocal=vocal, tempo_bpm=bpm, tempo_range=tempo)


def draw_unique_seeds(count, rng_seed, exclude=()):
    """Draw ``count`` distinct continuation seeds outside ``exclude``."""
    count = int(count)
    if count < 1:
        raise ValueError('Need at least one continuation seed')
    blocked = {int(seed) for seed in exclude}
    rng = torch.Generator().manual_seed(int(rng_seed))
    chosen = []
    seen = set(blocked)
    spins = 0
    while len(chosen) < count:
        spins += 1
        if spins > max(20, count):
            raise RuntimeError('Failed to allocate unique probe seeds')
        draw = torch.randint(0, 2**31 - 1, (max(count * 4, 64),), generator=rng).tolist()
        for value in draw:
            seed_i = int(value)
            if seed_i in seen:
                continue
            seen.add(seed_i)
            chosen.append(seed_i)
            if len(chosen) == count:
                return chosen
    return chosen


def plan_probe_seeds(n_templates, training_seeds, *, probe_per, listen_per, seed):
    """Per-template probe seeds plus a reserved listening set, disjoint from training."""
    n_templates = int(n_templates)
    probe_per = int(probe_per)
    listen_per = int(listen_per)
    if n_templates < 1 or probe_per < 1 or listen_per < 1:
        raise ValueError('Probe planning needs templates and at least one seed of each kind')
    training = [int(item) for item in training_seeds]
    listening_flat = draw_unique_seeds(
        n_templates * listen_per, int(seed) + LISTEN_SEED_OFFSET, training)
    probe_flat = draw_unique_seeds(
        n_templates * probe_per, int(seed) + PROBE_SEED_OFFSET,
        training + listening_flat)

    def assign(flat, per):
        rows = []
        for template_row in range(n_templates):
            chunk = flat[template_row * per:(template_row + 1) * per]
            rows.extend(dict(template_row=template_row, sample_seed=int(item)) for item in chunk)
        return rows

    plan = dict(listening=assign(listening_flat, listen_per), probe=assign(probe_flat, probe_per))
    probe_set = {item['sample_seed'] for item in plan['probe']}
    listen_set = {item['sample_seed'] for item in plan['listening']}
    if (len(probe_set) != n_templates * probe_per or len(listen_set) != n_templates * listen_per
            or probe_set & set(training) or listen_set & set(training) or probe_set & listen_set):
        raise RuntimeError('Probe seed allocation collided')
    return plan


def paired_edit_scale(rows):
    """Shared per-coordinate scale from the training paired edits."""
    if len(rows) < 2:
        raise ValueError('Paired-edit scale needs at least two training rows')
    targets = torch.stack([_vector(row, 'targets', 'positive') for row in rows])
    neutrals = torch.stack([_vector(row, 'neutral') for row in rows])

    class _Scale(nn.Module):
        pass

    holder = _Scale()
    register_paired_error_norm(holder, targets, neutrals)
    scale = holder.target_std.detach().float().cpu().contiguous()
    return scale, float(holder.edit_rms)


def projections_for(dim, count=PROJECTION_COUNT, seed=PROJECTION_SEED):
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(int(count), int(dim), generator=generator)
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def unit_noise(rows, dim, seed=NOISE_SEED):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(int(rows), int(dim), generator=generator)


def sliced_wasserstein(left, right, projections):
    """Mean, median and p95 of 1-D Wasserstein distances on fixed projections."""
    if left.shape != right.shape or left.ndim != 2 or projections.ndim != 2:
        raise ValueError('Sliced Wasserstein expects [rows, hidden] and [projections, hidden]')
    if projections.shape[1] != left.shape[1] or len(left) < 1 or len(projections) < 1:
        raise ValueError('Projection width or sample count is wrong')
    left_p = left @ projections.T
    right_p = right @ projections.T
    distances = (left_p.sort(0).values - right_p.sort(0).values).abs().mean(0)
    return dict(
        mean=float(distances.mean()),
        median=float(torch.quantile(distances, 0.5)),
        p95=float(torch.quantile(distances, 0.95)),
    )


def score_heldout(neutral, positive, prediction, scale, projections, noise, *,
                  strength, groups, sigmas=EVAL_SIGMAS):
    """Paired-error, gain, orthogonal leakage and fixed-sigma distribution distances."""
    neutral = neutral.detach().float().cpu()
    positive = positive.detach().float().cpu()
    prediction = prediction.detach().float().cpu()
    scale = scale.detach().float().cpu().reshape(-1)
    projections = projections.detach().float().cpu()
    noise = noise.detach().float().cpu()
    strength = float(strength)
    tensors = (neutral, positive, prediction, scale, projections, noise)
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError('Non-finite probe tensors')
    if neutral.ndim != 2 or positive.shape != neutral.shape or prediction.shape != neutral.shape:
        raise ValueError('Probe tensors must share shape [rows, hidden]')
    if tuple(scale.shape) != (neutral.shape[1],) or projections.shape[1] != neutral.shape[1]:
        raise ValueError('Paired-edit scale or projections do not match hidden width')
    if noise.shape != neutral.shape or len(groups) != len(neutral):
        raise ValueError('Eval noise and subgroup labels must align with probe rows')

    teacher_full = positive - neutral
    teacher_s = neutral + strength * teacher_full
    normalized = (prediction - teacher_s) / scale
    row_rms = normalized.pow(2).mean(-1).sqrt()
    row_stats = _distribution(row_rms)
    denom = teacher_full.pow(2).sum(-1).clamp_min(1e-12)
    predicted_edit = prediction - neutral
    gain = predicted_edit.mul(teacher_full).sum(-1) / denom
    gain_error = (gain - strength).abs()
    orthogonal = predicted_edit - gain[:, None] * teacher_full
    orthogonal_rel = orthogonal.norm(dim=-1) / teacher_full.norm(dim=-1).clamp_min(1e-12)
    cosine = F.cosine_similarity(predicted_edit, teacher_full, dim=-1, eps=1e-8)
    predicted_norm = predicted_edit / scale
    teacher_norm = (strength * teacher_full) / scale
    teacher_swd = sliced_wasserstein(predicted_norm, teacher_norm, projections)
    game = {}
    for sigma in sigmas:
        real = noise * float(sigma)
        game[SIGMA_FIELDS[float(sigma)]] = sliced_wasserstein(real, real + normalized, projections)['mean']
    record = dict(
        strength=strength,
        examples=int(len(neutral)),
        residual_rms=float(normalized.pow(2).mean().sqrt()),
        residual_row_mean=row_stats['mean'],
        residual_median=row_stats['median'],
        residual_p90=row_stats['p90'],
        residual_p95=row_stats['p95'],
        residual_max=row_stats['maximum'],
        gain_mean=float(gain.mean()),
        gain_p95_error=float(torch.quantile(gain_error.float(), 0.95)),
        orthogonal_error=float(orthogonal_rel.mean()),
        cosine_mean=float(cosine.mean()),
        teacher_swd=teacher_swd['mean'],
        teacher_swd_median=teacher_swd['median'],
        teacher_swd_p95=teacher_swd['p95'],
        evaluator_auc=independent_evaluator_auc(noise, normalized, groups),
        evaluator_status='not_trained',
        normalization='paired_edit_per_coordinate_std_median_rms_gain',
        subgroups=_subgroups(
            normalized, predicted_norm, teacher_norm, projections, groups,
            row_stats['median'], teacher_swd['mean'], row_rms),
    )
    record.update(game)
    _reject_nonfinite(record)
    return record


def milestone_steps(run):
    """EMA milestone exports, oldest first. Live weights are not included."""
    found = []
    for path in Path(run).glob('*_step*.safetensors'):
        if '_live_step' in path.name:
            continue
        mark = '_step'
        index = path.stem.rfind(mark)
        if index < 0:
            continue
        suffix = path.stem[index + len(mark):]
        if suffix.isdigit():
            found.append((int(suffix), path))
    return sorted(found)


def load_probe_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def needs_probe(run, step, kind='ema'):
    have = set()
    for record in load_probe_jsonl(Path(run) / 'probe.jsonl'):
        if int(record.get('step', -1)) == int(step) and record.get('weights_kind') == kind:
            have.add(round(float(record['strength']), 4))
    return any(round(strength, 4) not in have for strength in STRENGTHS)


def build_summary(records, *, stride, training_finite=True):
    grouped = _group_ema(records)
    stopping = stopping_status(grouped, stride=stride, training_finite=training_finite)
    chosen = select_checkpoint(grouped)
    return dict(
        schema=1,
        evaluator_status='not_trained',
        selection_rule='earliest_ema_within_1pct_residual_rms_1pct_p95_2pct_teacher_swd',
        strengths=list(STRENGTHS),
        eval_sigmas=list(EVAL_SIGMAS),
        fixed_eval_sigma=1.0,
        stopping=stopping,
        best_step=None if chosen is None else chosen['step'],
        quality_best_step=None if chosen is None else chosen['quality_step'],
        best_weights_kind='ema',
        gain_ok=None if chosen is None else chosen['gain_ok'],
        best_metrics=None if chosen is None else _public_metrics(chosen['record']),
        steps=sorted(grouped),
    )


@contextmanager
def preserve_rng(device):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if getattr(device, 'type', None) == 'cuda' else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def use_state(network, state):
    """Score ``state`` then restore the live weights, including on failure."""
    device = next(network.parameters()).device
    live = {key: value.detach().clone() for key, value in network.state_dict().items()}
    mapped = {key: value.detach().to(device=device) for key, value in state.items()}
    network.load_state_dict(mapped)
    try:
        yield
    finally:
        network.load_state_dict(live)


def ensure_fixtures(path, *, dim, rows, scale):
    path = Path(path)
    scale = scale.detach().float().cpu().reshape(-1).contiguous()
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=True)
        _validate_fixtures(blob, dim, rows, scale)
        return blob
    blob = dict(
        schema=1,
        projection_seed=PROJECTION_SEED,
        noise_seed=NOISE_SEED,
        projection_count=PROJECTION_COUNT,
        projections=projections_for(dim),
        noise=unit_noise(rows, dim),
        paired_edit_scale=scale,
        eval_sigmas=list(EVAL_SIGMAS),
    )
    _validate_fixtures(blob, dim, rows, scale)
    _torch_save(path, blob)
    return blob


def ensure_bank(path, backend, templates, training_rows, *, max_seq_len, history_tokens,
                probe_per, listen_per, seed, model_id, dummy):
    """Build ``prepared-eval.pt`` once. Listening seeds are reserved, not forwarded."""
    path = Path(path)
    if any('sample_seed' not in row for row in training_rows):
        raise ValueError('Training rows have no continuation seeds; the probe requires a seed bank')
    training_seeds = [int(row['sample_seed']) for row in training_rows]
    key = dict(
        schema=1, seed=int(seed), history_tokens=int(history_tokens),
        probe_per_template=int(probe_per), listening_per_template=int(listen_per),
        training_seeds=training_seeds, template_sha256=_template_sha(templates),
        model_id=model_id, dummy=bool(dummy), max_seq_len=int(max_seq_len),
    )
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=True)
        if blob.get('key') != key:
            raise ValueError(f'Held-out probe bank does not match this run; delete {path.name}')
        return blob
    if int(probe_per) < 16 and not dummy:
        print(json.dumps(dict(
            probe='warning', probe_seeds=int(probe_per),
            note='Held-out probe prefers at least 16 continuation seeds per template')), flush=True)
    plan = plan_probe_seeds(
        len(templates), training_seeds, probe_per=probe_per, listen_per=listen_per, seed=seed)
    limit = min(int(max_seq_len), backend.model.config.max_position_embeddings)
    prefixes = []
    for row in templates:
        neutral_prefix = backend.prefix(row['neutral'], row['lyrics'])
        positive_prefix = backend.prefix(row['positive'], row['lyrics'])
        if max(len(neutral_prefix), len(positive_prefix)) + int(history_tokens) > limit:
            raise ValueError('Prompt plus probe history exceeds context limit')
        prefixes.append((neutral_prefix, positive_prefix, prompt_subgroup(row['neutral'])))
    prepared = []
    total = len(plan['probe'])
    for cursor, item in enumerate(plan['probe'], start=1):
        template_index = int(item['template_row'])
        sample_seed = int(item['sample_seed'])
        neutral_prefix, positive_prefix, groups = prefixes[template_index]
        history = [int(token) for token in backend.continuation(neutral_prefix, int(history_tokens), sample_seed)]
        if len(history) != int(history_tokens):
            raise ValueError('Probe continuation length does not match history_tokens')
        neutral_ids = list(neutral_prefix) + history
        positive_ids = list(positive_prefix) + history
        neutral = backend.hidden(neutral_ids, checkpointing=False)[:, -1].float().reshape(-1).detach().cpu()
        positive = backend.hidden(positive_ids, checkpointing=False)[:, -1].float().reshape(-1).detach().cpu()
        if not torch.isfinite(neutral).all() or not torch.isfinite(positive).all():
            raise FloatingPointError('Non-finite probe teacher state')
        prepared.append(dict(
            template_row=template_index, template_id=f't{template_index}', sample_seed=sample_seed,
            history=history, train_ids=neutral_ids, prefix_len=len(neutral_ids),
            neutral=neutral.contiguous(), positive=positive.contiguous(),
            vocal=groups['vocal'], tempo_bpm=groups['tempo_bpm'], tempo_range=groups['tempo_range'],
        ))
        if cursor % 16 == 0 or cursor == total:
            print(json.dumps(dict(probe='bank', built=cursor, total=total, template=template_index)), flush=True)
    blob = dict(key=key, prepared=prepared, listening=plan['listening'],
                note='listening seeds are reserved and were not forwarded')
    _torch_save(path, blob)
    return blob


@torch.no_grad()
def probe_checkpoint(run, backend, network, templates, training_rows, *, step, state=None,
                     max_seq_len, history_tokens, probe_per_template, listening_per_template,
                     seed, model_id, dummy, stride, critic_scale=None, weights_kind='ema'):
    """Evaluate one EMA milestone and append ``probe.jsonl``. Idempotent per step."""
    run = Path(run)
    step = int(step)
    if weights_kind not in ('ema', 'live'):
        raise ValueError('weights_kind must be ema or live')
    if step < 1 or not needs_probe(run, step, weights_kind):
        summary = build_summary(
            load_probe_jsonl(run / 'probe.jsonl'), stride=stride,
            training_finite=_training_finite(run))
        _write_summary(run / 'probe-summary.json', summary)
        return summary
    device = next(network.parameters()).device
    with preserve_rng(device):
        with network.scaled(0.):
            bank = ensure_bank(
                run / 'prepared-eval.pt', backend, templates, training_rows,
                max_seq_len=max_seq_len, history_tokens=history_tokens,
                probe_per=probe_per_template, listen_per=listening_per_template,
                seed=seed, model_id=model_id, dummy=dummy)
        scale, edit_rms = paired_edit_scale(training_rows)
        if critic_scale is not None and not torch.allclose(
                scale, critic_scale.detach().float().cpu(), rtol=1e-4, atol=1e-5):
            raise ValueError('Probe paired-edit scale does not match the training critic')
        hidden = _vector(bank['prepared'][0], 'neutral')
        if hidden.numel() != scale.numel():
            raise ValueError('Probe hidden width does not match the paired-edit scale')
        fixtures = ensure_fixtures(
            run / 'probe-fixtures.pt', dim=scale.numel(), rows=len(bank['prepared']), scale=scale)
        print(json.dumps(dict(
            probe='start', step=step, weights_kind=weights_kind, examples=len(bank['prepared']),
            edit_rms=edit_rms, eval_sigma=1.0, evaluator='not_trained')), flush=True)
        context = use_state(network, state) if state is not None else nullcontext()
        with context:
            records = _score_bank(
                backend, network, bank['prepared'], fixtures, step=step, weights_kind=weights_kind)
    summary = _commit(run, records, stride=stride)
    at_full = next(record for record in records if round(float(record['strength']), 4) == 1.0)
    print(json.dumps(dict(
        probe='checkpoint', step=step, weights_kind=weights_kind,
        residual_rms=at_full['residual_rms'], residual_p95=at_full['residual_p95'],
        gain_mean=at_full['gain_mean'], teacher_swd=at_full['teacher_swd'],
        game_swd_sigma_1=at_full['game_swd_sigma_1'], evaluator_auc=None,
        stopping=summary['stopping']['status'], best_step=summary['best_step'])), flush=True)
    return summary


def sweep_milestones(run, backend, network, templates, training_rows, *, state_of, **kwargs):
    """Evaluate saved ``*_stepN.safetensors`` files that do not yet have probe records.

    ``state_of(path)`` returns the weight dict. Intended for a later sequential
    sweep when the training GPU is free. Does not train the classifier.
    """
    pending = []
    for step, path in milestone_steps(run):
        if needs_probe(run, step, kwargs.get('weights_kind', 'ema')):
            pending.append((step, path))
    for step, path in pending:
        probe_checkpoint(
            run, backend, network, templates, training_rows,
            step=step, state=state_of(path), **kwargs)
    return build_summary(
        load_probe_jsonl(Path(run) / 'probe.jsonl'), stride=kwargs['stride'],
        training_finite=_training_finite(run))


def stopping_status(grouped, *, stride, training_finite=True):
    base = dict(plateau=False, finished=False, evaluator_auc=None, evaluator_status='not_trained',
                window=None)
    if not training_finite:
        return dict(base, status='unstable',
                    note='Training metrics are not finite. Plateau is not declared.')
    steps = sorted(step for step, group in grouped.items() if _complete(group))
    if len(steps) < 3:
        return dict(base, status='collecting',
                    note='Need three consecutive EMA evaluations before a plateau check.')
    window = steps[-3:]
    if any(window[index] - window[index - 1] != int(stride) for index in (1, 2)):
        return dict(base, status='not_plateaued', window=window,
                    note='The last three evaluations are not evenly spaced by the save interval.')
    required = ('residual_rms', 'residual_p95', 'teacher_swd', 'game_swd_sigma_1', 'gain_mean')
    for step in window:
        record = grouped[step][1.0]
        if any(key not in record for key in required) or not _gains_close(grouped[step]):
            return dict(base, status='not_plateaued', window=window,
                        note='Gain is away from the requested strength, or a probe field is missing.')
    flat = True
    for prev_step, curr_step in zip(window, window[1:]):
        prev = grouped[prev_step][1.0]
        curr = grouped[curr_step][1.0]
        if (_rel(prev['residual_rms'], curr['residual_rms']) >= PLATEAU_RMS_TOL
                or _rel(prev['residual_p95'], curr['residual_p95']) >= PLATEAU_RMS_TOL
                or _rel(prev['teacher_swd'], curr['teacher_swd']) >= PLATEAU_SWD_TOL
                or _rel(prev['game_swd_sigma_1'], curr['game_swd_sigma_1']) >= PLATEAU_SWD_TOL
                or _worsened(prev, curr)):
            flat = False
            break
    if not flat:
        return dict(base, status='not_plateaued', window=window,
                    note='Held-out residual, tail, sliced-Wasserstein, or a subgroup is still moving.')
    return dict(
        base, status='plateau_pending_evaluator', plateau=True, window=window,
        note='Metrics are flat, but the independent evaluator is not trained. Do not treat this as finished.')


def select_checkpoint(grouped):
    """Earliest EMA step within a small band of the best later strength-1.0 metrics."""
    candidates = []
    for step, by_strength in grouped.items():
        if not _complete(by_strength):
            continue
        full = by_strength[1.0]
        if any(key not in full for key in ('residual_rms', 'residual_p95', 'teacher_swd', 'gain_mean')):
            continue
        candidates.append((int(step), full, _gain_error(by_strength), by_strength))
    if not candidates:
        return None
    quality = min(candidates, key=lambda item: (
        float(item[1]['residual_rms']), float(item[1]['residual_p95']),
        float(item[1]['teacher_swd']), float(item[2])))
    band = [item for item in candidates if _within_band(item, quality)]
    chosen = min(band, key=lambda item: item[0])
    return dict(
        step=chosen[0], quality_step=quality[0], record=chosen[1],
        gain_ok=_gains_close(chosen[3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        'Sweep saved EMA milestones into probe.jsonl. Does not train the '
        'independent classifier and should not be run on a GPU that is training.'))
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--prompts_file', type=Path, required=True)
    parser.add_argument('--model_id', default='m-a-p/YuE2-3B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--history_tokens', type=int, default=32)
    parser.add_argument('--probe_seeds', type=int, default=32)
    parser.add_argument('--probe_listening_seeds', type=int, default=4)
    parser.add_argument('--stride', type=int, default=100)
    args = parser.parse_args(argv)
    from safetensors.torch import load_file
    from conceptmod.textsliders.yue2_arm_b import load_prompts
    from conceptmod.textsliders.yue2_backend import YuE2Backend
    from conceptmod.textsliders.yue2_particle_bridge import ParticleSlider

    prepared_path = args.save_dir / 'prepared.pt'
    if not prepared_path.exists():
        raise SystemExit(f'Missing training bank {prepared_path}')
    templates, _meta = load_prompts(args.prompts_file)
    training_rows = torch.load(prepared_path, map_location='cpu', weights_only=True)['prepared']
    backend = YuE2Backend(args.model_id, device=args.device, dummy=args.dummy)
    network = ParticleSlider(backend.model)
    device = next(network.parameters()).device

    def state_of(path):
        return {key: value.to(device) for key, value in load_file(str(path), device='cpu').items()}

    summary = sweep_milestones(
        args.save_dir, backend, network, templates, training_rows, state_of=state_of,
        max_seq_len=args.max_seq_len, history_tokens=args.history_tokens,
        probe_per_template=args.probe_seeds, listening_per_template=args.probe_listening_seeds,
        seed=args.seed, model_id=args.model_id, dummy=args.dummy, stride=args.stride,
        weights_kind='ema')
    print(json.dumps(dict(
        probe='sweep_done', best_step=summary['best_step'],
        stopping=summary['stopping']['status'], evaluator='not_trained')))
    return 0


def _score_bank(backend, network, rows, fixtures, *, step, weights_kind):
    neutral = torch.stack([_vector(row, 'neutral') for row in rows])
    positive = torch.stack([_vector(row, 'positive') for row in rows])
    groups = [_label(row, index) for index, row in enumerate(rows)]
    records = []
    for strength in STRENGTHS:
        print(json.dumps(dict(probe='predict', step=step, strength=strength, rows=len(rows))), flush=True)
        prediction = _predict(backend, network, rows, strength)
        metrics = score_heldout(
            neutral, positive, prediction, fixtures['paired_edit_scale'],
            fixtures['projections'], fixtures['noise'], strength=strength, groups=groups)
        records.append(dict(schema=1, step=int(step), weights_kind=weights_kind, **metrics))
    return records


@torch.no_grad()
def _predict(backend, network, rows, strength):
    predictions = []
    with network.scaled(float(strength)):
        for index, row in enumerate(rows):
            if index % 32 == 0:
                print(json.dumps(dict(
                    probe='predict_row', strength=float(strength), row=index, rows=len(rows))), flush=True)
            hidden = backend.hidden(row['train_ids'], checkpointing=False)
            predictions.append(hidden[:, int(row['prefix_len']) - 1].float().reshape(-1).detach().cpu())
    return torch.stack(predictions)


def _label(row, index):
    bpm = row.get('tempo_bpm')
    return dict(
        template=str(row.get('template_id', f't{row.get("template_row", index)}')),
        vocal=str(row.get('vocal', 'unknown')),
        tempo_range=str(row.get('tempo_range', 'unknown')),
        sample_seed=int(row['sample_seed']),
        template_row=int(row.get('template_row', index)),
        tempo_bpm=None if bpm is None else int(bpm),
    )


def _subgroups(normalized, predicted_norm, teacher_norm, projections, groups,
               global_median, global_swd, row_rms):
    families = ('template', 'vocal', 'tempo_range')
    report = {family: {} for family in families}
    median_floor = max(float(global_median), 1e-6)
    swd_floor = max(float(global_swd), 1e-6)
    for family in families:
        keys = []
        for group in groups:
            key = str(group[family])
            if key not in keys:
                keys.append(key)
        for key in keys:
            mask = torch.tensor([str(group[family]) == key for group in groups], dtype=torch.bool)
            if int(mask.sum()) < 1:
                continue
            subset = normalized[mask]
            chosen_rows = subset.pow(2).mean(-1).sqrt()
            stats = _distribution(chosen_rows)
            distance = sliced_wasserstein(predicted_norm[mask], teacher_norm[mask], projections)
            median = stats['median']
            residual_rms = float(subset.pow(2).mean().sqrt())
            flagged = bool(
                median > FLAG_RATIO * median_floor
                or residual_rms > FLAG_RATIO * median_floor
                or distance['mean'] > FLAG_RATIO * swd_floor)
            report[family][key] = dict(
                n=int(mask.sum()), residual_rms=residual_rms, residual_median=median,
                residual_p95=stats['p95'], teacher_swd=distance['mean'],
                ratio_to_global_median=float(median / median_floor), flagged=flagged)
    flagged_seeds = []
    for index, group in enumerate(groups):
        ratio = float(row_rms[index]) / median_floor
        if ratio > FLAG_RATIO:
            flagged_seeds.append(dict(
                template=group.get('template'), template_row=group.get('template_row'),
                sample_seed=group.get('sample_seed'), vocal=group.get('vocal'),
                tempo_range=group.get('tempo_range'), tempo_bpm=group.get('tempo_bpm'),
                residual_rms=float(row_rms[index]), ratio_to_median=ratio))
    report['flagged_seeds'] = flagged_seeds
    return report


def _distribution(values):
    values = values.detach().float().flatten()
    if values.numel() < 1:
        raise ValueError('Empty probe distribution')
    return dict(
        mean=float(values.mean()),
        median=float(torch.quantile(values, 0.5)),
        p90=float(torch.quantile(values, 0.9)),
        p95=float(torch.quantile(values, 0.95)),
        maximum=float(values.max()),
    )


def _group_ema(records):
    grouped = {}
    for record in records:
        if record.get('weights_kind') != 'ema':
            continue
        grouped.setdefault(int(record['step']), {})[round(float(record['strength']), 4)] = record
    return grouped


def _complete(by_strength):
    return all(round(strength, 4) in by_strength for strength in STRENGTHS)


def _gain_close(gain, strength):
    return abs(float(gain) - float(strength)) <= max(GAIN_ABS_TOL, GAIN_REL_TOL * float(strength))


def _gains_close(by_strength):
    return all(_gain_close(by_strength[round(strength, 4)]['gain_mean'], strength) for strength in STRENGTHS)


def _gain_error(by_strength):
    return sum(abs(float(by_strength[round(strength, 4)]['gain_mean']) - float(strength))
               for strength in STRENGTHS) / len(STRENGTHS)


def _rel(prev, curr):
    return abs(float(curr) - float(prev)) / max(abs(float(prev)), 1e-8)


def _worsened(prev, curr):
    before = _subgroup_rms(prev)
    after = _subgroup_rms(curr)
    for key, value in after.items():
        prior = before.get(key)
        if prior is None:
            continue
        if value > prior * (1.0 + SUBGROUP_WORSE) and value > prior + 1e-8:
            return True
    return False


def _subgroup_rms(record):
    found = {}
    subgroups = record.get('subgroups') or {}
    for family in ('template', 'vocal', 'tempo_range'):
        for key, stats in (subgroups.get(family) or {}).items():
            if 'residual_rms' in stats:
                found[f'{family}:{key}'] = float(stats['residual_rms'])
    return found


def _within_band(item, quality):
    value, best = item[1], quality[1]
    checks = (
        ('residual_rms', SELECTION_RMS_TOL),
        ('residual_p95', SELECTION_RMS_TOL),
        ('teacher_swd', SELECTION_SWD_TOL),
    )
    for key, tolerance in checks:
        best_value = float(best[key])
        limit = max(best_value * (1.0 + tolerance), best_value + SELECTION_ABS)
        if float(value[key]) > limit + 1e-12:
            return False
    return True


def _public_metrics(record):
    keys = (
        'step', 'weights_kind', 'strength', 'examples', 'residual_rms', 'residual_p95',
        'gain_mean', 'gain_p95_error', 'orthogonal_error', 'teacher_swd',
        'game_swd_sigma_1', 'evaluator_auc', 'subgroups',
    )
    return {key: record.get(key) for key in keys}


def _commit(run, records, *, stride):
    merged = _upsert_jsonl(Path(run) / 'probe.jsonl', records)
    summary = build_summary(merged, stride=stride, training_finite=_training_finite(run))
    _write_summary(Path(run) / 'probe-summary.json', summary)
    return summary


def _upsert_jsonl(path, new_records):
    current = load_probe_jsonl(path)
    index = {}
    for record in list(current) + list(new_records):
        ident = (int(record['step']), record.get('weights_kind', 'ema'), round(float(record['strength']), 4))
        index[ident] = record
    ordered = [index[key] for key in sorted(index)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(record, allow_nan=False) + '\n' for record in ordered))
    temporary.replace(path)
    return ordered


def _write_summary(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _training_finite(run):
    path = Path(run) / 'progress.json'
    if not path.exists():
        return True
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    for key in ('loss', 'g_adv', 'd_loss', 'particle_vic', 'grad_norm', 'cos_pos'):
        if key not in record:
            continue
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def _validate_fixtures(blob, dim, rows, scale):
    projections = blob.get('projections')
    noise = blob.get('noise')
    saved = blob.get('paired_edit_scale')
    if projections is None or noise is None or saved is None:
        raise ValueError('probe-fixtures.pt is missing projections, noise, or scale')
    if tuple(projections.shape) != (PROJECTION_COUNT, int(dim)):
        raise ValueError('Saved probe projections do not match this hidden width')
    if tuple(noise.shape) != (int(rows), int(dim)):
        raise ValueError(
            f"probe-fixtures.pt has {tuple(noise.shape)} noise, this bank needs {(int(rows), int(dim))}")
    if tuple(saved.shape) != (int(dim),):
        raise ValueError('Saved paired-edit scale width does not match')
    if not torch.isfinite(projections).all() or not torch.allclose(
            projections.norm(dim=-1), torch.ones(PROJECTION_COUNT), atol=1e-4):
        raise ValueError('Probe projections are not finite unit vectors')
    if not torch.isfinite(noise).all() or not torch.isfinite(saved).all() or not torch.all(saved > 0):
        raise ValueError('Probe noise or paired-edit scale is invalid')
    if not torch.allclose(saved.float(), scale.float(), rtol=1e-4, atol=1e-5):
        raise ValueError('Saved paired-edit scale does not match this training bank')


def _torch_save(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(blob, temporary)
    temporary.replace(path)


def _template_sha(templates):
    digest = hashlib.sha256()
    for row in templates:
        for key in ('neutral', 'positive', 'lyrics'):
            digest.update(key.encode())
            digest.update(b'\0')
            digest.update(str(row[key]).encode())
            digest.update(b'\0')
    return digest.hexdigest()


def _vector(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            tensor = row[key].detach().float().cpu().reshape(-1)
            if tensor.ndim != 1:
                raise ValueError(f'{key} must be a hidden vector')
            return tensor.contiguous()
    raise KeyError(keys)


def _reject_nonfinite(record):
    def walk(value):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise FloatingPointError('Non-finite probe metric')
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(record)


if __name__ == '__main__':
    raise SystemExit(main())
