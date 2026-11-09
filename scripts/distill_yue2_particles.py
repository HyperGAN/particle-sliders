#!/usr/bin/env python3
"""Calibrate ordinary rank-8 LoRAs against frozen YuE2 particle teachers.

This is an approximate, data-dependent distillation, not an algebraic merge.
The teacher's up matrices are retained; its nonlinear rank-8 features are
regressed on real projection inputs with a homogeneous ridge regression.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT), str(ROOT.parent)]
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from particle_sliders.textsliders.yue2_backend import YuE2Slider, attention_targets, file_digest
from app.rewriter import _artist_name_hit
import distillation_artifacts as artifacts


def cohort_identity(args, pipe, model, release, spec, calibration_rows):
    """Bind the effective historical protocol without changing its objective."""
    weights = getattr(pipe, 'weights', None)
    if not isinstance(weights, dict):
        raise ValueError('Verified full model weight/config identity is required')
    for kind in ('mot', 'vae'):
        record = weights.get(kind, {})
        if not isinstance(record, dict):
            raise ValueError('Verified full model weight/config identity is required')
        files = record.get('files', {})
        if not isinstance(files, dict) or not files:
            raise ValueError('Verified full model weight/config identity is required')
        artifacts.require_sha256(record.get('config_sha256'), kind + ' configuration')
        for name, item in files.items():
            if not isinstance(item, dict) or type(item.get('bytes')) is not int or item['bytes'] < 1:
                raise ValueError('Verified full model weight/config identity is required')
            artifacts.require_sha256(item.get('sha256'), kind + '/' + name)
    artifacts.require_sha256(getattr(pipe, 'runtime_sha256', None), 'YuE2 runtime source')
    core = Path(sys.modules['particle_sliders'].__file__).resolve().parent
    sources = {f'core/{path.relative_to(core)}': path for path in core.rglob('*.py')}
    sources.update(distill=Path(__file__), artifacts=Path(artifacts.__file__),
                   rewriter=Path(sys.modules[_artist_name_hit.__module__].__file__))
    source = release / spec['weights']
    teacher = artifacts.file_identity(source)
    if teacher['sha256'] != spec['sha256']:
        raise ValueError('Teacher checkpoint content differs from the declared release')
    references = {}
    for row in range(2):
        for kind in ('off', 'metal'):
            for name in ('prefix.npy', 'semantic.npy'):
                label = f'row-{row}-seed-1709/{kind}/{name}'
                references[label] = artifacts.file_identity(args.reference / spec['id'] / label)
    return artifacts.snapshot(dict(protocol='historical-yue2-distillation-v1', qualification='historical-protocol-only',
        source=artifacts.source_identity(sources), runtime_sha256=pipe.runtime_sha256,
        versions=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                      yaml=yaml.__version__, safetensors=importlib.metadata.version('safetensors')),
        model=dict(weights=weights, config=model.config.to_dict(), training=model.training,
                   parameter_layout={name: dict(shape=list(parameter.shape), dtype=str(parameter.dtype),
                                                device=str(parameter.device))
                                     for name, parameter in model.named_parameters()},
                   tokenizer=artifacts.file_identity(Path(pipe.model_dir)/'qwen.tiktoken')),
        execution=dict(gpu=args.gpu, backend=pipe.backend, quantization=pipe.quantization,
            offload_ar=pipe.offload_ar, vae_core_frames=pipe.vae_core_frames,
            generation_config=pipe.generation_config.to_dict(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            default_dtype=str(torch.get_default_dtype()), threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), cuda_build=torch.version.cuda,
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
            cublas_workspace_config=os.getenv('CUBLAS_WORKSPACE_CONFIG'),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        release_catalog=artifacts.file_identity(release/'catalog.json'), spec=spec,
        teacher=teacher, prompts={split: artifacts.file_identity(release/spec['prompts'][split])
                                 for split in ('train', 'eval')},
        calibration_rows=calibration_rows, references=references))


def fit_identity(args, pipe, model, release, spec, calibration_rows, train_tokens):
    return dict(stage='ridge', cohort=cohort_identity(args, pipe, model, release, spec, calibration_rows),
        settings=dict(train_tokens=train_tokens, initialization_seed=9182, rollout_seed=4100,
            prefix_seed=7, heldout_seed=1709, ridges=[1e-4, 1e-3, 1e-2],
            calibration_scales=[0., 1.], comparison_scales=[.5, 1.], rank=8, alpha=8.))


def fit_outputs(out, concept, count):
    checkpoint = out/f'{concept}_distilled_rank8.safetensors'
    return [checkpoint, checkpoint.with_suffix('.json'),
            *(out/f'train-rollout-{index}.json' for index in range(count))]


def verify_fit(args, pipe, model, spec, calibration_rows):
    out = args.output/spec['id']
    saved = artifacts.read_json(out/'distillation-identity.json')
    try:
        train_tokens = saved['identity']['settings']['train_tokens']
        if type(train_tokens) is not int or train_tokens < 1:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError('First-stage distillation has no valid effective token budget; retain its evidence') from None
    identity = fit_identity(args, pipe, model, args.release, spec, calibration_rows, train_tokens)
    artifacts.bind_identity(out/'distillation-identity.json', identity)
    artifacts.verify_completion(out/'evaluation.json', identity,
                               fit_outputs(out, spec['id'], len(rows(args.release, spec, 'train'))))
    return identity


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def log(event, **values):
    print(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          event=event, **values)), flush=True)


def detach(model, network):
    for name, module in attention_targets(model).items():
        module.forward = network.adapters[name.replace('.', '-')].org_forward


@contextmanager
def attached(model, network, scale):
    targets = attention_targets(model)
    originals = {name: module.forward for name, module in targets.items()}
    try:
        if network is not None:
            for name, module in targets.items():
                module.forward = network.adapters[name.replace('.', '-')].forward
            with network.scaled(scale):
                yield
        else:
            yield
    finally:
        for name, module in targets.items():
            module.forward = originals[name]


def hidden(model, ids):
    tokens = torch.tensor([ids], device=next(model.parameters()).device)
    positions = torch.arange(len(ids), device=tokens.device)[None]
    return model.model(input_ids=tokens, position_ids=positions, use_cache=False)[0].float()


def rows(release, spec, split):
    result = yaml.safe_load((release / spec['prompts'][split]).read_text())['rows']
    for row in result:
        if _artist_name_hit('', row['neutral'] + '\n' + row['positive'], row['lyrics']):
            raise ValueError('Named music reference in source prompts; do not distill these weights')
    return result


def make_plan(pipe, row, seed):
    from yue2.protocol import SongRequest
    return pipe.plan(request=SongRequest(style=row['neutral'], lyrics=row['lyrics'], cot='off', seed=seed))


def rollout_identity(pipe, row, seed, count, cohort, prefix):
    return dict(stage='teacher-rollout', cohort=cohort, row=row, prefix=prefix,
                seed=seed, max_tokens=count, min_tokens=min(200, count), scale=1.)


def load_rollout(pipe, row, seed, count, cache, cohort):
    prefix = make_plan(pipe, row, seed).prefix
    identity = rollout_identity(pipe, row, seed, count, cohort, prefix)
    saved = artifacts.verify_record(cache, identity)
    if saved.get('prefix') != prefix or saved.get('seed') != seed or saved.get('max_tokens') != count:
        raise ValueError(f'Rollout protocol differs; retain evidence at {cache}')
    ids = saved.get('ids')
    if not isinstance(ids, list) or any(type(token) is not int for token in ids) or ids[:len(prefix)] != prefix:
        raise ValueError(f'Rollout tokens differ from their prefix; retain evidence at {cache}')
    return ids


def rollout(pipe, model, network, row, seed, count, cache, *, cohort):
    from yue2.protocol import CODEC_OFFSET
    plan = make_plan(pipe, row, seed)
    if cache.exists():
        return load_rollout(pipe, row, seed, count, cache, cohort)
    with attached(model, network, 1):
        semantic = pipe.generate_semantic(plan, sampling={'max_tokens': count, 'min_tokens': min(200, count)})
    ids = plan.prefix + [int(t) + CODEC_OFFSET for t in semantic.tokens]
    artifacts.write_new_json(cache, artifacts.seal(
        dict(prefix=plan.prefix, ids=ids, seed=seed, max_tokens=count,
             truncated=semantic.truncated, timing=semantic.timing),
        rollout_identity(pipe, row, seed, count, cohort, plan.prefix)))
    return ids


class Calibration:
    """One covariance per QKV input and per O input; no activation dump."""
    def __init__(self, model, teacher):
        self.groups = {}
        self.handles = []
        targets = attention_targets(model)
        for name, module in targets.items():
            projection = name.rsplit('.', 1)[1]
            if projection not in ('q_proj', 'o_proj'):
                continue
            names = ([name.rsplit('.', 1)[0] + '.' + p for p in ('q_proj', 'k_proj', 'v_proj')]
                     if projection == 'q_proj' else [name])
            adapters = [teacher.adapters[n.replace('.', '-')] for n in names]
            dim = module.in_features
            device = module.weight.device
            group = dict(names=names, xx=torch.zeros(dim, dim, device=device),
                         xy=torch.zeros(dim, 8 * len(names), device=device), count=0)
            self.groups[name] = group

            def collect(_module, inputs, group=group, adapters=adapters):
                x = inputs[0].reshape(-1, dim).float()
                y = torch.cat([a.bridge(a.lora_down(x), teacher.particles) for a in adapters], dim=-1)
                group['xx'].addmm_(x.T, x)
                group['xy'].addmm_(x.T, y)
                group['count'] += x.shape[0]

            self.handles.append(module.register_forward_pre_hook(collect))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def solve(self, student, teacher, ridge):
        for group in self.groups.values():
            xx, xy = group['xx'], group['xy']
            norm = xx.diag().clamp_min(1e-12).sqrt()
            correlation = xx / norm[:, None] / norm[None, :]
            correlation.diagonal().add_(ridge)
            chol = torch.linalg.cholesky(correlation)
            beta = torch.cholesky_solve(xy / norm[:, None], chol) / norm[:, None]
            for index, name in enumerate(group['names']):
                key = name.replace('.', '-')
                student.adapters[key].lora_down.weight.copy_(beta[:, index*8:(index+1)*8].T)
                student.adapters[key].lora_up.weight.copy_(teacher.adapters[key].lora_up.weight)


def compare(model, teacher, student, sequences, *, logits=False):
    records = []
    for label, ids, prefix_len in sequences:
        base = hidden(model, ids)
        with attached(model, student, 0):
            if not torch.equal(base, hidden(model, ids)):
                raise AssertionError('Scale zero changed the base model')
        for scale in (.5, 1.):
            with attached(model, teacher, scale):
                target = hidden(model, ids)
            with attached(model, student, scale):
                prediction = hidden(model, ids)
            regions = {'boundary': slice(prefix_len-1, prefix_len),
                       'music': slice(prefix_len-1, None)}
            for region, positions in regions.items():
                shift = target[:, positions] - base[:, positions]
                estimated = prediction[:, positions] - base[:, positions]
                error = (estimated - shift).square().sum()
                energy = shift.square().sum().clamp_min(1e-12)
                metric = dict(sequence=label, scale=scale, region=region,
                    relative_steering_mse=float(error / energy),
                    steering_cosine=float(F.cosine_similarity(estimated.flatten(), shift.flatten(), dim=0)),
                    steering_norm_ratio=float(estimated.norm() / shift.norm().clamp_min(1e-12)))
                if logits and region == 'music':
                    # Native semantic distribution, before sampling penalties/CFG.
                    from yue2.protocol import CODEC_OFFSET
                    positions = torch.arange(prefix_len-1, len(ids), 32, device=base.device)
                    weight = model.lm_head.weight[CODEC_OFFSET:CODEC_OFFSET+32768].float()
                    tl = F.log_softmax(target[0, positions] @ weight.T, dim=-1)
                    sl = F.log_softmax(prediction[0, positions] @ weight.T, dim=-1)
                    bl = F.log_softmax(base[0, positions] @ weight.T, dim=-1)
                    metric['teacher_student_kl'] = float((tl.exp() * (tl-sl)).sum(-1).mean())
                    metric['teacher_base_kl'] = float((tl.exp() * (tl-bl)).sum(-1).mean())
                records.append(metric)
    return records


def existing_eval(root, concept, row, seed):
    from yue2.protocol import CODEC_OFFSET
    result = []
    for kind in ('off', 'metal'):
        path = root / concept / f'row-{row}-seed-{seed}' / kind
        prefix = np.load(path / 'prefix.npy').reshape(-1).tolist()
        tokens = np.load(path / 'semantic.npy').reshape(-1).tolist()
        result.append((f'row{row}-{kind}-seed{seed}', prefix + [int(t)+CODEC_OFFSET for t in tokens], len(prefix)))
    return result


def fit_one(args, pipe, model, release, spec, calibration_rows):
    with artifacts.attempt_lock(args.output/spec['id']):
        return _fit_one(args, pipe, model, release, spec, calibration_rows)


def _fit_one(args, pipe, model, release, spec, calibration_rows):
    concept = spec['id']
    out = args.output / concept
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f'{concept}_distilled_rank8.safetensors'
    identity = fit_identity(args, pipe, model, release, spec, calibration_rows, args.train_tokens)
    artifacts.bind_identity(out/'distillation-identity.json', identity,
        evidence=[path for path in out.iterdir() if path.name != '.distillation.lock'])
    train, test = rows(release, spec, 'train'), rows(release, spec, 'eval')
    if (out / 'evaluation.json').exists():
        artifacts.verify_completion(out/'evaluation.json', identity, fit_outputs(out, concept, len(train)))
        log('skip_completed', concept=concept)
        return
    if checkpoint.exists() or checkpoint.with_suffix('.json').exists():
        raise ValueError(f'Uncommitted distillation export exists; retain evidence and use a new output directory: {out}')
    source = release / spec['weights']
    assert file_digest(source) == spec['sha256']
    teacher, metadata = YuE2Slider.load(model, source)
    detach(model, teacher)
    teacher.requires_grad_(False)
    assert metadata['model_identity'] == pipe.weights['mot']
    assert not {r['lyrics'] for r in train} & {r['lyrics'] for r in test}
    validation_lyrics = train[-1]['lyrics']
    fitting_rows = [r for r in calibration_rows if r['lyrics'] != validation_lyrics]
    log('rollouts', concept=concept, gpu=args.gpu, prefix_rows=len(fitting_rows))
    sequences = []
    for index, row in enumerate(train):
        ids = rollout(pipe, model, teacher, row, 4100+index, args.train_tokens,
                      out/f'train-rollout-{index}.json', cohort=identity)
        sequences.append((f'train-{index}', ids, len(make_plan(pipe, row, 4100+index).prefix)))
        log('rollout_complete', concept=concept, row=index, tokens=len(ids))
    student = YuE2Slider(model, rank=8, alpha=8.)
    detach(model, student)
    student.requires_grad_(False)
    stats = Calibration(model, teacher)
    try:
        for scale in (0., 1.):
            with attached(model, teacher, scale):
                for index, row in enumerate(fitting_rows):
                    hidden(model, make_plan(pipe, row, 7).prefix)
                for _, ids, _ in sequences[:-1]:
                    hidden(model, ids)
            log('calibration_pass', concept=concept, scale=scale,
                tokens=next(iter(stats.groups.values()))['count'])
    finally:
        stats.close()
    scores = []
    for ridge in (1e-4, 1e-3, 1e-2):
        stats.solve(student, teacher, ridge)
        metrics = compare(model, teacher, student, [sequences[-1]])
        score = sum(m['relative_steering_mse'] for m in metrics) / len(metrics)
        scores.append(dict(ridge=ridge, score=score, metrics=metrics))
        log('validation', concept=concept, ridge=ridge, relative_steering_mse=score)
    selected = min(scores, key=lambda s: s['score'])
    stats.solve(student, teacher, selected['ridge'])
    token_count = next(iter(stats.groups.values()))['count']
    del stats
    gc.collect()
    torch.cuda.empty_cache()
    record = dict(model_id=metadata['model_id'], model_identity=metadata['model_identity'],
        concept=concept, recipe='particle-to-linear-ridge-distillation-v1',
        teacher_sha256=spec['sha256'], teacher_format=metadata['format'],
        ridge=selected['ridge'], calibration_tokens=token_count,
        calibration_prompt_rows=len(fitting_rows), calibration_scales=[0, 1],
        train_rollout_tokens=args.train_tokens, validation_train_row=3,
        evaluation_lyrics_excluded=True, seed=4100, gpu=args.gpu,
        rank=8, alpha=8., ordinary_lora=True, approximate=True,
        inference_stages=['semantic'], acoustic_scale=0,
        method='Fix teacher U; solve min_A ||X A^T - routed_teacher(X)||^2 + normalized ridge',
        source_script_sha256=file_digest(__file__))
    student.save(checkpoint, record)
    # Validate the actual exported loader and all tensors before evaluating.
    exported, exported_record = YuE2Slider.load(model, checkpoint)
    detach(model, exported)
    assert exported_record['format'] == 'conceptmod-yue2-ar-v1'
    assert all('bridge' not in k and 'particles' not in k for k in exported.state_dict())
    assert all(torch.equal(v, exported.state_dict()[k]) for k, v in student.state_dict().items())
    del student
    heldout = []
    for row in range(2):
        found = existing_eval(args.reference, concept, row, 1709)
        expected = make_plan(pipe, test[row], 1709).prefix
        assert all(ids[:n] == expected for _, ids, n in found)
        heldout.extend(found)
    metrics = compare(model, teacher, exported, heldout, logits=True)
    artifacts.publish_completion(out / 'evaluation.json', identity, dict(concept=concept, checkpoint=checkpoint.name,
        checkpoint_sha256=file_digest(checkpoint), teacher_sha256=spec['sha256'],
        validation=scores, selected_ridge=selected['ridge'], heldout=metrics,
        zero_exact=True, roundtrip_exact=True, ordinary_lora_only=True),
        fit_outputs(out, concept, len(train)))
    log('fit_complete', concept=concept, checkpoint=str(checkpoint),
        heldout_music_mse=float(np.mean([m['relative_steering_mse'] for m in metrics if m['region']=='music'])))
    del teacher, exported
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path, default=ROOT.parent/'releases/yue2-concept-sliders/model')
    parser.add_argument('--reference', type=Path, default=ROOT/'eval/listen/yue2-particle-uni16-1200-20260917')
    parser.add_argument('--train-tokens', type=int, default=384)
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(4)
    torch.manual_seed(9182)
    torch.cuda.set_device(args.gpu)
    catalog = json.loads((args.release / 'catalog.json').read_text())
    specs = {s['id']: s for s in catalog['sliders']}
    all_rows = {}
    for spec in specs.values():
        for row in rows(args.release, spec, 'train'):
            all_rows[(row['neutral'], row['lyrics'])] = row
    from yue2 import YuE2Pipeline
    with torch.inference_mode(), YuE2Pipeline.from_pretrained(
            catalog['base_model'], revision=catalog['base_revision'],
            vae=catalog['vae'], vae_revision=catalog['vae_revision'],
            cache_dir=str(ROOT.parent/'.cache/huggingface/hub'), local_files_only=True,
            device=f'cuda:{args.gpu}', backend='torch', quantization='none',
            offload_ar=False, memory_budget_gib=26, progress=False) as pipe:
        model = pipe._load_model().requires_grad_(False)
        log('model_loaded', gpu=args.gpu, ids=args.ids)
        for concept in args.ids.split(','):
            try:
                fit_one(args, pipe, model, args.release, specs[concept], list(all_rows.values()))
            except Exception:
                import traceback
                artifacts.write_new_json(args.output / concept / f'error-{time.time_ns()}.json',
                                         dict(traceback=traceback.format_exc()))
                raise


if __name__ == '__main__':
    main()
