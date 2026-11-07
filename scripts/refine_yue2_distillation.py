#!/usr/bin/env python3
"""Refine distilled LoRAs on teacher hidden states and render held-out audio."""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import time

import distill_yue2_particles as d
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_hidden(model, ids):
    tokens = torch.tensor([ids], device=next(model.parameters()).device)
    positions = torch.arange(len(ids), device=tokens.device)[None]
    backbone = model.model
    x = backbone.embed_tokens(tokens)
    cos, sin = backbone.rotary_emb(positions)
    for layer in backbone.layers:
        x = checkpoint(layer, x, cos, sin, use_reentrant=False)
    return backbone.norm(x).float()


def score(metrics):
    return float(np.mean([m['relative_steering_mse'] for m in metrics]))


def refinement_identity(args, pipe, model, spec, all_rows):
    fit = d.verify_fit(args, pipe, model, spec, all_rows)
    return dict(stage='refinement', fit=fit,
        fit_evaluation=d.artifacts.file_identity(args.output/spec['id']/'evaluation.json'),
        source=d.artifacts.file_identity(__file__),
        settings=dict(steps=args.steps, initialization_seed=9182, sample_seed=9918,
            optimizer='Adam', learning_rate=1e-4, clip_grad_norm=1.,
            scales=[.5, 1.], validation_every=10))


def refined_outputs(out, concept):
    target = out/f'{concept}_distilled_refined_rank8.safetensors'
    return [target, target.with_suffix('.json')]


def refine(args, pipe, model, spec, all_rows):
    with d.artifacts.attempt_lock(args.output/spec['id']):
        return _refine(args, pipe, model, spec, all_rows)


def _refine(args, pipe, model, spec, all_rows):
    concept = spec['id']
    out = args.output / concept
    target = out / f'{concept}_distilled_refined_rank8.safetensors'
    identity = refinement_identity(args, pipe, model, spec, all_rows)
    d.artifacts.bind_identity(out/'refinement-identity.json', identity,
        evidence=[out/'refined-evaluation.json', *refined_outputs(out, concept)])
    if (out/'refined-evaluation.json').exists():
        d.artifacts.verify_completion(out/'refined-evaluation.json', identity, refined_outputs(out, concept))
        return target
    if any(path.exists() for path in refined_outputs(out, concept)):
        raise ValueError(f'Uncommitted refinement export exists; retain evidence and use a new output directory: {out}')
    source = args.release/spec['weights']
    assert d.file_digest(source) == spec['sha256']
    teacher, metadata = d.YuE2Slider.load(model, source)
    d.detach(model, teacher)
    teacher.requires_grad_(False)
    student, record = d.YuE2Slider.load(model, out/f'{concept}_distilled_rank8.safetensors')
    d.detach(model, student)
    train = d.rows(args.release, spec, 'train')
    fitting_rows = [row for row in all_rows if row['lyrics'] != train[-1]['lyrics']]
    sequences = []
    for index, row in enumerate(train):
        ids = d.load_rollout(pipe, row, 4100+index, identity['fit']['settings']['train_tokens'],
                             out/f'train-rollout-{index}.json', identity['fit'])
        sequences.append((f'train-{index}', ids, len(d.make_plan(pipe, row, 4100+index).prefix)))
    with torch.no_grad():
        baseline = d.compare(model, teacher, student, [sequences[-1]])
    best_score, best_step = score(baseline), 0
    best = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
    history = [dict(step=0, score=best_score, metrics=baseline)]
    examples = sequences[:-1] + [(f'prefix-{i}', d.make_plan(pipe, row, 7).prefix,
                                 len(d.make_plan(pipe, row, 7).prefix)) for i,row in enumerate(fitting_rows)]
    cached = {}
    with torch.no_grad():
        for label, ids, prefix in examples:
            base = d.hidden(model, ids)
            for scale in (.5, 1.):
                with d.attached(model, teacher, scale):
                    wanted = d.hidden(model, ids)
                cached[(label, scale)] = (wanted.cpu(), base.cpu())
    student.requires_grad_(True)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    rng = random.Random(9918)
    d.log('refine_start', concept=concept, gpu=args.gpu, baseline=best_score)
    for step in range(1, args.steps+1):
        # Half the updates include actual semantic continuations; half broaden
        # the captions. The fourth lyric sheet is reserved for model selection.
        pool = examples[:3] if step % 2 else examples[3:]
        label, ids, prefix = rng.choice(pool)
        scale = rng.choice((.5, 1.))
        wanted, base = (t.to(next(model.parameters()).device) for t in cached[(label, scale)])
        optimizer.zero_grad(set_to_none=True)
        with d.attached(model, student, scale):
            prediction = checkpoint_hidden(model, ids)
            loss = torch.zeros((), device=prediction.device)
            for positions in (slice(prefix-1, prefix), slice(prefix-1, None)):
                energy = (wanted[:, positions]-base[:, positions]).square().mean().clamp_min(1e-7)
                loss = loss + (prediction[:, positions]-wanted[:, positions]).square().mean() / energy / 2
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite distillation loss')
            # Keep the adapters installed until checkpoint recomputation ends.
            loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                metrics = d.compare(model, teacher, student, [sequences[-1]])
            value = score(metrics)
            history.append(dict(step=step, score=value, metrics=metrics, train_loss=float(loss)))
            if value < best_score:
                best_score, best_step = value, step
                best = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
            d.log('refine_validation', concept=concept, step=step, score=value, best_step=best_step)
    student.load_state_dict(best)
    student.requires_grad_(False)
    record.update(recipe='particle-to-linear-ridge-and-hidden-distillation-v1',
        refinement_steps=args.steps, selected_step=best_step, learning_rate=1e-4,
        selection='lowest hidden steering MSE on reserved training lyric sheet',
        refinement_script_sha256=d.file_digest(__file__))
    student.save(target, record)
    heldout = []
    for row in range(2):
        heldout.extend(d.existing_eval(args.reference, concept, row, 1709))
    with torch.no_grad():
        metrics = d.compare(model, teacher, student, heldout, logits=True)
    d.artifacts.publish_completion(out/'refined-evaluation.json', identity, dict(concept=concept,
        checkpoint=target.name, checkpoint_sha256=d.file_digest(target),
        history=history, selected_step=best_step, heldout=metrics,
        zero_exact=True, ordinary_lora_only=True), refined_outputs(out, concept))
    d.log('refine_complete', concept=concept, selected_step=best_step,
        heldout_music_mse=float(np.mean([m['relative_steering_mse'] for m in metrics if m['region']=='music'])))
    del optimizer, teacher, student, best, cached, loss, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return target


@torch.inference_mode()
def audio(args, pipe, model, spec, target, all_rows):
    with d.artifacts.attempt_lock(args.output/spec['id']/'audio'):
        generation_config = pipe.generation_config
        try:
            return _audio(args, pipe, model, spec, target, all_rows)
        finally:
            pipe.generation_config = generation_config


def _audio(args, pipe, model, spec, target, all_rows):
    import soundfile as sf
    from yue2.modeling_vae import YuE2VAE
    concept = spec['id']
    out = args.output/concept/'audio'
    out.mkdir(exist_ok=True)
    refinement = refinement_identity(args, pipe, model, spec, all_rows)
    d.artifacts.bind_identity(out.parent/'refinement-identity.json', refinement,
                             evidence=[out.parent/'refined-evaluation.json'])
    d.artifacts.verify_completion(out.parent/'refined-evaluation.json', refinement,
                                 refined_outputs(out.parent, concept))
    identity = dict(stage='audio', refinement=refinement,
        refinement_evaluation=d.artifacts.file_identity(out.parent/'refined-evaluation.json'),
        checkpoint=d.artifacts.file_identity(target),
        settings=dict(generation_config=replace(pipe.generation_config, ode_steps=16).to_dict(),
            max_tokens=500, min_tokens=200, seed=1709, scale=1., sample_rate=48000,
            subtype='PCM_16', halo_frames=16, soundfile=sf.__version__,
            libsndfile=sf.__libsndfile_version__))
    outputs = [out/f'row{index}-{kind}{suffix}'
        for index, _ in enumerate(d.rows(args.release, spec, 'eval'))
        for kind in ('off', 'teacher', 'plain-ar', 'plain-all')
        for suffix in ('.flac', '-semantic.npy', '.json')]
    d.artifacts.bind_identity(out/'audio-identity.json', identity,
        evidence=[path for path in out.iterdir() if path.name != '.distillation.lock'])
    if (out/'summary.json').exists():
        d.artifacts.verify_completion(out/'summary.json', identity, outputs)
        return
    if any(path.exists() for path in outputs):
        raise ValueError(f'Uncommitted audio outputs exist; retain evidence and use a new output directory: {out}')
    teacher, _ = d.YuE2Slider.load(model, args.release/spec['weights'])
    d.detach(model, teacher)
    student, _ = d.YuE2Slider.load(model, target)
    d.detach(model, student)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    if pipe._vae is None:
        pipe._vae = YuE2VAE.from_pretrained(pipe.vae_dir, decoder_only=True,
                                          device=pipe.device, local_files_only=True)
    pipe.generation_config = replace(pipe.generation_config, ode_steps=16)
    records = []
    for index, row in enumerate(d.rows(args.release, spec, 'eval')):
        seed = 1709
        plan = d.make_plan(pipe, row, seed)
        student_semantic = None
        for kind, network, acoustic in [('off', None, False), ('teacher', teacher, False),
                                         ('plain-ar', student, False), ('plain-all', student, True)]:
            start = time.monotonic()
            if kind == 'plain-all':
                semantic = student_semantic
            else:
                with d.attached(model, network, 1.):
                    semantic = pipe.generate_semantic(plan, sampling={'max_tokens': 500, 'min_tokens': 200})
                if kind == 'plain-ar':
                    student_semantic = semantic
            with d.attached(model, network if acoustic else None, 1.):
                latents = pipe.synthesize(semantic)
            z = torch.as_tensor(latents, dtype=torch.float32).T.unsqueeze(0)
            decoded = pipe._vae.decode_tiled(z, core_frames=pipe.vae_core_frames,
                                           halo_frames=16, output_device='cpu')
            if not torch.isfinite(decoded).all():
                raise RuntimeError('Non-finite audio')
            signal = decoded[0].float().T.contiguous().numpy()
            rms = float(np.sqrt(np.mean(signal.astype(np.float64)**2)))
            if rms < 1e-6:
                raise RuntimeError('Silent audio')
            filename = f'row{index}-{kind}.flac'
            sf.write(out/filename, np.clip(signal, -1, 1), 48000, subtype='PCM_16')
            np.save(out/f'row{index}-{kind}-semantic.npy', semantic.tokens)
            entry = dict(row=index, kind=kind, file=filename, seed=seed,
                seconds=len(signal)/48000, rms=rms, peak=float(np.max(np.abs(signal))),
                truncated=semantic.truncated, semantic_tokens=len(semantic.tokens),
                acoustic_prefix_adapted=acoustic, elapsed_seconds=time.monotonic()-start)
            d.write_json(out/f'row{index}-{kind}.json', dict(entry, caption=row['neutral'], lyrics=row['lyrics']))
            records.append(entry)
            d.log('audio_complete', concept=concept, **entry)
    d.artifacts.publish_completion(out/'summary.json', identity, dict(concept=concept, recordings=records,
        checkpoint_sha256=d.file_digest(target), teacher_sha256=spec['sha256'],
        protocol='Same caption, lyrics and seed; 20-second diagnostic limit, 16 ODE steps',
        plain_all_note='Native LoRA also applied to acoustic-prefix AR projections, mirroring standard ComfyUI scope; not a ComfyUI audio render'),
        outputs)
    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('--release', type=Path, default=d.ROOT.parent/'releases/yue2-concept-sliders/model')
    parser.add_argument('--reference', type=Path, default=d.ROOT/'eval/listen/yue2-particle-uni16-1200-20260917')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(9182)
    torch.cuda.set_device(args.gpu)
    catalog = json.loads((args.release/'catalog.json').read_text())
    specs = {s['id']:s for s in catalog['sliders']}
    all_rows = {}
    for spec in specs.values():
        for row in d.rows(args.release, spec, 'train'):
            all_rows[(row['neutral'],row['lyrics'])] = row
    from yue2 import YuE2Pipeline
    # Load outside inference_mode: frozen base weights participate in backward.
    with YuE2Pipeline.from_pretrained(catalog['base_model'], revision=catalog['base_revision'],
        vae=catalog['vae'], vae_revision=catalog['vae_revision'],
        cache_dir=str(d.ROOT.parent/'.cache/huggingface/hub'), local_files_only=True,
        device=f'cuda:{args.gpu}', backend='torch', quantization='none', offload_ar=False,
        memory_budget_gib=26, progress=False) as pipe:
        model = pipe._load_model().requires_grad_(False)
        for concept in args.ids.split(','):
            d.verify_fit(args, pipe, model, specs[concept], list(all_rows.values()))
            if args.audio_only:
                target = args.output/concept/f'{concept}_distilled_refined_rank8.safetensors'
            else:
                target = refine(args, pipe, model, specs[concept], list(all_rows.values()))
            audio(args, pipe, model, specs[concept], target, list(all_rows.values()))


if __name__ == '__main__':
    main()
