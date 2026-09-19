"""Resume a complete float32 fixed-history MMD state, preserving data, moments and RNG.

Run as a module from the repository root. Each invocation writes a fresh research
directory; a saved state is accepted as the source of a subsequent invocation.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from conceptmod.textsliders import train_lm_slider_music3 as legacy
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.gan_v2.data import ROOT, sha, validate_prompts, check_disjoint
from conceptmod.textsliders.gan_v2.state import cpu, code_fingerprints
from analysis.gan_bcap.objective_20260905.live import prepare, state_sequence
from analysis.gan_bcap.objective_20260905.distribution import paired_mmd
from .adaptive import adaptive_step, restore_research


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--source-weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=150)
    parser.add_argument('--save-every', type=int, default=30)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Explicitly select physical GPU 1')
    if args.out.exists() or min(args.updates, args.save_every) < 1:
        raise ValueError('Use a fresh output directory and positive budget')
    started = time.monotonic()
    args.out.mkdir(parents=True)
    source = torch.load(args.source_state, map_location='cpu', weights_only=True)
    previous = source['manifest']
    if (previous['kernel_bandwidths'] != [.03, .1, .3, 1., 3.] or
            previous['branches'] != ['conditional', 'unconditional']):
        raise ValueError('This round requires the unchanged seed objective and CFG branches')
    # Capture exact code before expensive model loading and reject changed parent dependencies.
    sources = code_fingerprints()
    computation_sources = set(sources) | {
        str((Path(__file__).parent.parent/'objective_20260905'/name).resolve())
        for name in ('live.py', 'conditional.py', 'distribution.py')} | {
        str((Path(__file__).parent/name).resolve()) for name in ('adaptive.py', 'train_adaptive.py', 'precision.py', 'coverage.py', 'train_float32.py', 'train_coverage.py', 'resume_fixed.py')}
    nontraining_changes = []
    for filename, expected in previous['sources'].items():
        actual = sha(filename)
        if actual != expected:
            if filename in computation_sources:
                raise ValueError(f'Parent computation source changed: {filename}')
            nontraining_changes.append(dict(path=filename, parent_sha256=expected, current_sha256=actual,
                reason='Not imported by the fixed-objective training computation'))
    for directory in (Path(__file__).parent, Path(__file__).parent.parent/'objective_20260905'):
        sources.update({str(path.resolve()): sha(path) for path in directory.glob('*.py')})
    for filename, expected in sources.items():
        path = Path(filename)
        destination = args.out/'provenance'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha(destination) != expected:
            raise RuntimeError('Source changed during capture')
    origin = source['step']
    configuration = previous['arguments']
    prompts, evaluation_prompts = Path(configuration['prompts']), Path(configuration['evaluation_prompts'])
    for path, key in ((prompts, 'prompts_sha256'), (evaluation_prompts, 'evaluation_prompts_sha256')):
        if sha(path) != previous[key]:
            raise ValueError('Prompt definition differs from source')
        shutil.copyfile(path, args.out/path.name)
    shutil.copyfile(args.source_state, args.out/'initial-state.pt')
    shutil.copytree(args.source_state.parent/'provenance', args.out/'parent-provenance')
    shutil.copyfile(args.source_weights, args.out/'initial.safetensors')
    sidecar = json.loads(args.source_weights.with_suffix('.json').read_text())
    write(args.out/'initial.json', sidecar)
    rows, _ = legacy._load_rows(prompts)
    heldout_rows, _ = legacy._load_rows(evaluation_prompts)
    heldout_rows = heldout_rows[:configuration['evaluation_rows']]
    validate_prompts(rows); validate_prompts(heldout_rows); check_disjoint(rows, heldout_rows)
    torch.set_num_threads(4); torch.manual_seed(7)
    device = torch.device('cuda:0')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(legacy.DEFAULT_MODEL/'tokenizer'), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(str(legacy.DEFAULT_MODEL/'language_model'),
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    if not previous.get('precision', '').startswith('float32'):
        raise ValueError('This loader requires a saved float32 objective')
    lm.lm_head.to('cpu')
    lm.model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    prepared_path = args.source_state.parent/'prepared.pt'
    if previous.get('prepared_sha256'):
        if sha(prepared_path) != previous['prepared_sha256']:
            raise ValueError('Frozen prepared histories changed')
        prepared = torch.load(prepared_path, map_location='cpu', weights_only=True)
        train, heldout = prepared['train'], prepared['heldout']
    else:
        raise ValueError('Float32 continuation requires the exact saved prepared tensors')
    scale = math.sqrt(sum(float(row['energy_target'].double().square().sum()) for row in train)
                      /sum(row['energy_target'].numel() for row in train))
    if not math.isclose(scale, previous['scale'], rel_tol=0., abs_tol=1e-12):
        raise ValueError(f'Calibration changed: {scale} versus {previous["scale"]}')
    torch.save(dict(train=train, heldout=heldout), args.out/'prepared.pt')
    history_files = {}
    for row in train+heldout:
        path = Path(row['history_cache'])
        if sha(path) != row['history_cache_sha256']:
            raise ValueError('History cache changed after preparation')
        destination = args.out/'histories'/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        history_files[str(path)] = sha(destination)
    network = LoRANetwork(lm, multiplier=1., rank=sidecar['rank'], alpha=sidecar['alpha'],
        delimiter=sidecar['delimiter'], target_replace=sidecar['target_replace'],
        prefix=sidecar['prefix'], train_method=sidecar['train_method']).to(device)
    network.requires_grad_(True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, betas=(0., .999), weight_decay=0.)
    parameters = list(network.parameters())
    original_weights = load_file(str(args.source_weights))
    if (original_weights.keys() != source['network'].keys() or
            any(not torch.equal(value, source['network'][key]) for key, value in original_weights.items())):
        raise ValueError('Scored checkpoint and resumable state have different parameters')

    def forward(row):
        legacy._set_scale(network, 1.)
        return state_sequence(lm, row, device)-row['energy_neutral'].to(device)

    def loss_for(fake, row):
        return paired_mmd(fake, row['energy_target'].to(device), scale=previous['scale'],
                          bandwidths=previous['kernel_bandwidths'])

    @torch.no_grad()
    def batch_value():
        return sum(float(loss_for(forward(row), row)) for row in train)/len(train)

    @torch.no_grad()
    def evaluate(step):
        results = dict(step=step)
        for label, group in (('train', train), ('heldout', heldout)):
            values = []
            for row in group:
                fake = forward(row); real = row['energy_target'].to(device)
                error = (fake-real).double(); span_length = row['real'].shape[1]
                values.append(dict(loss=float(loss_for(fake, row)),
                    relative_error=float(error.norm()/real.double().norm().clamp_min(1e-10)),
                    continuation_rmse=float(error[:, span_length-1:].square().mean().sqrt()),
                    magnitude=float(fake.double().norm()/real.double().norm().clamp_min(1e-10)),
                    branch_continuation_rmse=[float(e.square().mean().sqrt()) for e in error[:, span_length-1:]]))
            results[label] = values
        write(args.out/f'evaluation-{step}.json', results)
        print('EVALUATION', json.dumps(results), flush=True)
        return results

    network.load_state_dict(original_weights, strict=True)
    with torch.no_grad():
        checkpoint_outputs = [forward(row).cpu() for row in train+heldout]
    proposal = restore_research(source, network, optimizer)
    with torch.no_grad():
        for row, expected in zip(train+heldout, checkpoint_outputs):
            if not torch.equal(forward(row).cpu(), expected):
                raise RuntimeError('Scored-checkpoint versus restored-state output parity failed')
        legacy._set_scale(network, 0.)
        for row in train+heldout:
            if not torch.equal(state_sequence(lm, row, device).cpu(), row['energy_neutral']):
                raise RuntimeError('Zero-scale output identity failed')
    initial = evaluate(origin)
    historical = json.loads((args.source_state.parent/f'evaluation-{origin}.json').read_text())
    for label in ('train', 'heldout'):
        if any(abs(a['loss']-b['loss']) > 1e-7 for a, b in zip(initial[label], historical[label])):
            raise RuntimeError('Starting loss differs from the saved evaluation')
    torch.save(checkpoint_outputs, args.out/'starting-outputs.pt')
    write(args.out/'resume-parity.json', dict(status='passed', step=origin,
        checkpoint_state_tensors='bitwise identical', restored_outputs=f'bitwise identical, all {len(train)+len(heldout)} histories and both branches',
        zero_scale='bitwise identical', starting_loss=sum(x['loss'] for x in initial['train'])/len(train),
        historical_loss_tolerance=1e-7, calibration=scale,
        optimizer='all source AdamW moments, step counts and parameter groups restored', rng='torch CPU and visible CUDA restored'))
    manifest = copy.deepcopy(previous)
    manifest.update(arguments=dict(configuration, source_state=str(args.source_state.resolve()),
        source_weights=str(args.source_weights.resolve()), out=str(args.out.resolve()),
        updates=args.updates, save_every=args.save_every, fresh_optimizer=False),
        source_sha256=sha(args.source_state), source_weights_sha256=sha(args.source_weights), source_step=origin,
        sources=sources, prepared_sha256=sha(args.out/'prepared.pt'), history_files=history_files,
        gpu=1, gpu_name=torch.cuda.get_device_name(),
        packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'diffusers', 'safetensors')},
        generator_optimizer='restored AdamW moments and groups; persistent adaptive fraction; normalized negative-gradient fallback restores Adam moments',
        proposal_initial=copy.deepcopy(proposal), search_trials_per_direction=8,
        stopping=f'{args.updates} attempted updates at most; stop after three unsuccessful attempts with changed scale and gradient fallback',
        parent_manifest=previous, nontraining_source_changes=nontraining_changes)
    write(args.out/'manifest.json', manifest)
    history = copy.deepcopy(source['history'])
    del source, original_weights, checkpoint_outputs

    def save(step):
        path = args.out/f'{args.out.name}_step{step}.safetensors'
        save_file({key: value.detach().cpu().contiguous() for key, value in network.state_dict().items()}, str(path))
        write(path.with_suffix('.json'), dict(sidecar, steps=step, research_objective=manifest['objective'],
              quality_status='unvalidated_research_candidate'))
        blob = dict(schema='mmd-adaptive-1', manifest=manifest, step=step, network=cpu(network.state_dict()),
                    optimizer=cpu(optimizer.state_dict()), metric=None, d_optimizer=None,
                    history=history, proposal=copy.deepcopy(proposal),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        temporary = args.out/'state.pt.tmp'
        torch.save(blob, temporary); temporary.replace(args.out/'state.pt')
        os.link(args.out/'state.pt', args.out/f'state-step{step}.pt')

    save(origin)
    train_started = time.monotonic()
    reason = 'budget_complete'
    with (args.out/'train.jsonl').open('w') as log:
        for update in range(1, args.updates+1):
            optimizer.zero_grad(set_to_none=True)
            initial_loss = 0.
            for row in train:
                loss = loss_for(forward(row), row)/len(train)
                initial_loss += float(loss.detach()); loss.backward()
            gradient_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in parameters if p.grad is not None))
            result = adaptive_step(parameters, optimizer, batch_value, initial_loss, proposal)
            record = dict(step=origin+update, loss_before=initial_loss, gradient_norm=gradient_norm,
                          parameter_step=result['parameter_step'], line_search=result,
                          seconds=time.monotonic()-train_started)
            history.append(record); log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
            write(args.out/'progress.json', record)
            if update % 10 == 0 or not result['accepted']:
                print(json.dumps(record), flush=True)
            stalled = proposal['consecutive_rejections'] >= 3
            if update % args.save_every == 0 or update == args.updates or stalled:
                save(origin+update); evaluate(origin+update)
            if stalled:
                reason = 'stalled_after_changed_scales_and_gradient_fallback'; break
    current = history[-update:]
    write(args.out/'completion.json', dict(status=reason, step=origin+update, attempted_updates=update,
        accepted=sum(row['line_search']['accepted'] for row in current),
        gradient_fallback_acceptances=sum(row['line_search']['method']=='gradient' for row in current),
        objective_evaluations=sum(row['line_search']['trials'] for row in current),
        training_seconds=time.monotonic()-train_started, total_seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), final_weights=str(
            (args.out/f'{args.out.name}_step{origin+update}.safetensors').resolve())))


if __name__ == '__main__':
    main()
