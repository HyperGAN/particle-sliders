"""Float32 paired MMD on a frozen equal mixture of neutral and positive-scale parent histories.

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
from .precision import float32_teachers
from .precision_memory import directional_probe
from .coverage import collect_parent_histories


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--source-weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--reference-float32-prepared', type=Path, required=True)
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
        str((Path(__file__).parent/name).resolve()) for name in ('adaptive.py', 'train_adaptive.py', 'precision.py', 'train_float32.py', 'coverage.py', 'train_coverage.py', 'precision_memory.py')}
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
    lm.lm_head.to('cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared_path = args.prepared
    prepared = torch.load(prepared_path, map_location='cpu', weights_only=True)
    train, heldout = prepared['train'], prepared['heldout']
    for group, definitions in ((train, rows), (heldout, heldout_rows)):
        if len(group) != len(definitions):
            raise ValueError('Prepared prompt count changed')
        from conceptmod.textsliders.gan_v2.data import digest
        for row, definition in zip(group, definitions):
            if row['prompt_hash'] != digest(definition):
                raise ValueError('Prepared input and prompt definition differ')
    shutil.copyfile(prepared_path, args.out/'prepared-bf16.pt')
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
        return paired_mmd(fake, row['energy_target'].to(device), scale=scale,
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
                values.append(dict(history_origin=row.get('history_origin', 'original frozen neutral history'),
                    loss=float(loss_for(fake, row)),
                    relative_error=float(error.norm()/real.double().norm().clamp_min(1e-10)),
                    continuation_rmse=float(error[:, span_length-1:].square().mean().sqrt()),
                    magnitude=float(fake.double().norm()/real.double().norm().clamp_min(1e-10)),
                    branch_continuation_rmse=[float(e.square().mean().sqrt()) for e in error[:, span_length-1:]]))
            results[label] = values
            if label == 'train':
                results['train_groups'] = {kind: sum(v['loss'] for v in values if v['history_origin'] == kind)
                    /sum(v['history_origin'] == kind for v in values)
                    for kind in sorted({v['history_origin'] for v in values})}
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
        checkpoint_state_tensors='bitwise identical', restored_outputs='bitwise identical, all six rows and both branches',
        zero_scale='bitwise identical', starting_loss=sum(x['loss'] for x in initial['train'])/len(train),
        historical_loss_tolerance=1e-7, calibration=scale,
        optimizer='all source AdamW moments, step counts and parameter groups restored', rng='torch CPU and visible CUDA restored'))
    # Probe the exact source arithmetic before changing anything. Then widen the
    # base's existing bf16 values; do not silently load different base weights.
    def losses():
        for row in train:
            yield loss_for(forward(row), row)/len(train)
    probe_started = time.monotonic()
    bf16_probe = directional_probe(parameters, losses)
    write(args.out/'precision-bf16.json', bf16_probe)
    print('BF16_PROBE', json.dumps(bf16_probe), flush=True)
    added = collect_parent_histories(lm, tokenizer, network, train, rows, device,
        args.out/'parent-histories', parent_weights=args.source_weights, frames_cap=configuration['frames'], seed=17)
    if any(not torch.equal(value.detach().cpu(), original_weights[key])
           for key, value in network.state_dict().items()):
        raise RuntimeError('Parent parameters changed during frozen history collection')
    for row in added:
        history_files[row['history_cache']] = row['history_cache_sha256']
    original_count = len(train)
    train = train+added
    for row in train[:original_count]:
        row['history_origin'] = 'original frozen neutral history'
    lm.model.float()
    torch.cuda.empty_cache()
    train = float32_teachers(lm, tokenizer, network, train, rows+rows, device)
    heldout = float32_teachers(lm, tokenizer, network, heldout, heldout_rows, device)
    reference = torch.load(args.reference_float32_prepared, map_location='cpu', weights_only=True)
    for group, expected_group in ((train[:original_count], reference['train']), (heldout, reference['heldout'])):
        if len(group) != len(expected_group):
            raise ValueError('Prior float32 diagnostic history count differs')
        for row, expected in zip(group, expected_group):
            for key in ('prompt_embeds', 'span_mask', 'frame_embeds', 'energy_neutral', 'energy_target'):
                a, b = row[key], expected[key]
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    raise RuntimeError(f'Original float32 data changed after collection: {key}')
    write(args.out/'coverage-parity.json', dict(status='passed',
        parent_parameters_after_collection='bitwise identical to scored parent',
        original_and_heldout_data='all inputs and float32 teachers bitwise identical to round two',
        reference_prepared_sha256=sha(args.reference_float32_prepared),
        added_histories=len(added), original_histories=original_count))
    del reference
    scale = math.sqrt(sum(float(row['energy_target'].double().square().sum()) for row in train)
                      /sum(row['energy_target'].numel() for row in train))
    torch.save(dict(train=train, heldout=heldout), args.out/'prepared.pt')
    with torch.no_grad():
        legacy._set_scale(network, 0.)
        for row in train+heldout:
            if not torch.equal(state_sequence(lm, row, device).cpu(), row['energy_neutral']):
                raise RuntimeError('Float32 zero-scale identity failed')
    float32_probe = directional_probe(parameters, losses)
    write(args.out/'precision-float32.json', float32_probe)
    print('FLOAT32_PROBE', json.dumps(float32_probe), flush=True)
    if max(float32_probe['no_grad_repeats'])-min(float32_probe['no_grad_repeats']) > 1e-7:
        raise RuntimeError('Float32 objective is not repeatable')
    if abs(float32_probe['backward_loss']-float32_probe['no_grad_repeats'][0]) > 1e-6:
        raise RuntimeError('Float32 backward and line-search losses differ')
    if not any(t['sides']['1']['change'] < 0 for t in float32_probe['trials']):
        raise RuntimeError('Float32 negative-gradient probe found no descent')
    # Targets and calibration changed, so initialize the optimizer explicitly.
    optimizer = torch.optim.AdamW(parameters, lr=.0005, betas=(0., .999), weight_decay=0.)
    proposal = dict(scale=1., consecutive_rejections=0)
    initial_float32 = evaluate(origin)
    write(args.out/'precision-transition.json', dict(
        bf16_scale=previous['scale'], float32_scale=scale,
        bf16_loss=bf16_probe['no_grad_repeats'][0],
        float32_loss=float32_probe['no_grad_repeats'][0],
        probe_seconds=time.monotonic()-probe_started,
        input_values='Original neutral histories plus frozen parent +1 histories, equal weight; cached bf16 values widened exactly',
        base_weights='existing bf16 values widened to float32; LM head held on CPU and unused',
        optimizer='fresh AdamW; discarded parent moments archived in initial-state.pt',
        zero_scale='bitwise identity against recomputed float32 neutral teachers'))
    manifest = copy.deepcopy(previous)
    manifest.update(arguments=dict(configuration, source_state=str(args.source_state.resolve()),
        source_weights=str(args.source_weights.resolve()), out=str(args.out.resolve()),
        updates=args.updates, save_every=args.save_every, fresh_optimizer=True, prepared=str(args.prepared.resolve()),
        reference_float32_prepared=str(args.reference_float32_prepared.resolve())),
        source_sha256=sha(args.source_state), source_weights_sha256=sha(args.source_weights), source_step=origin,
        sources=sources, prepared_sha256=sha(args.out/'prepared.pt'), history_files=history_files,
        gpu=1, gpu_name=torch.cuda.get_device_name(),
        packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'diffusers', 'safetensors')},
        generator_optimizer='fresh AdamW, lr 0.0005, betas (0, 0.999), no weight decay; adaptive fraction and transactional gradient fallback',
        precision='float32 base forward and LoRA; TF32 disabled',
        source_prepared_sha256=sha(args.prepared),
        reference_float32_prepared_sha256=sha(args.reference_float32_prepared),
        scale=scale, objective='paired multiscale RBF MMD with float32 teachers on an equal mixture of frozen neutral and parent +1 histories',
        training_history_count=len(train), history_mix='One original neutral and one new frozen parent +1 sample per condition, equal weighting',
        parent_history_manifest_sha256=sha(args.out/'parent-histories/manifest.json'),
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
    del prepared, initial_float32
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
            if update % 5 == 0 or not result['accepted']:
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
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), initial_float32_loss=float32_probe['no_grad_repeats'][0],
        final_loss=current[-1]['line_search']['loss'], final_weights=str(
            (args.out/f'{args.out.name}_step{origin+update}.safetensors').resolve())))


if __name__ == '__main__':
    main()
