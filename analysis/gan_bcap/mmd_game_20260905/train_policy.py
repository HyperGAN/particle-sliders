"""Train or exactly resume readout-aware distillation on frozen coverage data.

This is a separately labeled distillation candidate, not a pure MMD continuation.
The parent parameters are retained, with fresh moments for a changed objective.
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

import torch
from safetensors.torch import load_file, save_file

from conceptmod.textsliders import train_lm_slider_music3 as legacy
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.gan_v2.data import ROOT, sha, validate_prompts, check_disjoint, digest
from conceptmod.textsliders.gan_v2.state import cpu, code_fingerprints
from analysis.gan_bcap.objective_20260905.live import state_sequence
from .adaptive import adaptive_step, restore_research
from .precision_memory import directional_probe
from .policy_geometry import PolicyObjective, policy_diagnostics


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--source-weights', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=150)
    parser.add_argument('--save-every', type=int, default=30)
    parser.add_argument('--resume-objective', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Select physical GPU 1 explicitly')
    if args.out.exists() or min(args.updates, args.save_every) < 1:
        raise ValueError('Use a fresh output directory and positive budget')
    started = time.monotonic()
    args.out.mkdir(parents=True)
    source = torch.load(args.source_state, map_location='cpu', weights_only=True)
    previous = source['manifest']
    if args.resume_objective != (source.get('schema') == 'policy-distillation-1'):
        raise ValueError('Declare resume-objective only for a saved policy objective')
    origin = source['step']
    data_manifest = json.loads((args.prepared.parent/'manifest.json').read_text())
    if sha(args.prepared) != data_manifest['prepared_sha256']:
        raise ValueError('Prepared data do not match their archived manifest')
    if not data_manifest.get('precision', '').startswith('float32'):
        raise ValueError('Prepared teachers must use float32 arithmetic')
    if data_manifest['branches'] != ['conditional', 'unconditional']:
        raise ValueError('Expected the actual CFG pair')
    sources = code_fingerprints()
    changes = []
    for filename, expected in previous['sources'].items():
        actual = sha(filename)
        if actual != expected:
            if filename in sources:
                raise ValueError(f'Parent base computation changed: {filename}')
            changes.append(dict(path=filename, parent_sha256=expected, current_sha256=actual))
    for directory in (Path(__file__).parent, Path(__file__).parent.parent/'objective_20260905'):
        sources.update({str(path.resolve()): sha(path) for path in directory.glob('*.py')})
    for filename, expected in sources.items():
        destination = args.out/'provenance'/Path(filename).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, destination)
        if sha(destination) != expected:
            raise RuntimeError('Source changed during capture')
    shutil.copyfile(args.source_state, args.out/'initial-state.pt')
    shutil.copyfile(args.source_weights, args.out/'initial.safetensors')
    shutil.copytree(args.source_state.parent/'provenance', args.out/'parent-provenance')
    shutil.copyfile(args.prepared, args.out/'source-prepared.pt')
    write(args.out/'data-source-manifest.json', data_manifest)
    sidecar = json.loads(args.source_weights.with_suffix('.json').read_text())
    write(args.out/'initial.json', sidecar)
    configuration = data_manifest['arguments']
    prompts, evaluation_prompts = Path(configuration['prompts']), Path(configuration['evaluation_prompts'])
    for path, key in ((prompts, 'prompts_sha256'), (evaluation_prompts, 'evaluation_prompts_sha256')):
        if sha(path) != data_manifest[key]:
            raise ValueError('Frozen prompt definition changed')
        shutil.copyfile(path, args.out/path.name)
    rows, _ = legacy._load_rows(prompts)
    heldout_rows, _ = legacy._load_rows(evaluation_prompts)
    heldout_rows = heldout_rows[:configuration['evaluation_rows']]
    validate_prompts(rows); validate_prompts(heldout_rows); check_disjoint(rows, heldout_rows)
    prepared = torch.load(args.prepared, map_location='cpu', weights_only=True)
    train, heldout = prepared['train'], prepared['heldout']
    if any(row['prompt_hash'] not in {digest(r) for r in rows} for row in train):
        raise ValueError('A training history has an undeclared condition')
    if len(heldout) != len(heldout_rows) or any(row['prompt_hash'] != digest(r) for row, r in zip(heldout, heldout_rows)):
        raise ValueError('Diagnostic conditions differ')
    history_files = {}
    for row in train+heldout:
        path = Path(row['history_cache'])
        if sha(path) != row['history_cache_sha256']:
            raise ValueError('Frozen history changed')
        destination = args.out/'histories'/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        history_files[str(path)] = sha(destination)
    torch.set_num_threads(4); torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    from transformers import AutoModelForCausalLM
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET, _SEMANTIC_VOCAB_SIZE, _AUDIO_END_TOKEN_ID, _AR_CFG_SCALE)
    lm = AutoModelForCausalLM.from_pretrained(str(legacy.DEFAULT_MODEL/'language_model'),
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    lm.lm_head.to('cpu')
    if lm.lm_head.bias is not None:
        raise ValueError('Readout extraction requires the native bias-free head')
    ids = list(range(_AUDIO_CODE_OFFSET, _AUDIO_CODE_OFFSET+_SEMANTIC_VOCAB_SIZE))+[_AUDIO_END_TOKEN_ID]
    readout_cpu = lm.lm_head.weight.detach()[ids].float().contiguous()
    readout_path = args.out/'semantic-readout.pt'
    torch.save(dict(weight=readout_cpu, token_ids=ids, cfg_scale=_AR_CFG_SCALE), readout_path)
    readout = readout_cpu.to(device)
    del readout_cpu
    lm.model.float()
    torch.cuda.empty_cache()
    network = LoRANetwork(lm, multiplier=1., rank=sidecar['rank'], alpha=sidecar['alpha'],
        delimiter=sidecar['delimiter'], target_replace=sidecar['target_replace'],
        prefix=sidecar['prefix'], train_method=sidecar['train_method']).to(device)
    network.requires_grad_(True)
    parameters = list(network.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.0005, betas=(0., .999), weight_decay=0.)
    initial_weights = load_file(str(args.source_weights))
    if initial_weights.keys() != source['network'].keys() or any(
            not torch.equal(value, source['network'][key]) for key, value in initial_weights.items()):
        raise ValueError('Scored parent and saved state differ')

    def forward(row):
        legacy._set_scale(network, 1.)
        return state_sequence(lm, row, device)-row['energy_neutral'].to(device)

    network.load_state_dict(initial_weights, strict=True)
    with torch.no_grad():
        checkpoint_outputs = [forward(row).cpu() for row in train+heldout]
    if args.resume_objective:
        network.load_state_dict(source['network'], strict=True)
        optimizer.load_state_dict(source['optimizer'])
        if source['history'] and source['history'][-1]['step'] != origin:
            raise ValueError('History does not reach saved step')
        torch.set_rng_state(source['torch_rng'])
        if len(source['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Visible GPU topology changed')
        torch.cuda.set_rng_state_all(source['cuda_rng'])
        proposal = copy.deepcopy(source['proposal'])
        if sha(args.prepared) != previous['prepared_sha256']:
            raise ValueError('Resume requires exact policy teachers and histories')
        old_readout = torch.load(args.source_state.parent/'semantic-readout.pt', map_location='cpu', weights_only=True)
        if old_readout['token_ids'] != ids or old_readout['cfg_scale'] != _AR_CFG_SCALE or not torch.equal(old_readout['weight'], readout.cpu()):
            raise ValueError('Native readout changed on resume')
        del old_readout
    else:
        restore_research(source, network, optimizer)
        optimizer = torch.optim.AdamW(parameters, lr=.0005, betas=(0., .999), weight_decay=0.)
        proposal = dict(scale=1., consecutive_rejections=0)
    with torch.no_grad():
        for row, expected in zip(train+heldout, checkpoint_outputs):
            if not torch.equal(forward(row).cpu(), expected):
                raise RuntimeError('Checkpoint versus restored-state output parity failed')
        legacy._set_scale(network, 0.)
        for row in train+heldout:
            if not torch.equal(state_sequence(lm, row, device).cpu(), row['energy_neutral']):
                raise RuntimeError('Base arithmetic differs from frozen teachers')
    torch.save(checkpoint_outputs, args.out/'starting-outputs.pt')
    objective = PolicyObjective(readout, hidden_scale=data_manifest['scale'],
        bandwidths=data_manifest['kernel_bandwidths'], cfg_scale=_AR_CFG_SCALE)
    if args.resume_objective and any(previous[key] != value for key, value in (
            ('policy_stride', objective.stride), ('cfg_scale', objective.cfg_scale),
            ('hidden_weight', objective.hidden_weight))):
        raise ValueError('Policy objective configuration changed on resume')
    for row in train+heldout:
        teacher = objective.teacher(row)
        if args.resume_objective and not torch.equal(teacher, row['policy_teacher_log_probs']):
            raise ValueError('Policy teacher changed on resume')
        row['policy_teacher_log_probs'] = teacher
    if args.resume_objective:
        objective.policy_calibration = previous['policy_calibration']
        objective.hidden_calibration = previous['hidden_calibration']
    else:
        with torch.no_grad():
            terms = [tuple(float(v) for v in objective.terms(forward(row), row)) for row in train]
        objective.policy_calibration = sum(v[0] for v in terms)/len(terms)
        objective.hidden_calibration = sum(v[1] for v in terms)/len(terms)
        if min(objective.policy_calibration, objective.hidden_calibration) <= 1e-10:
            raise ValueError('Degenerate objective calibration')
    torch.save(dict(train=train, heldout=heldout), args.out/'prepared.pt')

    @torch.no_grad()
    def batch_value():
        return sum(float(objective(forward(row), row)) for row in train)/len(train)

    @torch.no_grad()
    def evaluate(step):
        result = dict(step=step)
        for label, group in (('train', train), ('heldout', heldout)):
            values = []
            for row in group:
                fake = forward(row)
                policy, hidden_loss = objective.terms(fake, row)
                student = objective.log_probs(fake+row['energy_neutral'].to(fake), row)
                diagnostics = policy_diagnostics(student, row['policy_teacher_log_probs'].to(fake))
                values.append(dict(history_origin=row.get('history_origin', 'neutral'),
                    loss=float(objective(fake, row)), policy_kl=float(policy), hidden_mmd=float(hidden_loss),
                    **diagnostics))
            result[label] = values
        write(args.out/f'evaluation-{step}.json', result)
        print('EVALUATION', json.dumps(result), flush=True)
        return result

    initial = evaluate(origin)
    if args.resume_objective:
        historical = json.loads((args.source_state.parent/f'evaluation-{origin}.json').read_text())
        for group in ('train', 'heldout'):
            if len(initial[group]) != len(historical[group]) or any(abs(a['loss']-b['loss']) > 1e-7 for a, b in zip(initial[group], historical[group])):
                raise ValueError('Starting objective differs from saved evaluation')

    def losses():
        for row in train:
            yield objective(forward(row), row)/len(train)
    probe = directional_probe(parameters, losses)
    write(args.out/'precision-float32.json', probe)
    if max(probe['no_grad_repeats'])-min(probe['no_grad_repeats']) > 1e-7 or abs(probe['backward_loss']-probe['no_grad_repeats'][0]) > 1e-6:
        raise RuntimeError('Objective forward and backward are inconsistent')
    if not any(trial['sides']['1']['change'] < 0 for trial in probe['trials']):
        raise RuntimeError('Negative-gradient probe found no descent')
    write(args.out/'resume-parity.json', dict(status='passed', step=origin,
        weights='bitwise identical to scored parent', starting_outputs='checkpoint and restored state bitwise identical',
        zero_scale='bitwise identical to supplied frozen float32 teachers',
        optimizer='restored exactly' if args.resume_objective else 'fresh AdamW; source moments archived',
        calibration_scope='initial training histories only; fixed throughout updates'))
    manifest = copy.deepcopy(data_manifest)
    manifest.update(arguments=dict(configuration, source_state=str(args.source_state.resolve()),
        source_weights=str(args.source_weights.resolve()), out=str(args.out.resolve()),
        prepared=str(args.prepared.resolve()), updates=args.updates, save_every=args.save_every,
        fresh_optimizer=not args.resume_objective),
        family='distillation', objective_kind='policy-distillation-1',
        objective='Forward KL to positive-caption conditional and CFG-guided semantic policies, plus 0.1 normalized paired hidden MMD; fixed initial training calibrations',
        policy_calibration=objective.policy_calibration, hidden_calibration=objective.hidden_calibration,
        policy_stride=objective.stride, cfg_scale=objective.cfg_scale, hidden_weight=objective.hidden_weight,
        semantic_readout_sha256=sha(readout_path), source_step=origin,
        source_sha256=sha(args.source_state), source_weights_sha256=sha(args.source_weights),
        source_prepared_sha256=sha(args.prepared), prepared_sha256=sha(args.out/'prepared.pt'),
        sources=sources, history_files=history_files, gpu=1, gpu_name=torch.cuda.get_device_name(),
        precision='float32 widened from existing bf16 base values; TF32 disabled',
        generator_optimizer='AdamW lr0.0005, betas(0,0.999), no weight decay; adaptive transactional line search',
        proposal_initial=copy.deepcopy(proposal), parent_manifest=previous,
        source_changes=changes, packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'diffusers', 'safetensors')},
        limitations=['Frozen-history smooth policy surrogate, not full trajectory matching',
            'Native conditional top-50 masking and depth-code policy are not explicitly optimized',
            'Soft semantic targets do not establish audio mode coverage', 'Audio judge and render arithmetic stay unchanged'])
    write(args.out/'manifest.json', manifest)
    history = copy.deepcopy(source['history'])
    del source, checkpoint_outputs, initial_weights

    def save(step):
        path = args.out/f'{args.out.name}_step{step}.safetensors'
        save_file({key: value.detach().cpu().contiguous() for key, value in network.state_dict().items()}, str(path))
        write(path.with_suffix('.json'), dict(sidecar, steps=step, research_objective=manifest['objective'],
            quality_status='unvalidated_research_candidate'))
        blob = dict(schema='policy-distillation-1', manifest=manifest, step=step, network=cpu(network.state_dict()),
            optimizer=cpu(optimizer.state_dict()), metric=None, d_optimizer=None, history=history,
            proposal=copy.deepcopy(proposal), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
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
                loss = objective(forward(row), row)/len(train)
                initial_loss += float(loss.detach()); loss.backward()
            gradient_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in parameters if p.grad is not None))
            result = adaptive_step(parameters, optimizer, batch_value, initial_loss, proposal)
            record = dict(step=origin+update, loss_before=initial_loss, gradient_norm=gradient_norm,
                parameter_step=result['parameter_step'], line_search=result, seconds=time.monotonic()-train_started)
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
