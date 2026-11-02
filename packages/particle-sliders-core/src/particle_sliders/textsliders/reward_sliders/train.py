"""Train/export the normal rank-8 LoRA only after the frozen causal gate passes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import torch

from .specs import DEFAULT_RUN, MODEL, WORKSPACE, digest, read_json, write_json, sha, save_tensor
from .directions import RewardTeacher
from .data import prepare_reward_rows, RewardStudentForward, FamilySampler, set_scale, generation_hidden
from .experiment import verify, status


def require_causal_evidence(run, manifest):
    gate = read_json(run/'causal-result.json')
    if not gate['passed'] or gate['manifest_sha256'] != digest(manifest) or gate['teacher_sha256'] != sha(run/'teacher.pt'):
        raise ValueError('LoRA training requires a passing causal result for this exact manifest and teacher')
    declared = read_json(run/'causal-arms.json')
    if len(declared)!=64:
        raise ValueError('The causal gate requires all 64 predeclared observations, including reversed controls')
    observations, fingerprints = [], {}
    for job in declared:
        ident=f"{job['family']}-s{job['seed']}-{job['arm']['name']}"
        path=run/'observations'/f'{ident}.json'
        observation=read_json(path)
        if (observation['status']!='complete' or not observation.get('reward',{}).get('valid')
            or observation['arm']!=job['arm'] or observation['seed']!=job['seed']
            or observation['family']!=job['family'] or observation['split']!='test'
            or observation['provenance']['manifest_sha256']!=digest(manifest)
            or sha(observation['audio'])!=observation['audio_sha256']):
            raise ValueError('Missing, failed, or changed causal evidence; training remains gated')
        observations.append(observation); fingerprints[ident]=sha(path)
    from .directions import causal_gate
    verified=causal_gate(observations,min_gain=manifest['pilot']['minimum_causal_ce_gain'])
    if not verified['passed'] or any(gate[k]!=verified[k] for k in ('positive_vs_off','positive_vs_random','reversed_vs_off')):
        raise ValueError('Causal summary does not match its observations')
    gate=dict(gate,observation_sha256=fingerprints)
    return gate


def audit_adapter(network, lm):
    names = [m.lora_name for m in network.unet_loras]
    if len(names) != 144 or len(set(names)) != 144:
        raise ValueError('Expected all 144 q/k/v/o attention projections')
    for module in network.unet_loras:
        if module.lora_dim != 8 or float(module.alpha) != 8 or module.scale != 1.:
            raise ValueError('Rank/alpha mismatch')
        if not any(module.lora_name.endswith('-'+p+'_proj') for p in ('q','k','v','o')):
            raise ValueError('Unexpected adapter projection')
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise FloatingPointError('Nonfinite adapter')
    return dict(modules=144, tensors=len(network.state_dict()), rank=8, alpha=8,
                keys=names, layers=len(lm.model.layers))


@torch.no_grad()
def prompt_diagnostics(lm, network, rows, apply_styles, device):
    result = []
    for family in sorted({r['family'] for r in rows}):
        row = next(r for r in rows if r['family']==family and r['branch']==0)
        apply_styles(family)
        set_scale(network, 1.)
        hidden = generation_hidden(lm, row['embeds'].to(device))
        prompt = hidden[:, :row['boundary']].float()
        baseline = row['prompt_teacher'].to(device).float()
        delta = prompt-baseline
        fake = hidden[:,row['boundary']::row['stride']].float()-row['neutral_span'].to(device)
        real = row['real'].to(device)
        result.append(dict(family=family, prompt_relative_rms=float(delta.norm()/baseline.norm().clamp_min(1e-8)),
                           generation_relative_error=float((fake-real).norm()/real.norm()),
                           residual_cosine=float(torch.nn.functional.cosine_similarity(fake.flatten(),real.flatten(),dim=0))))
    return result


def train(run):
    from .lifecycle import exclusive_stage
    with exclusive_stage(run):
        return _train(run)


def _train(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Train on reserved physical GPU 1')
    run = Path(run)
    manifest = read_json(run/'manifest.json'); verify(manifest)
    gate = require_causal_evidence(run, manifest)
    sys.path.insert(0, str(WORKSPACE))
    from app import generator
    from app.lora_runtime import LoRANetwork
    from transformers import AutoModelForCausalLM
    from ..gan_v2.critic import SpanCritic, pad_sequences
    from ..gan_v2.engine import GANEngine
    from ..gan_v2.train import arm_settings
    from ..gan_v2 import state
    torch.set_num_threads(4); torch.cuda.set_device(0)
    torch.manual_seed(7); random.seed(7)
    device = torch.device('cuda:0')
    folder = run/'student'; folder.mkdir(exist_ok=True)
    teacher = RewardTeacher(**torch.load(run/'teacher.pt', map_location='cpu', weights_only=True))
    recipe, critic_config = arm_settings('baseline', origin=0, horizon=660, diagnostics_every=25)
    # Reporting/orchestration edits must not invalidate an unchanged optimizer
    # or teacher state. Collection dependencies are bound by manifest_sha256.
    source_paths = [Path(__file__).parent/name for name in ('train.py','data.py','capture.py','specs.py','directions.py','lifecycle.py')]
    sources = {str(p): sha(p) for p in source_paths}
    sources.update(state.code_fingerprints())
    signature = dict(schema='reward-generation-student-v1', manifest_sha256=digest(manifest),
                     teacher_sha256=sha(run/'teacher.pt'), gate_sha256=sha(run/'causal-result.json'),
                     causal_observation_sha256=gate['observation_sha256'],
                     sources=sources, recipe=asdict(recipe), critic=critic_config, rank=8, alpha=8,
                     frames=128, stride=4, sampler='equal-family homogeneous four-row batches',
                     style_apply='frozen ordinary merged style weights', cfg='independent conditional/unconditional rows',
                     checkpoints=manifest['student']['steps'],
                     gross_divergence_stop=dict(prompt_relative_rms=2.,generation_relative_error=20.),
                     rng_policy='reseed after row preparation; preserve CPU/CUDA RNG around frozen style loading')
    frozen = folder/'manifest.json'
    if frozen.exists() and read_json(frozen) != signature:
        raise ValueError('Student recipe/source changed; cannot silently resume')
    write_json(frozen, signature)
    for source, expected in sources.items():
        target = folder/'provenance'/Path(source).relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(source).read_bytes())
        assert sha(target) == expected
    status(run, 'preparing_generation_student')
    lm = AutoModelForCausalLM.from_pretrained(str(MODEL/'language_model'), torch_dtype=torch.bfloat16,
                                             local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    pipe = SimpleNamespace(language_model=lm)
    def apply_styles(family):
        # Constructing frozen weight holders initializes throwaway tensors.
        # Cache hits/misses must not change the student's stochastic sequence.
        with torch.random.fork_rng(devices=[0]):
            generator._merge_sliders(pipe, 'cuda:0', manifest['style_components'][family])
    observations = read_json(run/'baseline-observations.json')
    cache = folder/'rows.pt'
    if cache.exists():
        cached = torch.load(cache, map_location='cpu', weights_only=True)
        if cached['signature'] != digest(signature):
            raise ValueError('Student row cache identity changed')
        rows = cached['rows']
    else:
        rows = prepare_reward_rows(lm, observations, teacher, apply_styles, device=device)
        save_tensor(cache, dict(signature=digest(signature), rows=rows))
    torch.manual_seed(7); random.seed(7)
    network = LoRANetwork(lm, rank=8, alpha=8., multiplier=0., target_replace=['Qwen3Attention'],
                          prefix='lora_te', delimiter='-', train_method='full').to(device)
    network.requires_grad_(True)
    topology = audit_adapter(network, lm)
    # Exact zero-output initialization and same-shape baseline restoration.
    if any(torch.count_nonzero(m.lora_up.weight) for m in network.unet_loras):
        raise ValueError('Fresh student is not zero-output')
    row = rows[0]; apply_styles(row['family']); set_scale(network,0.)
    with torch.no_grad():
        zero = generation_hidden(lm, row['embeds'].to(device))[:,row['boundary']::row['stride']].float().cpu()
    if not torch.equal(zero,row['neutral_span']):
        raise ValueError('Student zero differs from baseline teacher on the same generation history')
    critic = SpanCritic(4096, **critic_config).to(device)
    real, mask = pad_sequences([r['real'].to(device) for r in rows])
    scale = critic.calibrate_input_scale(real, mask)
    del real, mask
    engine = GANEngine(network, critic, RewardStudentForward(lm,network,apply_styles,device), rows, recipe)
    sampler = FamilySampler(rows)
    history = state.restore(folder/'state.pt',engine,sampler,None,signature) if (folder/'state.pt').exists() else []
    write_json(folder/'geometry-audit.json', dict(passed=True, topology=topology, zero_parity=True,
               row_count=len(rows), critic_input_scale=scale, frames=128, stride=4,
               prompt_and_lyric_positions_excluded=True, branches=['conditional','unconditional']))
    stop = False
    def cancel(signum, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,cancel)

    def export():
        from safetensors.torch import save_file, load_file
        checkpoint = folder/f'reward-ce-v1_step{engine.completed}.safetensors'
        tensors = {k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()}
        audit_adapter(network,lm)
        save_file(tensors,str(checkpoint))
        saved = load_file(str(checkpoint))
        if set(saved)!=set(tensors) or any(not torch.equal(saved[k],tensors[k]) for k in saved):
            raise ValueError('Export differs from live rank-8 adapter')
        metadata = dict(kind='language_model',rank=8,alpha=8,target_replace=['Qwen3Attention'],
                        prefix='lora_te',delimiter='-',train_method='full',unit_scale=1.,
                        plus_label='CE',minus_label='Off',recommended_range=[0.,1.],unipolar=True,
                        steps=engine.completed, prompts_file=str(run/'manifest.json'),
                        weights_sha256=sha(checkpoint), reward=dict(name='reward-ce-v1',
                            spec=manifest['reward_spec'], teacher_sha256=sha(run/'teacher.pt'),
                            manifest_sha256=digest(manifest), student_signature_sha256=digest(signature),
                            interpretation='research CE student; free-running selection pending'))
        write_json(checkpoint.with_suffix('.json'),metadata)
        diagnostics = prompt_diagnostics(lm,network,rows,apply_styles,device)
        write_json(folder/f'diagnostics-step{engine.completed}.json',diagnostics)
        state.save(folder/'state.pt',engine,sampler,None,signature,history)
        state.save(folder/f'step{engine.completed}_state.pt',engine,sampler,None,signature,history)
        write_json(folder/'latest.json',dict(step=engine.completed,weights=str(checkpoint)))
        if any(r['prompt_relative_rms']>signature['gross_divergence_stop']['prompt_relative_rms']
               or r['generation_relative_error']>signature['gross_divergence_stop']['generation_relative_error']
               for r in diagnostics):
            raise FloatingPointError('Gross prompt/teacher divergence at a bounded checkpoint; stop before continuing')

    try:
        with (folder/f'train-from-{engine.completed}.jsonl').open('a') as log:
            while engine.completed < max(manifest['student']['steps']) and not stop:
                started = time.monotonic()
                update = engine.update(sampler.next())
                update['seconds'] = time.monotonic()-started
                history.append(update)
                log.write(__import__('json').dumps(update,allow_nan=False)+'\n'); log.flush()
                print(f"UPDATE {engine.completed} losses={update['losses']} seconds={update['seconds']:.2f}",flush=True)
                status(run,'training_lora',completed=engine.completed,total=660)
                if engine.completed in manifest['student']['steps']:
                    export()
                elif engine.completed % 30 == 0:
                    state.save(folder/'state.pt',engine,sampler,None,signature,history)
            if stop:
                export()
                status(run,'training_interrupted',completed=engine.completed)
                raise SystemExit(130)
            status(run,'lora_training_budget_complete',completed=engine.completed)
    except BaseException as exc:
        state.save(folder/'state.pt',engine,sampler,None,signature,history)
        if not isinstance(exc,SystemExit):
            status(run,'training_error',completed=engine.completed,error=f'{type(exc).__name__}: {exc}')
            write_json(run/'decision.json',dict(decision='failed_student_training',
                       completed=engine.completed,reason=f'{type(exc).__name__}: {exc}',
                       activation_teacher_supported=True))
        raise
    finally:
        set_scale(network,0.); network.detach()
        generator._merge_sliders(pipe,'cuda:0',[])
        pristine = generator._merge_state('cuda:0').pristine
        exact = all(torch.equal(m.weight.detach().cpu(),p) for m,p in pristine.items())
        write_json(folder/'off-restoration.json',dict(exact=exact,modules=len(pristine)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    train(parser.parse_args().run_dir)
