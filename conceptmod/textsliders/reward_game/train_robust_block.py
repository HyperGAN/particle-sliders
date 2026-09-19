"""Joint full-batch CE refinement of an ordinary block parent.

Reward credit is limited to the final two flow steps, with parent states fixed.
Partial batch gradients are saved so clean interruption can resume at a case
boundary without repeating already completed cases.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from .core import read, write, immutable, sha, digest, locked, IntegrityError
from .block_artifact import STRUCTURE, checkpoint, component
from .acoustic_tail import replay_latents, decode, decode_piece
from .ce_wave_gradient import value_and_gradient
from .merged_forward import MergedForward
from .robust_ce import value_and_weight


def train(home, folder):
    import torch
    import numpy as np
    import soundfile as sf
    from diffusers import ModularPipeline
    from safetensors.torch import load_file, save_file
    from app import generator
    from app.lora_runtime import LoRANetwork
    from ..reward_sliders.specs import MODEL, RewardSpec, save_tensor, validate_families
    from ..reward_sliders.reward import CEReward
    from ..gan_v2.state import cpu
    from .setup import verify
    home = Path(home).resolve(); folder = Path(folder).resolve()
    game, _ = verify(home/'block-v1'); recipe = read(folder/'recipe.json')
    for path, expected in recipe['source_hashes'].items():
        if sha(path) != expected:
            raise IntegrityError('Frozen joint block training source changed')
    if checkpoint(recipe['parent']['path'], 1.) != recipe['parent']:
        raise IntegrityError('Frozen block parent changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not read(home/'block-v1/audit/off-pcm-v1/result.json')['passed']:
        raise IntegrityError('Wrong training GPU or missing block Off compatibility')
    collection = read(folder/'collection-recipe.json'); rows = read(folder/'captures.json')
    if (read(folder/'collection-status.json')['state'] != 'complete' or len(rows) != 8
            or not read(folder/'collection-off-restoration.json')['exact'] or collection['parent'] != recipe['parent']):
        raise IntegrityError('Eight verified parent captures and exact Off required')
    cases = {c['id']: c for c in collection['cases']}
    families = list({c['family']['family']: c['family'] for c in cases.values()}.values())
    validate_families(families)
    if len(families) != 4 or any(f['split'] != 'train' for f in families):
        raise IntegrityError('Balanced parent training families required')
    for row in rows:
        if row['status'] != 'complete' or row['recipe_sha256'] != digest(collection):
            raise IntegrityError('Unverified parent capture')
        if any(sha(row[k]) != row[k+'_sha256'] for k in ('capture', 'audio')):
            raise IntegrityError('Parent capture bytes changed')
    immutable(folder/'training-prompts.json', dict(families=families, parent_prompt_sha256=recipe['parent']['prompt_sha256'],
        collection_recipe_sha256=sha(folder/'collection-recipe.json')))
    if (folder/'error.json').exists():
        raise IntegrityError('Failed scientific/engineering attempt retained; explicit amendment required')
    with locked(folder/'train.lock'):
        torch.set_num_threads(4); torch.cuda.set_device(0); began = time.monotonic()
        pipe = ModularPipeline.from_pretrained(str(MODEL), local_files_only=True)
        for name in ('transformer', 'vocoder'):
            pipe.load_components(names=name, pretrained_model_name_or_path=str(MODEL), local_files_only=True, dtype=torch.bfloat16)
        pipe.to('cuda:0'); tf = pipe.transformer.eval().requires_grad_(False)
        vocoder = pipe.vocoder.eval().requires_grad_(False); tf.enable_gradient_checkpointing()
        torch.manual_seed(9131)
        network = LoRANetwork(tf, rank=8, alpha=8., multiplier=1., target_replace=['MiniMaxMusic3TransformerBlock'],
            prefix='lora_unet', delimiter='-', train_method='full', attach=False).to('cuda:0').requires_grad_(True)
        network.load_state_dict(load_file(recipe['parent']['path']), strict=True)
        if len(network.unet_loras) != 216 or len(list(network.parameters())) != 432:
            raise IntegrityError('Unexpected jointly trainable block topology')
        wrapper = MergedForward(network)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, betas=(.9, .999), weight_decay=0.)
        scorer = CEReward(RewardSpec(**game['reward_spec'])); scorer.load()
        load_seconds = time.monotonic()-began; completed = 0; history = []; pending = []; stopped = False
        def save():
            save_tensor(folder/'state.pt', dict(recipe_sha256=digest(recipe), network=cpu(network.state_dict()),
                optimizer=cpu(optimizer.state_dict()), completed=completed, history=history, pending=pending,
                gradients={k: None if p.grad is None else p.grad.detach().cpu() for k, p in network.named_parameters()},
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
        def export(step):
            path = folder/f'reward-ce-robust-block_step{step}.safetensors'
            save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}, str(path))
            write(path.with_suffix('.json'), dict(STRUCTURE, weights_sha256=sha(path), prompts_file=str(folder/'training-prompts.json'),
                steps=step, reward=dict(name='reward-ce-robust-block', method=recipe['method'], recipe_sha256=digest(recipe),
                    parent_weights_sha256=recipe['parent']['weights_sha256'], interpretation=recipe['gradient_scope'], full_generation_gradient=False)))
            checkpoint(path, 1.); return path
        def cancel(*args):
            nonlocal stopped
            stopped = True
        signal.signal(signal.SIGTERM, cancel); signal.signal(signal.SIGINT, cancel)
        try:
            wrapper.attach()
            if (folder/'state.pt').exists():
                state = torch.load(folder/'state.pt', map_location='cpu', weights_only=True)
                if state['recipe_sha256'] != digest(recipe):
                    raise IntegrityError('Joint block resume recipe changed')
                network.load_state_dict(state['network']); optimizer.load_state_dict(state['optimizer'])
                completed = state['completed']; history = state['history']; pending = state['pending']
                for key, parameter in network.named_parameters():
                    gradient = state['gradients'][key]
                    parameter.grad = None if gradient is None else gradient.to(parameter)
                torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
            while completed < 2 and not stopped:
                if not pending:
                    optimizer.zero_grad(set_to_none=True)
                for row in rows[len(pending):]:
                    began = time.monotonic(); case = cases[row['id']]
                    capture = torch.load(row['capture'], map_location='cpu', weights_only=True)
                    with torch.no_grad():
                        latents, checks = replay_latents(tf, capture, record_checks=completed == 0)
                        wave = decode(vocoder, latents, capture['latent_hop_length'])
                    if completed == 0:
                        raw, rate = sf.read(row['audio'], dtype='float32', always_2d=True)
                        if not all(c['exact'] for c in checks) or not np.array_equal(wave[0].T.cpu().numpy(), raw):
                            raise IntegrityError('Initial jointly trainable forward changed parent waveform')
                    ce, wave_gradient = value_and_gradient(scorer, wave, capture['sampling_rate'])
                    if not 0 <= ce <= 10:
                        raise IntegrityError('Joint block CE outside valid range')
                    if completed == 0 and abs(ce-row['parent_ce']) > 1e-5:
                        raise IntegrityError('Differentiable parent CE differs from ordinary scoring')
                    robust_value, weight = value_and_weight(ce, case['reference_ce'])
                    current_wave = wave.detach().cpu(); del wave, latents
                    offset = 0; penalty_value = 0.; worst_latent = 0.; wave_error = 0.
                    for index in range(len(capture['chunks'])):
                        latents, _ = replay_latents(tf, capture, chunk_indices=(index,)); latent = latents[0]
                        reference = capture['chunks'][index]['latent'].to(latent)
                        fidelity = (latent.float()-reference.float()).square().mean()/reference.float().square().mean().clamp_min(1e-8)
                        relative = float(fidelity.detach().sqrt()); worst_latent = max(worst_latent, relative)
                        if relative > .1:
                            write(folder/'drift-stop.json', dict(update=completed+1, case_id=row['id'], chunk=index,
                                relative_latent_l2=relative, limit=.1, actual_updates=completed, completed_cases_in_pending_batch=len(pending)))
                            raise FloatingPointError('Predeclared joint block parent-latent drift limit')
                        piece = decode_piece(vocoder, latent, index, len(capture['chunks']), capture['latent_hop_length'])
                        length = piece.shape[-1]
                        difference = float((piece.detach().cpu()-current_wave[..., offset:offset+length]).abs().max())
                        wave_error = max(wave_error, difference)
                        if difference > 1e-6:
                            raise IntegrityError('Joint block gradient-mode waveform differs from ordinary values')
                        penalty = 10.*fidelity/len(capture['chunks'])
                        local = (-weight*(piece*wave_gradient[..., offset:offset+length].to(piece)).sum()+penalty)/8
                        if not torch.isfinite(local):
                            raise IntegrityError('Nonfinite joint block objective')
                        local.backward(); penalty_value += float(penalty.detach()); offset += length
                        print('ROBUST BLOCK CHUNK', completed+1, row['id'], index+1, flush=True)
                        del latents, latent, reference, fidelity, piece, penalty, local
                    if offset != current_wave.shape[-1]:
                        raise IntegrityError('Joint block gradient pieces do not tile waveform')
                    groups = {'attention': [], 'feed_forward': []}
                    for name, parameter in network.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise IntegrityError('Missing or nonfinite joint block factor gradient')
                        groups['attention' if '-attn-' in name else 'feed_forward'].append(float(parameter.grad.abs().sum()))
                    if any(not sum(values) > 0 for values in groups.values()):
                        raise IntegrityError('Both attention and feed-forward groups need nonzero CE gradients')
                    metric = dict(case_id=row['id'], family=case['family']['family'], seed=case['seed'], ce=ce,
                        off_ce=case['reference_ce'], parent_ce=row['parent_ce'], off_gain=ce-case['reference_ce'],
                        robust_ce_objective=robust_value, ce_pullback_weight=weight, latent_penalty=penalty_value,
                        worst_chunk_relative_latent_l2=worst_latent, gradient_forward_wave_max_error=wave_error,
                        accumulated_group_gradient_absolute_sum={k: sum(v) for k, v in groups.items()}, seconds=time.monotonic()-began)
                    pending.append(metric)
                    immutable(folder/'batches'/f'update{completed+1}'/f"{row['id']}.json", metric)
                    save()
                    write(folder/'status.json', dict(state='training', actual_updates=completed, total=2,
                        completed_cases_in_pending_batch=len(pending), cases_per_batch=8, load_seconds=load_seconds, updated_unix=time.time()))
                    print('ROBUST BLOCK CASE', completed+1, metric, flush=True)
                    if stopped:
                        break
                if stopped:
                    break
                if len(pending) != 8:
                    raise IntegrityError('Joint block optimizer requires every balanced training case')
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                if not float(norm) > 0:
                    raise IntegrityError('Zero joint block full-batch gradient')
                optimizer.step(); completed += 1
                metric = dict(step=completed, examples=8, objective=sum(r['robust_ce_objective']+r['latent_penalty'] for r in pending)/8,
                    mean_ce_gain=sum(r['off_gain'] for r in pending)/8, gradient_norm=float(norm),
                    worst_chunk_relative_latent_l2=max(r['worst_chunk_relative_latent_l2'] for r in pending),
                    gradient_forward_wave_max_error=max(r['gradient_forward_wave_max_error'] for r in pending),
                    seconds=sum(r['seconds'] for r in pending), cases=pending)
                history.append(metric); pending = []; save()
                with (folder/'updates.jsonl').open('a') as file:
                    file.write(json.dumps(metric)+'\n')
                path = export(completed)
                save_tensor(folder/f'state-step{completed}.pt', torch.load(folder/'state.pt', map_location='cpu', weights_only=True))
                # Both branches of an existing parent training probe check the
                # updated adapter against the same deployed native merger.
                capture = torch.load(rows[-1]['capture'], map_location='cpu', weights_only=True)
                probe = capture['chunks'][0]['steps'][29]['branches']
                kwargs = [dict(hidden_states=b['latent'].to('cuda:0'), timestep=b['timestep'].to('cuda:0'),
                    encoder_hidden_states=b['condition'].to('cuda:0'), return_dict=False) for b in probe.values()]
                with torch.no_grad():
                    differentiable = [tf(**k)[0].detach().cpu() for k in kwargs]
                wrapper.detach(); generator._merge_sliders(pipe, 'cuda:0', [component(path)])
                with torch.no_grad():
                    ordinary = [tf(**k)[0].detach().cpu() for k in kwargs]
                exact = all(torch.equal(a, b) for a, b in zip(differentiable, ordinary))
                generator._merge_sliders(pipe, 'cuda:0', [])
                frozen = wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters()) and all(p.grad is None for p in vocoder.parameters())
                audit = dict(passed=exact and frozen, both_branch_native_merge_exact=exact, base_vocoder_frozen=frozen,
                             actual_candidate_updates=completed, new_audio_generated=0, checkpoint_sha256=sha(path))
                immutable(folder/f'native-step{completed}-audit.json', audit)
                if not audit['passed']:
                    raise IntegrityError('Joint block updated native merger audit failed')
                wrapper.attach()
                write(folder/'status.json', dict(state='training', actual_updates=completed, total=2,
                    completed_cases_in_pending_batch=0, cases_per_batch=8, load_seconds=load_seconds, updated_unix=time.time()))
                print('ROBUST BLOCK UPDATE', metric, flush=True)
            save()
            if stopped:
                write(folder/'status.json', dict(state='interrupted', actual_updates=completed, total=2,
                    completed_cases_in_pending_batch=len(pending), clean_case_boundary=True))
                raise SystemExit(130)
            path = export(completed)
            write(folder/'status.json', dict(state='checkpoint_ready', actual_updates=completed, checkpoint=str(path),
                load_seconds=load_seconds, optimizer_seconds=sum(r['seconds'] for r in history), full_batch_examples=16))
        except BaseException as exc:
            save()
            if not (isinstance(exc, SystemExit) and exc.code == 130):
                write(folder/'error.json', dict(error=repr(exc), actual_updates=completed, pending_cases=len(pending)))
            raise
        finally:
            wrapper.detach(); generator._merge_sliders(pipe, 'cuda:0', [])
            exact = wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters()) and all(p.grad is None for p in vocoder.parameters())
            write(folder/'off-restoration-2.json', dict(exact=exact, physical_gpu=1))
            if not exact:
                raise IntegrityError('Joint block training failed exact Off restoration')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--home', required=True); p.add_argument('--folder', required=True)
    a = p.parse_args(); train(a.home, a.folder)
