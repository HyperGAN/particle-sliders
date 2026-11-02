"""Distill bounded CE latent targets into an ordinary acoustic attention LoRA."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .core import read, write, immutable, sha, digest, locked, IntegrityError
from .latent_transport import schedule, transported, velocity_loss, overlap_frames
from .merged_forward import MergedForward
from .acoustic_artifact import STRUCTURE, checkpoint, component


def train(home, game_home, folder, teacher_folder):
    import torch
    from diffusers import ModularPipeline
    from safetensors.torch import save_file
    from app import generator
    from app.lora_runtime import LoRANetwork
    from ..reward_sliders.specs import MODEL, save_tensor, validate_families
    from ..gan_v2.state import cpu
    from .setup import verify
    home = Path(home).resolve(); game_home = Path(game_home).resolve()
    folder = Path(folder).resolve(); teacher_folder = Path(teacher_folder).resolve()
    game, _ = verify(home); acoustic, _ = verify(game_home)
    recipe = read(folder/'recipe.json')
    for path, expected in recipe['source_hashes'].items():
        if sha(path) != expected:
            raise IntegrityError('Frozen transport training source changed')
    if not read(game_home/'audit/off-pcm-v1/result.json')['passed']:
        raise IntegrityError('Ordinary acoustic Off compatibility must pass')
    if read(game_home/'game.json')['parent_home'] != str(home) or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise IntegrityError('Wrong acoustic game or training GPU')
    result = read(teacher_folder/'result.json')
    if not result['passed'] or sha(teacher_folder/'result.json') != recipe['teacher_result_sha256']:
        raise IntegrityError('Teacher feasibility did not pass or changed')
    if not read(teacher_folder/'off-restoration.json')['exact']:
        raise IntegrityError('Teacher frozen-model check failed')
    cr = read(folder/'collection-recipe.json'); rows = read(folder/'captures.json')
    targets = {r['id']: r for r in read(teacher_folder/'targets.json')}
    cases = {c['id']: c for c in cr['cases']}
    families = list({c['family']['family']: c['family'] for c in cases.values()}.values())
    validate_families(families)
    if len(rows) != 8 or len(families) != 4 or any(f['split'] != 'train' for f in families):
        raise IntegrityError('Unexpected transport training cohort')
    immutable(folder/'training-prompts.json', dict(families=families,
        capture_collection_sha256=sha(folder/'collection-recipe.json'), teacher_recipe_sha256=sha(teacher_folder/'teacher-recipe.json')))
    data = []
    for row in rows:
        target = targets[row['id']]
        if sha(row['capture']) != row['capture_sha256'] or sha(target['target']) != target['target_sha256']:
            raise IntegrityError('Transport source or target bytes changed')
        capture = torch.load(row['capture'], map_location='cpu', weights_only=True)
        selected = torch.load(target['target'], map_location='cpu', weights_only=True)
        if selected['source_capture_sha256'] != row['capture_sha256'] or len(capture['chunks']) != 5:
            raise IntegrityError('Transport target does not match its capture')
        deltas = []
        for index, (chunk, latent) in enumerate(zip(capture['chunks'], selected['latents'])):
            delta = latent.float()-chunk['latent'].float()
            if float(delta.norm()/chunk['latent'].float().norm()) > .03 or bool(delta[..., :overlap_frames(capture, index)].count_nonzero()):
                raise IntegrityError('Teacher violated displacement or overlap constraints')
            deltas.append(delta)
        data.append((row, capture, deltas))
    with locked(folder/'train.lock'):
        torch.set_num_threads(4); torch.cuda.set_device(0); began = time.monotonic()
        pipe = ModularPipeline.from_pretrained(str(MODEL), local_files_only=True)
        pipe.load_components(names='transformer', pretrained_model_name_or_path=str(MODEL), local_files_only=True, dtype=torch.bfloat16)
        pipe.to('cuda:0'); tf = pipe.transformer.eval().requires_grad_(False); tf.enable_gradient_checkpointing()
        torch.manual_seed(9131)
        network = LoRANetwork(tf, rank=8, alpha=8., multiplier=1., target_replace=['MiniMaxMusic3Attention'],
            prefix='lora_unet', delimiter='-', train_method='full', attach=False).to('cuda:0').requires_grad_(True)
        if len(network.unet_loras) != 144:
            raise IntegrityError('Unexpected transport adapter topology')
        wrapper = MergedForward(network)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0003, betas=(.9, .999), weight_decay=0.)
        completed = 0; history = []; stopped = False; load_seconds = time.monotonic()-began
        def save():
            save_tensor(folder/'state.pt', dict(recipe_sha256=digest(recipe), network=cpu(network.state_dict()),
                optimizer=cpu(optimizer.state_dict()), completed=completed, history=history,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
        def export(step):
            path = folder/f'reward-ce-latent-transport_step{step}.safetensors'
            save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}, str(path))
            write(path.with_suffix('.json'), dict(STRUCTURE, weights_sha256=sha(path), prompts_file=str(folder/'training-prompts.json'), steps=step,
                reward=dict(name='reward-ce-latent-transport', method=recipe['method'], recipe_sha256=digest(recipe),
                    interpretation='Supervised transport of bounded CE-improved latent targets; fixed conditions and overlap; no full generator derivative')))
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
                    raise IntegrityError('Transport resume recipe changed')
                network.load_state_dict(state['network']); optimizer.load_state_dict(state['optimizer'])
                completed = state['completed']; history = state['history']
                torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
                if completed and not read(folder/'native-step1-audit.json')['passed']:
                    raise IntegrityError('Initial native audit must pass before resume')
            while completed < 80 and not stopped:
                began = time.monotonic(); epoch = completed//40; chunk_index = (completed % 40)//8; case_index = completed % 8
                row, capture, deltas = data[case_index]
                step_index = (5*chunk_index+10*epoch+3*case_index) % 30
                sched = schedule(capture); sigmas = sched.sigmas
                entry = capture['chunks'][chunk_index]['steps'][step_index]
                optimizer.zero_grad(set_to_none=True); loss_value = 0.; branch_inputs = []
                for branch in (0, 1):
                    source = entry['branches'][branch]
                    kwargs, target = transported(source, deltas[chunk_index], sigmas[step_index], sigmas[0], sigmas[-1], 'cuda:0')
                    prediction = tf(**kwargs)[0]
                    if completed == 0 and not torch.equal(prediction.detach().cpu(), source['velocity']):
                        raise IntegrityError('Initial zero-delta forward differs from recorded velocity')
                    loss = .5*velocity_loss(prediction, target, source['velocity'].to('cuda:0'))
                    if not torch.isfinite(loss):
                        raise IntegrityError('Nonfinite transport objective')
                    loss.backward(); loss_value += float(loss.detach()); branch_inputs.append(kwargs)
                    del prediction, target, loss
                gradients = [p.grad for p in network.parameters()]
                if len(gradients) != 288 or any(g is None or not torch.isfinite(g).all() for g in gradients):
                    raise IntegrityError('Missing or nonfinite transport factor gradients')
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                if not float(norm) > 0:
                    raise IntegrityError('Zero transport factor gradient')
                optimizer.step(); completed += 1
                metric = dict(step=completed, family=cases[row['id']]['family']['family'], seed=cases[row['id']]['seed'],
                    case_id=row['id'], chunk=chunk_index, flow_step=step_index, epoch=epoch,
                    objective=loss_value, gradient_norm=float(norm), seconds=time.monotonic()-began)
                history.append(metric)
                with (folder/'updates.jsonl').open('a') as file:
                    file.write(json.dumps(metric)+'\n')
                save()
                if completed in (1, 40, 80):
                    path = export(completed)
                    save_tensor(folder/f'state-step{completed}.pt', torch.load(folder/'state.pt', map_location='cpu', weights_only=True))
                if completed == 1:
                    with torch.no_grad():
                        differentiable = [tf(**kwargs)[0].detach().cpu() for kwargs in branch_inputs]
                    wrapper.detach(); generator._merge_sliders(pipe, 'cuda:0', [component(path)])
                    with torch.no_grad():
                        ordinary = [tf(**kwargs)[0].detach().cpu() for kwargs in branch_inputs]
                    exact = all(torch.equal(a, b) for a, b in zip(differentiable, ordinary))
                    generator._merge_sliders(pipe, 'cuda:0', [])
                    frozen = wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters())
                    audit = dict(passed=exact and frozen, actual_candidate_updates=1, discarded_updates=0,
                        zero_forward_exact=True, both_branch_native_merge_exact=exact, base_frozen=frozen,
                        factor_gradients_present_finite=288, gradient_norm=float(norm), new_audio_generated=0,
                        checkpoint_sha256=sha(path), interpretation='Native transport training/export integrity, not quality evidence')
                    immutable(folder/'native-step1-audit.json', audit)
                    if not audit['passed']:
                        raise IntegrityError('Actual transport native-merge audit failed')
                    wrapper.attach()
                write(folder/'status.json', dict(state='training', actual_updates=completed, total=80, load_seconds=load_seconds, updated_unix=time.time()))
                print('TRANSPORT UPDATE', metric, flush=True)
            save()
            if stopped:
                raise SystemExit(130)
            path = export(completed)
            write(folder/'status.json', dict(state='checkpoint_ready', actual_updates=completed, checkpoint=str(path),
                load_seconds=load_seconds, optimizer_seconds=sum(r['seconds'] for r in history)))
        except BaseException as exc:
            save(); write(folder/'error.json', dict(error=repr(exc), actual_updates=completed)); raise
        finally:
            wrapper.detach(); generator._merge_sliders(pipe, 'cuda:0', [])
            exact = wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters())
            write(folder/'off-restoration-80.json', dict(exact=exact, physical_gpu=1))
            if not exact:
                raise IntegrityError('Transport training changed frozen model weights')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('home', 'game-home', 'folder', 'teacher-folder'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); train(a.home, a.game_home, a.folder, a.teacher_folder)
