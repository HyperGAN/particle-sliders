"""Construct bounded CE-improved acoustic targets from existing train captures."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .core import read, write, immutable, sha, digest, locked, IntegrityError
from .latent_transport import bounded_step, overlap_frames
from .acoustic_tail import decode, decode_piece
from .ce_wave_gradient import value_and_gradient


def build(home, folder):
    import numpy as np
    import soundfile as sf
    import torch
    from diffusers import ModularPipeline
    from ..reward_sliders.specs import MODEL, RewardSpec, save_tensor, validate_families
    from ..reward_sliders.reward import CEReward
    from .setup import verify
    from .resources import gpu_lease
    home = Path(home).resolve(); folder = Path(folder).resolve()
    game, _ = verify(home)
    recipe = read(folder/'teacher-recipe.json')
    for path, expected in recipe['source_hashes'].items():
        if sha(path) != expected:
            raise IntegrityError('Frozen teacher source changed')
    collection = Path(recipe['collection_folder'])
    if sha(collection/'collection-recipe.json') != recipe['collection_recipe_sha256']:
        raise IntegrityError('Teacher collection changed')
    cr = read(collection/'collection-recipe.json'); rows = read(collection/'captures.json')
    if len(rows) != 8 or not read(collection/'collection-off-restoration.json')['exact']:
        raise IntegrityError('Need eight verified training captures and exact Off')
    cases = {c['id']: c for c in cr['cases']}
    validate_families(list({c['family']['family']: c['family'] for c in cases.values()}.values()))
    if any(c['family']['split'] != 'train' for c in cases.values()):
        raise IntegrityError('Teacher data must be training-only')
    for row in rows:
        if row['status'] != 'complete' or row['recipe_sha256'] != digest(cr):
            raise IntegrityError('Unverified teacher source')
        for key in ('audio', 'capture'):
            if sha(row[key]) != row[key+'_sha256']:
                raise IntegrityError('Teacher source bytes changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise IntegrityError('Teacher owns physical GPU 1')
    if (folder/'attempt.json').exists():
        raise IntegrityError('Teacher attempt already exists; retain interruption and explicitly amend')
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM, cancel); signal.signal(signal.SIGINT, cancel)
    with locked(folder/'teacher.lock'), gpu_lease(home, 1):
        immutable(folder/'attempt.json', dict(started_unix=time.time(), recipe_sha256=digest(recipe)))
        torch.set_num_threads(4); torch.cuda.set_device(0); began = time.monotonic()
        pipe = ModularPipeline.from_pretrained(str(MODEL), local_files_only=True)
        pipe.load_components(names='vocoder', pretrained_model_name_or_path=str(MODEL), local_files_only=True, dtype=torch.bfloat16)
        pipe.to('cuda:0'); vocoder = pipe.vocoder.eval().requires_grad_(False)
        scorer = CEReward(RewardSpec(**game['reward_spec'])); scorer.load()
        results = []; target_steps = 0
        try:
            for row in rows:
                case = cases[row['id']]
                capture = torch.load(row['capture'], map_location='cpu', weights_only=True)
                reference = [c['latent'].to('cuda:0') for c in capture['chunks']]
                current = [x.clone() for x in reference]
                with torch.no_grad():
                    wave = decode(vocoder, current, capture['latent_hop_length'])
                raw, rate = sf.read(row['audio'], dtype='float32', always_2d=True)
                if rate != capture['sampling_rate'] or not np.array_equal(wave[0].T.cpu().numpy(), raw):
                    raise IntegrityError('Latent target baseline differs from original raw PCM')
                baseline, wave_gradient = value_and_gradient(scorer, wave, rate)
                if abs(baseline-case['reference_ce']) > 1e-5:
                    raise IntegrityError('Teacher baseline CE differs from frozen reference')
                selected = [x.detach().cpu() for x in current]; selected_ce = baseline; selected_step = 0
                history = []; case_folder = folder/'targets'/row['id']; case_folder.mkdir(parents=True, exist_ok=True)
                for step in range(1, 5):
                    offset = 0; gradients = []; max_wave_error = 0.
                    for index, value in enumerate(current):
                        leaf = value.float().detach().requires_grad_(True)
                        piece = decode_piece(vocoder, leaf, index, len(current), capture['latent_hop_length'])
                        length = piece.shape[-1]
                        error = float((piece.detach()-wave[..., offset:offset+length]).abs().max())
                        max_wave_error = max(max_wave_error, error)
                        if error > 1e-6:
                            raise IntegrityError('Latent gradient-mode decoder differs from ordinary waveform')
                        gradient, = torch.autograd.grad(piece, leaf, wave_gradient[..., offset:offset+length].to(piece))
                        gradients.append(gradient.detach()); offset += length
                        del leaf, piece, gradient
                    if offset != wave.shape[-1]:
                        raise IntegrityError('Teacher derivative pieces do not tile the waveform')
                    proposed = [bounded_step(r, x, g, overlap_frames(capture, i))
                                for i, (r, x, g) in enumerate(zip(reference, current, gradients))]
                    current = [x for x, drift in proposed]
                    del gradients
                    with torch.no_grad():
                        wave = decode(vocoder, current, capture['latent_hop_length'])
                    ce, wave_gradient = value_and_gradient(scorer, wave, rate)
                    if not 0 <= ce <= 10:
                        raise IntegrityError('Teacher CE outside declared valid range')
                    audio = case_folder/f'target-step{step}.wav'
                    sf.write(audio, wave[0].T.cpu().numpy(), rate, subtype='FLOAT')
                    state = case_folder/f'target-step{step}.pt'
                    save_tensor(state, dict(latents=[x.cpu() for x in current], source_capture_sha256=row['capture_sha256'],
                                            teacher_recipe_sha256=digest(recipe), step=step, ce=ce))
                    observation = dict(id=f"{row['id']}-latent-target-{step}", audio=str(audio), audio_sha256=sha(audio),
                        physical_gpu=1, family=case['family']['family'], seed=case['seed'],
                        source_audio_sha256=row['audio_sha256'], source_capture_sha256=row['capture_sha256'],
                        reward=dict(scalar=ce), baseline_ce=baseline, target_step=step,
                        relative_latent_l2=[drift for x, drift in proposed], gradient_forward_wave_max_error=max_wave_error,
                        target_state=str(state), target_state_sha256=sha(state), new_clips=1,
                        independent_training_recordings_added=0, interpretation='Optimized latent training target; not an ordinary LoRA sample or independent musical example')
                    immutable(folder/'observations'/f"{row['id']}-step{step}.json", observation)
                    history.append(observation); target_steps += 1
                    if ce > selected_ce:
                        selected_ce = ce; selected_step = step; selected = [x.detach().cpu() for x in current]
                    write(folder/'status.json', dict(state='building_targets', completed_cases=len(results),
                        case_id=row['id'], target_step=step, actual_target_ascent_steps=target_steps, actual_updates=0))
                    print('LATENT TEACHER', row['id'], step, ce, 'delta', ce-baseline, flush=True)
                target = case_folder/'selected.pt'
                save_tensor(target, dict(latents=selected, source_capture_sha256=row['capture_sha256'],
                                        teacher_recipe_sha256=digest(recipe), selected_step=selected_step, ce=selected_ce))
                result = dict(id=row['id'], target=str(target), target_sha256=sha(target), selected_step=selected_step,
                              baseline_ce=baseline, selected_ce=selected_ce, gain=selected_ce-baseline,
                              source_capture_sha256=row['capture_sha256'], history=history)
                immutable(case_folder/'result.json', result); results.append(result)
            mean = sum(r['gain'] for r in results)/8
            meaningful = sum(r['gain'] >= .02 for r in results)
            passed = meaningful >= 6 and mean >= .05
            immutable(folder/'targets.json', results)
            result = dict(passed=passed, meaningful_training_targets=meaningful, mean_training_gain=mean,
                          selected_steps=[r['selected_step'] for r in results], new_audio_generated=32,
                          independent_training_recordings_added=0, actual_target_ascent_steps=target_steps,
                          actual_updates=0, seconds=time.monotonic()-began, teacher_recipe_sha256=digest(recipe),
                          interpretation='Teacher feasibility only. No adapter improvement or preservation claim.')
            immutable(folder/'result.json', result)
            write(folder/'status.json', dict(state='targets_ready' if passed else 'rejected_teacher_feasibility', **result))
            return result
        except BaseException as exc:
            write(folder/'error.json', dict(error=repr(exc), completed_cases=len(results), actual_target_ascent_steps=target_steps))
            raise
        finally:
            exact = all(p.grad is None and not p.requires_grad for p in vocoder.parameters())
            write(folder/'off-restoration.json', dict(exact=exact, transformer_loaded=False,
                  base_vocoder_weights_unmodified=True, physical_gpu=1, actual_optimizer_updates=0))
            if not exact:
                raise IntegrityError('Teacher changed frozen vocoder gradient state')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--home', required=True); p.add_argument('--folder', required=True)
    a = p.parse_args(); print(json.dumps(build(a.home, a.folder), indent=2))
