"""Isolated research rendering with the studio's exact full-delta merge code."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

import numpy as np
import torch

from .capture import ResidualCapture, pooled
from .specs import WORKSPACE, Observation, digest, sha, write_json, save_tensor, read_json


def pcm_sha(path):
    """Hash decoded samples, excluding the FLOAT WAV PEAK chunk timestamp."""
    import hashlib
    import soundfile as sf
    data, rate = sf.read(path, dtype='float32', always_2d=True)
    return hashlib.sha256(str((rate,data.shape)).encode()+data.tobytes()).hexdigest()


def studio_snapshot():
    with urllib.request.urlopen('http://127.0.0.1:7860/api/studio', timeout=10) as response:
        state = __import__('json').load(response)
    return dict(workers=state['workers'], loaded=state['loaded'],
                active=[dict(id=j['id'], device=j.get('device')) for j in state['active']],
                queued=len(state['queued']), keep_enabled=state['keep']['enabled'])


def resolve_styles(multipliers):
    sys.path.insert(0, str(WORKSPACE))
    from app import sliders
    energy = sum(abs(v) for v in multipliers.values())
    if energy > sliders.catalog()['energy']['language_model']['max']:
        raise ValueError('Requested effective multipliers exceed the host energy limit')
    if not multipliers:
        return []
    resolved = sliders.resolve([dict(id=k, scale=v/energy) for k,v in multipliers.items()],
                               host_energy={'language_model': energy, 'transformer': 0.})
    actual = sorted(float(c['multiplier']) for c in resolved)
    if len(actual) != len(multipliers) or not np.allclose(actual, sorted(multipliers.values()), rtol=0, atol=1e-8):
        raise ValueError('Host resolver changed the requested effective multipliers')
    return resolved


class ResearchRenderer:
    def __init__(self, run, manifest):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise RuntimeError('This investigation owns physical GPU 1; set CUDA_VISIBLE_DEVICES=1')
        self.run, self.manifest = Path(run), manifest
        self.device = 'cuda:0'
        sys.path.insert(0, str(WORKSPACE))
        from app import generator
        self.host = generator
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        self.ownership = studio_snapshot()
        if self.ownership['workers'] != 1:
            raise RuntimeError('Studio must have its single GPU-0 worker before research starts')
        started = time.monotonic()
        self.pipe = generator._ensure_loaded(self.device)
        self.load_seconds = time.monotonic()-started
        if generator._apply_mode(self.device) != 'merge':
            raise RuntimeError('Research contract requires a resident host and ordinary merged adapters')
        lm = self.pipe.language_model
        if len(lm.model.layers) != 36 or lm.config.hidden_size != 4096:
            raise ValueError('Unexpected composer topology; revise capture spec before collection')
        self.pipe.set_progress_bar_config(disable=True)

    def components(self, family, extra=None):
        components = [dict(c) for c in self.manifest['style_components'][family['family']]]
        if extra:
            components += [dict(extra)]
        effective = sum(abs(c['multiplier']*c['alpha']/c['rank']) for c in components)
        if effective > self.manifest['host_energy_max']+1e-8:
            raise ValueError('Composition exceeds the actual host energy limit')
        return components

    def generate(self, family, seed, path, *, arm=None, teacher=None, capture=False,
                 duration=None, extra=None):
        import soundfile as sf
        duration = duration or self.manifest['pilot']['render_cap_seconds']
        components = self.components(family, extra)
        started = time.monotonic()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        generator = torch.Generator(self.device).manual_seed(seed)
        rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state(), generator=generator.get_state(), numpy_seed=seed)
        torch.cuda.reset_peak_memory_stats()
        self.host._merge_sliders(self.pipe, self.device, components)
        torch.cuda.synchronize()
        setup = time.monotonic()-started
        kwargs = teacher.hook_kwargs() if teacher is not None else {}
        with ResidualCapture(self.pipe.language_model, capture=capture, **kwargs) as hooks:
            with torch.inference_mode():
                audio = self.pipe(prompt=family['caption'], lyrics=family['lyrics'], audio_duration=duration,
                                  generator=generator, num_inference_steps=self.manifest['sampler']['flow_steps'],
                                  output='audios')[0]
        torch.cuda.synchronize()
        wav = self.host._to_wav_array(audio)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wav, int(self.pipe.sampling_rate), subtype='FLOAT')
        timing = dict(total_seconds=time.monotonic()-started, setup_merge_seconds=setup,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), pipeline_load_seconds=self.load_seconds)
        trajectory = None
        if capture:
            trajectory = hooks.trajectory()
            trajectory.update(rng_before=rng, rng_after=dict(python=random.getstate(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state(), generator=generator.get_state()),
                              sample_rate=int(self.pipe.sampling_rate), audio_samples=len(wav),
                              nominal_audio_frames=round(duration*25), capture_spec=self.manifest['capture_spec'])
        return timing, trajectory

    def observe(self, family, seed, arm, scorer, teacher=None, capture=False, extra=None, full_song=False):
        ident = f"{family['family']}-s{seed}-{arm['name']}"
        record_path = self.run/'observations'/f'{ident}.json'
        cell_hash = digest({k: family[k] for k in ('caption', 'lyrics', 'style_multipliers')})
        identity = dict(manifest_sha256=digest(self.manifest), family_sha256=digest(family),
                        components=self.components(family, extra), physical_gpu=1,
                        gpu_name=torch.cuda.get_device_name(), sampler=self.manifest['sampler'],
                        capture_spec_sha256=digest(self.manifest['capture_spec']),
                        base_sha256=digest(self.manifest['model_hashes']), studio=self.ownership)
        if record_path.exists():
            previous = read_json(record_path)
            if previous['arm'] != arm or previous['provenance']['manifest_sha256'] != identity['manifest_sha256']:
                raise ValueError('Observation identity changed; cannot reuse')
            for label in ('audio', 'trajectory'):
                if previous.get(label) and previous.get(label+'_sha256') != sha(previous[label]):
                    raise ValueError(f'Corrupted saved {label}')
            if previous['status'] != 'running':
                return previous
            previous.update(status='failed', error='Interrupted render retained without reroll')
            write_json(record_path, previous)
            return previous
        observation = Observation(ident, family['family'], family['split'], seed, cell_hash, arm,
                                  status='running', provenance=identity)
        write_json(record_path, observation.json())
        print(f"RENDER {ident}", flush=True)
        try:
            audio_path = self.run/'audio'/f'{ident}.wav'
            timing, trajectory = self.generate(family, seed, audio_path, arm=arm, teacher=teacher,
                                               capture=capture, extra=extra,
                                               duration=family.get('render_cap_seconds') if full_song else None)
            observation.audio, observation.audio_sha256 = str(audio_path), sha(audio_path)
            observation.timing = timing
            if trajectory is not None:
                trajectory_path = self.run/'trajectories'/f'{ident}.pt'
                save_tensor(trajectory_path, trajectory)
                observation.trajectory, observation.trajectory_sha256 = str(trajectory_path), sha(trajectory_path)
            observation.reward = scorer.measure(audio_path, full_song=full_song)
            observation.status = 'complete' if observation.reward['valid'] else 'failed'
            observation.error = observation.reward.get('error')
        except BaseException as exc:
            observation.status, observation.error = 'failed', f'{type(exc).__name__}: {exc}'
            write_json(record_path, observation.json())
            if not isinstance(exc, Exception):
                raise
        write_json(record_path, observation.json())
        print(f"RESULT {ident}: {observation.status} CE={None if observation.reward is None else observation.reward['scalar']}", flush=True)
        return observation.json()

    def parity_audit(self, family):
        from .directions import RewardTeacher
        folder = self.run/'audit'; folder.mkdir(exist_ok=True)
        gen = torch.Generator().manual_seed(1729)
        vector = torch.randn(4096, generator=gen); vector /= vector.norm()
        teacher = RewardTeacher(11, .01, 100., vector, torch.zeros_like(vector), [])
        hashes, pcm_hashes, times, trajectories = {}, {}, {}, {}
        # Nonzero between zero runs catches stale hooks and altered RNG draws.
        for name, capture, intervention in [('plain', False, None), ('capture_zero', True, None),
                                            ('nonzero', False, teacher), ('zero_after', True, None)]:
            path = folder/f'{name}.wav'
            times[name], trajectory = self.generate(family, 4001, path, capture=capture, teacher=intervention, duration=2.)
            hashes[name] = sha(path)
            pcm_hashes[name] = pcm_sha(path)
            if trajectory is not None:
                trajectories[name] = trajectory
                save_tensor(folder/f'{name}.pt', trajectory)
        if len({pcm_hashes[name] for name in ('plain', 'capture_zero', 'zero_after')}) != 1:
            raise RuntimeError('Zero steering/capture lacks bit-exact audio parity')
        lm = self.pipe.language_model
        # Cancellation/exception cleanup is exercised against this actual model.
        try:
            with ResidualCapture(lm, capture=False, **teacher.hook_kwargs()):
                raise KeyboardInterrupt('declared cancellation audit')
        except KeyboardInterrupt:
            pass
        if getattr(lm, '_reward_capture_owner', None) is not None:
            raise RuntimeError('Cancellation leaked capture ownership')
        self.host._merge_sliders(self.pipe, self.device, [])
        state = self.host._merge_state(self.device)
        exact = all(torch.equal(module.weight.detach().cpu(), pristine) for module, pristine in state.pristine.items())
        if not exact:
            raise RuntimeError('Style Off did not restore exact base tensors')
        result = dict(passed=True, audio_hashes=hashes, pcm_hashes=pcm_hashes, timings=times, cancellation_cleanup=True,
                      exact_style_off_restoration=exact, physical_gpu=1,
                      isolation='CUDA_VISIBLE_DEVICES=1 in separate process; studio process restricted to GPU 0',
                      capture_frames={name: data['frame_embeds'].shape[1] for name,data in trajectories.items()})
        write_json(folder/'parity.json', result)
        return result
