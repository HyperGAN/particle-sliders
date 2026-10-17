"""Neutral-conditioned TinyQwen run using the shared PG 0.9 trainer.

The native backend, routed bank and response adapter are reused. ConditionalPG09
owns every update, sampler and training checkpoint. This protocol deliberately
rejects the previous local PairedPG09 states and the archival Rp/VIC trainer.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
from importlib.metadata import version
import json
from pathlib import Path

import torch
from safetensors import safe_open

from particle_sliders.pg09 import PG_VERSION
from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic
from particle_sliders.pg09_text_video import TinyQwenResponseAdapter, TextResponseRow, require_continuous_vector
from particle_sliders.tiny_native import TinyParticleSlider
from .text_package import require_text_package

PROTOCOL = 'tiny-qwen-shared-package-v3'
CRITIC = 'ConditionalReadoutCritic/MLP/raw_logits'


def _portable_package_pin(identity):
    return {key: identity[key] for key in ('manifest_sha256', 'core_source_commit',
                                         'versions', 'wheel_sha256')}


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _vocab(backend):
    return dict(backend.tokenizer._vocab) if backend.dummy else None


def _inference_sources(backend):
    from particle_sliders.reference import RoutedMLP
    from particle_sliders.pg09_backends import AdapterParameters
    from particle_sliders.tiny_native import TinyParticleProjection
    classes = (type(backend), type(backend.model), TinyQwenResponseAdapter,
               AdapterParameters, TinyParticleSlider, TinyParticleProjection, RoutedMLP)
    return {cls.__module__: hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()
            for cls in classes}


def _validate_vocab(backend, vocab):
    if not backend.dummy:
        if vocab is not None:
            raise ValueError('live tokenizer cannot accept CPU-fixture vocabulary')
        return
    # DummyTokenizer allocates the next id from len(vocab)+1. Sparse/oversized
    # vocabularies can collide or index outside the frozen embedding table.
    config = backend.model.config
    limit = int(config['vocab_size'] if isinstance(config, dict) else config.vocab_size)
    if (not isinstance(vocab, dict) or any(not isinstance(k, str) or type(v) is not int
                                         for k, v in vocab.items())
            or sorted(vocab.values()) != list(range(1, len(vocab) + 1))
            or len(vocab) >= limit):
        raise ValueError('invalid CPU-fixture tokenizer state')


class PG09TinyRun:
    def __init__(self, backend, rows, *, seed=7, total_steps=1200,
                 batch_size=8, model_fingerprint=None, preset='bispf_optimized',
                 pg_source_profile='release', penalty_input_domain='auto'):
        from particle_sliders.pg09 import ConditionalPG09

        self.package_pin = _portable_package_pin(require_text_package(pg_source_profile=pg_source_profile))
        self.pg_source_profile = pg_source_profile

        self.backend = backend
        self.rows = deepcopy(rows)
        if len(rows) < 2 or any(not r.get('neutral') or not r.get('positive') or
                                r['neutral'] == r['positive'] for r in rows):
            raise ValueError('at least two distinct neutral and positive captions required')
        config = backend.model.config
        width = int(config['hidden_size'] if isinstance(config, dict) else config.hidden_size)
        # Validate the source/domain BEFORE attaching adapters or running a model.
        require_continuous_vector(width, 'TinyQwen', pg_source_profile=pg_source_profile,
                                  penalty_input_domain=penalty_input_domain)
        if not backend.dummy and not model_fingerprint:
            raise ValueError('live backend requires a verified staged-model fingerprint')
        self.model_fingerprint = model_fingerprint or tensor_digest(backend.model.state_dict())
        token_rows = [TextResponseRow(tuple(backend.encode(r['neutral']).ids),
                                     tuple(backend.encode(r['positive']).ids)) for r in rows]
        self.initial_vocab = _vocab(backend)
        torch.manual_seed(seed)
        # Preserve FP32 leaves even when the immutable host is BF16.
        self.network = TinyParticleSlider(backend.model, rank=8, alpha=8.).float()
        self.adapter = TinyQwenResponseAdapter(backend, self.network,
            model_identity=dict(model_fingerprint=self.model_fingerprint, dummy=bool(backend.dummy)),
            pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
        self.fixed = [self.adapter.prepare(row) for row in token_rows]
        conditions = torch.cat([torch.cat((r.neutral.flatten(),
                               r.neutral.new_tensor([r.row.scale, 0.])))[None] for r in self.fixed])
        self.critic = ConditionalReadoutCritic(width, conditions.shape[1]).to(self.network.particles.device)
        self.game = ConditionalPG09(self.adapter, self.fixed, self.critic,
            conditions=conditions, particles=self.network.particles, **self.adapter.domain_options,
            preset=preset, batch_size=batch_size, seed=seed, total_steps=total_steps)
        self.signature = dict(protocol=PROTOCOL, pg_version=PG_VERSION, core_version='0.6.0', seed=seed,
            package_pin=self.package_pin,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            recipe=asdict(self.game.recipe), model_fingerprint=self.model_fingerprint,
            rows=self.rows, initial_tokenizer_vocab=self.initial_vocab,
            critic=CRITIC, adapter_dtype='float32')

    def step(self):
        return self.game.step()

    @torch.no_grad()
    def sample(self, text, scale):
        ids = tuple(self.backend.encode(text).ids)
        # Reuse the exact adapter readout path used by the shared game.
        devices = [self.network.particles.device.index or 0] if self.network.particles.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            return self.adapter.render(TextResponseRow(ids, ids), scale=scale).detach().cpu()

    def state_dict(self):
        return deepcopy(dict(protocol=PROTOCOL, signature=self.signature,
                             tokenizer_vocab=_vocab(self.backend), game=self.game.state_dict()))

    def load_state_dict(self, state):
        if state.get('protocol') != PROTOCOL or state.get('signature') != self.signature:
            raise ValueError('PG0.9 protocol, model, conditioning, targets or recipe differ')
        vocab = state.get('tokenizer_vocab')
        _validate_vocab(self.backend, vocab)
        if self.backend.dummy and any(vocab.get(k) != v for k, v in self.initial_vocab.items()):
            raise ValueError('training tokenizer vocabulary differs')
        # Shared restore validates every field before mutating live game state.
        self.game.load_state_dict(state['game'])
        if self.backend.dummy:
            self.backend.tokenizer._vocab = dict(vocab)

    def export(self, path):
        # Reuse the shared checkpoint guard; failed partial updates cannot export.
        self.game.state_dict()
        self.network.save(path, dict(protocol=PROTOCOL, pg_version=PG_VERSION, core_version='0.6.0',
            package_pin=self.package_pin, pg_source_profile=self.pg_source_profile,
            resolved_recipe=asdict(self.game.recipe), model_fingerprint=self.model_fingerprint,
            student_conditioning='neutral', teacher_conditioning='positive',
            adapter_dtype='float32', critic=CRITIC,
            inference_sources=_inference_sources(self.backend),
            source_sha256=self.signature['source_sha256'], dummy=bool(self.backend.dummy),
            dummy_tokenizer_vocab=_vocab(self.backend),
            completed_steps=self.game.completed_steps, quality_qualified=False))


def load_export(backend, path, *, pg_source_profile='release'):
    """Validate before attaching; load FP32 leaves without host-dtype rounding."""
    path = Path(path)
    package_pin = _portable_package_pin(require_text_package(pg_source_profile=pg_source_profile))
    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata = json.loads((file.metadata() or {}).get('conceptmod', '{}'))
    expected = dict(protocol=PROTOCOL, pg_version=PG_VERSION, core_version='0.6.0', adapter_dtype='float32',
                    package_pin=package_pin, pg_source_profile=pg_source_profile,
                    student_conditioning='neutral', teacher_conditioning='positive', critic=CRITIC,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('not a qualified shared PG0.9 TinyQwen export')
    if not backend.dummy:
        raise ValueError('live export loading awaits staged fingerprint verification')
    if metadata.get('dummy') is not True:
        raise ValueError('export backend kind differs')
    if tensor_digest(backend.model.state_dict()) != metadata.get('model_fingerprint'):
        raise ValueError('frozen base fingerprint differs')
    if metadata.get('inference_sources') != _inference_sources(backend):
        raise ValueError('inference implementation sources differ')
    from particlegan import get_recipe
    recipe = metadata.get('resolved_recipe')
    if (version('particlegan') != PG_VERSION or version('particle-sliders-core') != '0.6.0'
            or not isinstance(recipe, dict)
            or recipe.get('name') not in ('bispf', 'bispf_optimized')
            or type(recipe.get('batch_size')) is not int or not 2 <= recipe['batch_size'] <= 512
            or type(recipe.get('total_steps')) is not int or recipe['total_steps'] < 1):
        raise ValueError('invalid PG0.9 export recipe')
    expected_recipe = asdict(get_recipe(recipe['name'], num_particles=128, z_dim=4,
                                       batch_size=recipe['batch_size'], total_steps=recipe['total_steps'],
                                       training_precision='float32'))
    if recipe != json.loads(json.dumps(expected_recipe)):
        raise ValueError('export recipe differs from the public preset')
    if (type(metadata.get('completed_steps')) is not int
            or not 0 <= metadata['completed_steps'] <= recipe['total_steps']):
        raise ValueError('invalid export step')
    vocab = metadata.get('dummy_tokenizer_vocab')
    _validate_vocab(backend, vocab)
    network, _ = TinyParticleSlider.load(backend.model, path, adapter_dtype=torch.float32)
    backend.tokenizer._vocab = dict(vocab)
    return network, metadata
