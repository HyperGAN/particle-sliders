"""Explicit PG0.9 port: neutral-conditioned TinyQwen hidden-state sliders.

Uses the existing backend and routed adapter. The legacy positive-caption
Rp/Adam/VIC trainer is preserved. This new arm uses the released optimized
recipe and an explicit MLP critic; it cannot resume legacy optimizer state.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .particle_bridge_gan import ErrorCritic
from .pg09_game import PairedPG09
from .tiny_llm_particle import TinyParticleSlider

PROTOCOL = 'tiny-qwen-pg090-neutral-v1'


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PG09TinyRun:
    def __init__(self, backend, rows, *, seed=7, total_steps=1200,
                 batch_size=8, model_fingerprint=None):
        self.backend = backend
        self.rows = deepcopy(rows)
        if not rows or any(not r.get('neutral') or not r.get('positive') or
                           r['neutral'] == r['positive'] for r in rows):
            raise ValueError('distinct neutral and positive captions required')
        if not backend.dummy and not model_fingerprint:
            raise ValueError('live backend requires a verified staged-model fingerprint')
        self.model_fingerprint = model_fingerprint or tensor_digest(backend.model.state_dict())
        self.fixed = []
        with torch.no_grad():
            for row in rows:
                # Tokenize both prompts, but the student always receives neutral.
                positive_ids = backend.encode(row['positive']).ids
                neutral_ids = backend.encode(row['neutral']).ids
                self.fixed.append(dict(positive_ids=positive_ids, neutral_ids=neutral_ids,
                    positive=backend.teacher_hidden(positive_ids)[:, -1].float().cpu(),
                    neutral=backend.teacher_hidden(neutral_ids)[:, -1].float().cpu()))
        torch.manual_seed(seed)
        # Keep adapters FP32 even when the frozen pretrained base is BF16.
        self.network = TinyParticleSlider(backend.model, rank=8, alpha=8.).float()
        device = self.network.particles.device
        targets = torch.cat([r['positive'] for r in self.fixed]).to(device)
        neutrals = torch.cat([r['neutral'] for r in self.fixed]).to(device)
        self.critic = ErrorCritic(targets, neutrals=neutrals).to(device)
        self.game = PairedPG09(self.network, self.critic, self.critic.normalize(targets),
            particles=self.network.particles, preset='bispf_optimized', batch_size=batch_size,
            seed=seed, total_steps=total_steps)
        self.last_forward_rows = []
        self.signature = dict(protocol=PROTOCOL, pg_version='0.9.0',
            seed=seed,
            source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('tiny_llm_pg09.py','pg09_game.py','tiny_llm_particle.py',
                             'particle_bridge_gan.py','tiny_llm_backend.py')},
            recipe=asdict(self.game.recipe), critic='ErrorCritic/MLP/raw_logits',
            model_fingerprint=self.model_fingerprint, rows=self.rows,
            fixed_sha256=tensor_digest({f'{i}.{k}': r[k]
                for i,r in enumerate(self.fixed) for k in ('positive','neutral')}),
            neutral_ids=[r['neutral_ids'] for r in self.fixed],
            positive_ids=[r['positive_ids'] for r in self.fixed],
            adapter_dtype='float32')

    def predict(self, index, phase):
        row = self.fixed[index]
        self.last_forward_rows.append(dict(row=index, phase=phase, input_ids=list(row['neutral_ids'])))
        return self.critic.normalize(self.backend.hidden(row['neutral_ids'])[:, -1].float())

    def step(self):
        self.last_forward_rows = []
        return self.game.step(self.predict)

    @torch.no_grad()
    def sample(self, text, scale):
        ids = self.backend.encode(text).ids
        with self.network.scaled(scale):
            return self.backend.hidden(ids)[:, -1].float().detach().cpu()

    def state_dict(self):
        return deepcopy(dict(protocol=PROTOCOL, signature=self.signature,
                             game=self.game.state_dict()))

    def load_state_dict(self, state):
        if state.get('protocol') != PROTOCOL or state.get('signature') != self.signature:
            raise ValueError('PG0.9 protocol, model, conditioning, targets or recipe differ')
        self.game.load_state_dict(state['game'])

    def export(self, path):
        self.network.save(path, dict(protocol=PROTOCOL, pg_version='0.9.0',
            resolved_recipe=asdict(self.game.recipe), model_fingerprint=self.model_fingerprint,
            student_conditioning='neutral', teacher_conditioning='positive',
            adapter_dtype='float32', critic='ErrorCritic/MLP/raw_logits',
            source_sha256=self.signature['source_sha256'],
            dummy=bool(self.backend.dummy),
            dummy_tokenizer_vocab=(dict(self.backend.tokenizer._vocab) if self.backend.dummy else None),
            completed_steps=self.game.completed_steps, quality_qualified=False))


def load_export(backend, path):
    """Load the new FP32 adapter contract without BF16 intermediate rounding."""
    path=Path(path)
    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata=json.loads((file.metadata() or {}).get('conceptmod','{}'))
    if metadata.get('protocol') != PROTOCOL or metadata.get('adapter_dtype') != 'float32':
        raise ValueError('not a qualified PG0.9 TinyQwen export')
    expected = metadata['model_fingerprint']
    if backend.dummy and tensor_digest(backend.model.state_dict()) != expected:
        raise ValueError('frozen base fingerprint differs')
    if not backend.dummy:
        raise ValueError('live export loading awaits staged fingerprint verification')
    vocab=metadata.get('dummy_tokenizer_vocab')
    if (not metadata.get('dummy') or not isinstance(vocab,dict) or
            any(not isinstance(k,str) or type(v) is not int or v<1 for k,v in vocab.items()) or
            len(set(vocab.values()))!=len(vocab)):
        raise ValueError('invalid CPU-fixture tokenizer state')
    backend.tokenizer._vocab=dict(vocab)
    network=TinyParticleSlider(backend.model, rank=8, alpha=8.).float()
    if metadata['targets'] != network.target_names:
        raise ValueError('adapter targets differ')
    state=load_file(str(path),device='cpu')
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('invalid adapter tensor dtype or value')
    network.load_state_dict(state,strict=True)
    return network,metadata
