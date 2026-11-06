"""Exercise the actual Music3 particle save/resume loop with a tiny CPU host."""
import json

import pytest
import torch
from safetensors.torch import load_file

from particle_sliders.textsliders import music3_particle_bridge as game
from particle_sliders.textsliders import particle_bridge_gan as shared
from particle_sliders.textsliders import train_lora_music3_particle as trainer
from test_lm_gan_trainer_integration import _TinyLM


def test_current_raw_export_exact_resume_and_old_shadow_refusal(tmp_path, monkeypatch):
    class Backend:
        def __init__(self, *args, **kwargs):
            self.model = _TinyLM().requires_grad_(False)
            self.dummy = True
            self.identity = {'test': 'synthetic-cpu-only'}

        def encode_text(self, style, lyrics):
            token = 1 if style.startswith('Quiet') else 2
            ids = torch.tensor([[token, 3, 4]])
            return ids, torch.ones_like(ids)

        def hidden(self, ids, mask, **kwargs):
            return self.model.model(input_ids=ids).last_hidden_state[:, -1]

    monkeypatch.setattr(game, 'Music3Backend', Backend)
    monkeypatch.setattr(game, 'RECIPE', dict(game.RECIPE))
    monkeypatch.setattr(shared, 'REFERENCE', dict(shared.REFERENCE))
    for name in ('initialize_ema', 'update_ema'):
        monkeypatch.setattr(shared, name, lambda *a: pytest.fail('weight EMA invoked'))
    prompts = tmp_path / 'prompts.yaml'
    prompts.write_text('rows:\n' + '\n'.join(
        f'  - neutral: Quiet guitar {i}\n    positive: Loud guitar {i}\n    lyrics: We carry crate {i}'
        for i in range(2)))
    common = ['--dummy', '--device', 'cpu', '--steps', '2', '--save_every', '1',
              '--name', 'raw-music3', '--critic', 'mlp', '--prompts_file', str(prompts)]
    first = tmp_path / 'split'
    trainer.train(trainer.parse_args(common + ['--save_dir', str(first), '--until', '1']))
    trainer.train(trainer.parse_args(common + ['--save_dir', str(first)]))
    full = tmp_path / 'full'
    trainer.train(trainer.parse_args(common + ['--save_dir', str(full)]))
    saved = torch.load(first / 'state.pt', weights_only=True)
    other = torch.load(full / 'state.pt', weights_only=True)
    assert saved['ema'] is None and saved['completed'] == 2
    assert saved['signature']['settings']['recipe']['ema'] is None
    assert all(torch.equal(v, other['network'][k]) for k, v in saved['network'].items())
    for suffix in ('last', 'live_last', 'step2', 'live_step2'):
        output = first / f'raw-music3_{suffix}.safetensors'
        assert all(torch.equal(v, load_file(output)[k]) for k, v in saved['network'].items())
        assert json.loads(output.with_suffix('.json').read_text())['weights_kind'] in ('raw', 'live')
    # A historical shadow-bearing checkpoint is retained and rejected before
    # constructing any host; its averaged values never become raw by default.
    saved['ema'] = saved['network']
    torch.save(saved, first / 'state.pt')
    monkeypatch.setattr(game, 'Music3Backend', lambda *a, **k: pytest.fail('host before EMA refusal'))
    with pytest.raises(ValueError, match='EMA checkpoint'):
        trainer.train(trainer.parse_args(common + ['--save_dir', str(first)]))
    assert torch.load(first / 'state.pt', weights_only=True)['ema'] is not None
    assert not torch.cuda.is_initialized()
