"""Small CPU fault injection; no pretrained models or weight averaging."""
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from particle_sliders.textsliders import yue2_uni
from particle_sliders.textsliders.lm_adv import SpanTransformerD


def test_uni16_partial_update_latches_optimizer_pair_and_restores_flags(monkeypatch):
    torch.manual_seed(7)
    class Network(nn.Linear):
        @contextmanager
        def scaled(self, scale):
            yield
    network = Network(3, 3, bias=False)
    critic = SpanTransformerD(3, width=4, n_layers=1, n_heads=1,
                             in_mode='scaled', readout='mean_last')
    rows = [dict(x=torch.randn(1, 2, 3), real=torch.randn(1, 2, 3),
                 end_teacher=torch.zeros(1, 2)) for _ in range(2)]
    critic.calibrate_input_scale(torch.cat([row['real'] for row in rows]))
    def student(backend, row, checkpointing):
        value = network(row['x'])
        return value, value.mean(-1)
    monkeypatch.setattr(yue2_uni, 'student', student)
    g = torch.optim.Adam(network.parameters(), lr=.001)
    d = torch.optim.Adam(critic.parameters(), lr=.001)
    before = [p.detach().clone() for p in critic.parameters()]
    flags = [p.requires_grad for p in critic.parameters()]
    def fail_generator():
        raise RuntimeError('injected G optimizer failure')
    with monkeypatch.context() as patch:
        patch.setattr(g, 'step', fail_generator)
        with pytest.raises(RuntimeError, match='injected G optimizer failure'):
            yue2_uni.update(None, network, critic, g, d, rows)
    assert any(not torch.equal(p, old) for p, old in zip(critic.parameters(), before))
    assert [p.requires_grad for p in critic.parameters()] == flags
    with pytest.raises(RuntimeError, match='failed training update'):
        yue2_uni.update(None, network, critic, g, d, rows)
