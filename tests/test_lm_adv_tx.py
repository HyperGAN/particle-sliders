"""Gates for the span-set transformer discriminator (``--adv_arch tx``).

No model weights, no GPU. Covers ``SpanTransformerD`` in
``conceptmod.textsliders.lm_adv``: the soundness constraints are all
load-bearing for the adversarial game, so each gets a test.
"""

from __future__ import annotations

import pytest
import torch

from conceptmod.textsliders import lm_adv
from conceptmod.textsliders.lm_adv import LMDiscriminator, SpanTransformerD


def _tiny() -> SpanTransformerD:
    torch.manual_seed(0)
    return SpanTransformerD(32, width=16, n_layers=2, n_heads=4)


def test_shapes_and_features():
    d = _tiny()
    seq = torch.randn(3, 9, 32)
    assert d(seq).shape == (3,)
    assert d.features(seq).shape == (3, 16)
    mask = torch.ones(3, 9, dtype=torch.bool)
    assert d(seq, mask).shape == (3,)
    assert d.features(seq, mask).shape == (3, 16)


def test_rejects_bad_hparams():
    with pytest.raises(ValueError):
        SpanTransformerD(0)
    with pytest.raises(ValueError):
        SpanTransformerD(32, width=0)
    with pytest.raises(ValueError):
        SpanTransformerD(32, n_layers=0)
    with pytest.raises(ValueError):
        SpanTransformerD(32, in_mode="fourier")


def test_padding_length_is_invisible():
    # Same two rows, different pad lengths: row scores must match (the
    # minibatch-stddev channel is computed over padding-invariant pooled
    # vectors, so it cannot leak pad length either).
    d = _tiny()
    s1 = torch.randn(1, 10, 32)
    s2 = torch.randn(1, 14, 32)

    def run(pad_to: int) -> torch.Tensor:
        pad = torch.zeros(2, pad_to, 32)
        pad[0, :10] = s1[0]
        pad[1, :14] = s2[0]
        m = torch.zeros(2, pad_to, dtype=torch.bool)
        m[0, :10] = True
        m[1, :14] = True
        return d(pad, m)

    a, b = run(14), run(22)
    assert torch.allclose(a, b, atol=1e-5)


def test_pad_values_are_invisible():
    # Whatever sits under the mask must not move the score (and therefore
    # contributes exactly 0 to the b_cap input-grad norm).
    d = _tiny()
    pad = torch.randn(2, 12, 32)
    m = torch.zeros(2, 12, dtype=torch.bool)
    m[0, :7] = True
    m[1, :12] = True
    dirty = pad.clone()
    dirty[0, 7:] = 99.0
    assert torch.allclose(d(pad, m), d(dirty, m), atol=1e-6)
    x = pad.clone().requires_grad_(True)
    d(x, m).sum().backward()
    assert bool((x.grad[0, 7:].abs().sum() == 0).item())
    assert bool((x.grad[0, :7].abs().sum() > 0).item())


def test_permutation_invariant_no_posemb():
    # No positional embeddings by design (span offsets differ per caption and
    # would be a spurious cue G cannot answer): permuting positions of the
    # same multiset must not move the score.
    d = _tiny()
    seq = torch.randn(2, 11, 32)
    perm = torch.randperm(11)
    assert torch.allclose(d(seq), d(seq[:, perm]), atol=1e-5)


def test_degenerate_sequence_is_finite():
    # A fresh LoRA's span delta is exactly zero: must score finite (the
    # trainer's DEGENERATE mask keeps it out of b_cap, but the forward must
    # not NaN first).
    d = _tiny()
    z = torch.zeros(2, 6, 32)
    out = d(z)
    assert bool(torch.isfinite(out).all())


def test_grad_flows_to_input_and_params():
    # G-use: score gradients must reach the fake sequence. D-use: params get
    # grads from the RpGAN pair loss (frozen-D path tested by zeroing instead).
    d = _tiny()
    x = torch.randn(2, 8, 32, requires_grad=True)
    m = torch.ones(2, 8, dtype=torch.bool)
    d(x, m).sum().backward()
    assert bool((x.grad.abs().sum() > 0).item())

    opt = torch.optim.SGD(d.parameters(), lr=0.1)
    opt.zero_grad()
    real = torch.randn(2, 8, 32)
    fake = torch.randn(2, 8, 32, requires_grad=True)
    loss = lm_adv.rp_d_loss(d(real, m), d(fake, m))
    loss.backward()
    assert all(p.grad is not None for p in d.parameters())
    assert bool((fake.grad.abs().sum() > 0).item())


def test_b_cap_double_backward_through_attention():
    # Regression: the b_cap penalty differentiates the input-grad norm w.r.t.
    # D params (create_graph through attention). The memory-efficient SDPA
    # backend has no double-backward formula, so the block is handwritten
    # matmul/softmax — this test pins that the penalty's backward reaches
    # every D param.
    d = _tiny()
    torch.manual_seed(2)
    real = torch.randn(2, 9, 32)
    fake = torch.randn(2, 9, 32)
    m = torch.ones(2, 9, dtype=torch.bool)
    pen, stat = lm_adv.cap_penalty(d, real, fake, mask_real=m, mask_fake=m)
    assert stat["applied"]
    opt = torch.optim.SGD(d.parameters(), lr=0.1)
    opt.zero_grad()
    pen.backward()
    # head.bias is provably Jacobian-invisible (final additive constant), so
    # the penalty cannot depend on it — same as the MLP head. Everything
    # else, attention weights included, must carry penalty grads.
    missing = [n for n, p in d.named_parameters() if p.grad is None]
    assert missing == ["head.bias"], f"unexpected grad gaps: {missing}"


def test_b_cap_runs_with_masks_and_ignores_padding():
    d = _tiny()
    torch.manual_seed(1)
    real = torch.randn(2, 9, 32)
    fake = torch.randn(2, 12, 32)
    m_real = torch.ones(2, 9, dtype=torch.bool)
    m_fake = torch.ones(2, 12, dtype=torch.bool)
    pen, stat = lm_adv.cap_penalty(d, real, fake, mask_real=m_real, mask_fake=m_fake)
    assert stat["applied"]
    assert float(pen.detach()) >= 0.0
    # Same rows with extra zero padding + extended mask-off tail: same penalty.
    real2 = torch.zeros(2, 15, 32)
    real2[:, :9] = real
    m_real2 = torch.zeros(2, 15, dtype=torch.bool)
    m_real2[:, :9] = True
    pen2, _ = lm_adv.cap_penalty(d, real2, fake, mask_real=m_real2, mask_fake=m_fake)
    assert abs(float(pen) - float(pen2)) < 1e-4


def test_mlp_rejects_masks_and_stem_parity():
    # The MLP head is last-token only; the shared stem refactor must not have
    # changed its math.
    torch.manual_seed(0)
    mlp = LMDiscriminator(32, hidden_dim=16, n_hidden=1)
    x = torch.randn(4, 32)
    with pytest.raises(ValueError):
        mlp(x, torch.ones(4, 1, dtype=torch.bool))
    with pytest.raises(ValueError):
        mlp.features(x, torch.ones(4, 1, dtype=torch.bool))
    assert torch.allclose(
        mlp._embed(x), lm_adv._stem_embed(x, "unit").to(mlp._embed(x).dtype)
    )


@pytest.mark.parametrize("readout", ["mean", "mean_last"])
def test_batched_d_matches_accumulated_single_row_g(readout):
    # A batch-dependent score is a shortcut unavailable to the rowwise G
    # graph, and also invalidates the per-sample gradient penalty.
    torch.manual_seed(7)
    d = SpanTransformerD(32, width=16, n_layers=1, readout=readout)
    seq = torch.randn(4, 9, 32, requires_grad=True)
    batched = d(seq)
    singles = torch.cat([d(row.unsqueeze(0)) for row in seq])
    assert torch.allclose(batched, singles, atol=1e-5)
    grad = torch.autograd.grad(batched[0], seq)[0]
    assert torch.count_nonzero(grad[1:]) == 0


def test_scaled_zero_fakes_have_finite_usable_cap_double_backward():
    torch.manual_seed(7)
    d = SpanTransformerD(32, width=16, n_layers=2, in_mode="scaled", readout="mean_last")
    real = torch.randn(4, 12, 32)
    d.calibrate_input_scale(real)
    fake = torch.zeros_like(real, requires_grad=True)
    g_loss = lm_adv.rp_g_loss(d(fake), d(real))
    grad = torch.autograd.grad(g_loss, fake)[0]
    assert torch.isfinite(grad).all()
    assert 0 < float(grad.norm()) < 100
    pen, stat = lm_adv.cap_penalty(d, real, fake)
    assert stat["fake_kept"] == 4
    assert 0 < stat["fake_grad_max"] < 100
    assert float(pen.detach()) < 100
    pen.backward()
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)


def test_mean_last_keeps_continue_token_separate_and_ignores_padding():
    torch.manual_seed(7)
    d = SpanTransformerD(32, width=16, n_layers=1, in_mode="scaled", readout="mean_last")
    seq = torch.randn(2, 9, 32)
    features = d.features(seq)
    assert features.shape == (2, 32)
    # Reordering lyric tokens leaves the result alone; moving the continue
    # token into the lyric set changes the distinguished last-token channel.
    lyric_order = torch.tensor([5, 3, 1, 6, 0, 2, 7, 4, 8])
    assert torch.allclose(features, d.features(seq[:, lyric_order]), atol=1e-5)
    moved = torch.tensor([8, 1, 2, 3, 4, 5, 6, 7, 0])
    assert not torch.allclose(features[:, 16:], d.features(seq[:, moved])[:, 16:])
    pad = torch.cat([seq, torch.randn(2, 6, 32) * 100], dim=1)
    mask = torch.arange(15).unsqueeze(0).expand(2, -1) < 9
    assert torch.allclose(features, d.features(pad, mask), atol=1e-5)
