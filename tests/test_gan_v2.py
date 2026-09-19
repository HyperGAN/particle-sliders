"""Behavioral checks for gaps identified by the saved-critic architecture audit."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conceptmod.textsliders.gan_v2.critic import (
    SpanCritic, conditional_cap, matching_loss, full_width_guard, bounded_policy_kl)
from conceptmod.textsliders.gan_v2.optimization import (
    add_bounded_fm, factor_distance_squared, limit_effective_step, effective_distance,
    compress_factors, EffectiveEMA, cosine_rate)
from conceptmod.textsliders.gan_v2.engine import GANEngine, Recipe
from conceptmod.textsliders.lora import LoRAModule


def critic(**kwargs):
    torch.manual_seed(407)
    return SpanCritic(16, width=12, layers=1, heads=3, **kwargs)


def test_normalized_fm_has_no_learned_scale_escape():
    d = critic()
    x, c = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    a = d.features(x, condition=c)
    with torch.no_grad():
        d.out_norm.weight.mul_(100)
    b = d.features(x, condition=c)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert float(a.detach().norm(dim=-1).max()) <= (2*d.width)**.5


def test_relative_positions_observe_order_but_ignore_padding():
    d = critic()
    x, c = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    perm = torch.tensor([4, 3, 2, 1, 0, 5])
    assert not torch.allclose(d(x, condition=c), d(x[:, perm], condition=c[:, perm]), atol=1e-6)
    padded_x, padded_c = torch.nn.functional.pad(x, (0, 0, 0, 3)), torch.nn.functional.pad(c, (0, 0, 0, 3))
    mask = torch.arange(9)[None].expand(2, -1) < 6
    torch.testing.assert_close(d(x, condition=c), d(padded_x, mask, padded_c), atol=1e-5, rtol=1e-5)


def test_condition_affects_scores_and_single_rows_match_batch():
    d = critic()
    x, c = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    assert not torch.allclose(d(x, condition=c), d(x, condition=c.flip(0)))
    single = torch.cat([d(x[i:i+1], condition=c[i:i+1]) for i in range(2)])
    torch.testing.assert_close(single, d(x, condition=c), atol=1e-5, rtol=1e-5)


def test_conditional_cap_double_backward_with_zero_student_and_padding():
    d = critic()
    real, fake, c = torch.randn(2, 6, 16), torch.zeros(2, 6, 16), torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, -2:] = False
    loss, stats = conditional_cap(d, real, fake, mask, c, kappa=.01)
    loss.backward()
    assert stats['fake_kept'] == 2
    assert d.condition_proj.weight.grad is not None
    assert torch.isfinite(d.condition_proj.weight.grad).all()
    x = fake.requires_grad_(True)
    gradient = torch.autograd.grad(d(x, mask, c).sum(), x)[0]
    assert torch.count_nonzero(gradient[0, -2:]) == 0


def test_full_width_guard_sees_the_legacy_projection_nullspace():
    d = critic(conditioned=False, ordered=False)
    q = torch.linalg.qr(d.proj.weight.detach().T, mode='reduced').Q
    delta = torch.randn(1, 4, 16)
    delta = delta - (delta @ q) @ q.T
    delta = (4 * delta / delta.square().mean().sqrt()).requires_grad_(True)
    zero = torch.zeros_like(delta)
    torch.testing.assert_close(d(delta), d(zero), atol=1e-4, rtol=1e-4)
    loss = full_width_guard(delta, zero, 1., radius=1.)
    assert loss > 1
    assert torch.autograd.grad(loss, delta)[0].norm() > 0
    inside = (.001 * delta.detach()).requires_grad_(True)
    held = full_width_guard(inside, zero, 1., radius=1.)
    assert held == 0
    assert torch.autograd.grad(held, inside)[0].norm() == 0


def test_paired_component_detects_row_permutation_and_centroid_collapse():
    real = torch.tensor([[-2., 1.], [2., -1.]])
    for fake in (real.flip(0), real.mean(0).repeat(2, 1)):
        fake = fake.clone().requires_grad_(True)
        loss = sum(matching_loss(fake[i:i+1], fake.detach().mean(0), real.mean(0), real[i:i+1],
                                 paired_fraction=.1) / 2 for i in range(2))
        assert loss > 0
        assert torch.autograd.grad(loss, fake)[0].norm() > 0


def test_policy_guard_sees_changes_invisible_to_end_margin():
    teacher = torch.tensor([[8., 0., -2., -4., -3.]])
    student = torch.tensor([[-4., 8., 0., -2., -3.]], requires_grad=True)
    assert torch.equal(teacher[:, -1] - teacher[:, :-1].logsumexp(-1),
                       student[:, -1] - student[:, :-1].logsumexp(-1))
    loss, kl = bounded_policy_kl(student, teacher)
    assert kl.item() > 5 and loss > 5
    assert torch.autograd.grad(loss, student)[0].norm() > 0
    assert bounded_policy_kl(teacher, teacher)[0] == 0


def test_fm_limit_is_applied_to_accumulated_weighted_gradient():
    fm = torch.tensor([30., 40.])
    rest = torch.tensor([-3., 2.])
    out, stats = add_bounded_fm(rest, fm, 10.)
    torch.testing.assert_close(out - rest, torch.tensor([6., 8.]))
    assert stats['applied_norm'] == 10
    with pytest.raises(FloatingPointError):
        add_bounded_fm(rest, fm * float('nan'), 10.)


def test_effective_distance_and_cap_agree_with_dense_weights():
    torch.manual_seed(44)
    b, a = torch.randn(7, 3), torch.randn(3, 9)
    old_b, old_a = b.clone(), a.clone()
    b.add_(torch.randn_like(b)*.3)
    a.add_(torch.randn_like(a)*.3)
    expected = ((b @ a) - (old_b @ old_a)).double().square().sum()
    torch.testing.assert_close(factor_distance_squared(b,a,old_b,old_a), expected, rtol=1e-5, atol=1e-6)
    before = [(old_b, old_a, 2.)]
    result = limit_effective_step([(b,a,2.)], before, maximum=.15)
    assert result['actual_norm'] <= .15
    assert result['factor'] < 1
    torch.testing.assert_close(torch.tensor(result['actual_norm']), (2*b@a - 2*old_b@old_a).norm(), atol=1e-5, rtol=1e-4)


def test_effective_distance_is_factor_gauge_invariant():
    torch.manual_seed(45)
    b, a, db, da = torch.randn(7,3), torch.randn(3,9), torch.randn(7,3)*.1, torch.randn(3,9)*.1
    one = effective_distance([(b+db,a+da,1.)],[(b,a,1.)])
    two = effective_distance([(10*(b+db),(a+da)/10,1.)],[(10*b,a/10,1.)])
    assert one == pytest.approx(two, rel=1e-5)


def test_effective_ema_preserves_identical_adapters_with_opposite_factor_signs():
    torch.manual_seed(46)
    b, a = nn.Parameter(torch.randn(7,3)), nn.Parameter(torch.randn(3,9))
    m = SimpleNamespace(lora_name='test', lora_up=SimpleNamespace(weight=b),
                        lora_down=SimpleNamespace(weight=a), scale=2.)
    net = SimpleNamespace(unet_loras=[m])
    ema = EffectiveEMA(rank=4, decay=.5)
    ema.update(net, 0)
    with torch.no_grad(): b.neg_(); a.neg_()
    ema.update(net, 1)
    state = ema.inference_state()
    torch.testing.assert_close(state['test.lora_up.weight'] @ state['test.lora_down.weight'], 2*b@a, atol=1e-5, rtol=1e-5)
    restored = EffectiveEMA(rank=4, decay=.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.step == 1
    assert restored.last_error < 1e-6


def test_compression_reports_true_discarded_error():
    torch.manual_seed(47)
    b, a = torch.randn(7,5), torch.randn(5,9)
    left, right, result = compress_factors(b,a,2)
    observed = float((left@right - b@a).norm() / (b@a).norm())
    assert observed == pytest.approx(result['relative_frobenius_error'], rel=1e-5)


def test_schedule_has_fixed_origin_horizon_and_floor():
    opts=dict(origin=600,horizon=900)
    assert cosine_rate(600,**opts) == 1
    assert cosine_rate(780,**opts) == 1
    assert cosine_rate(900,**opts) == .05
    assert cosine_rate(1500,**opts) == .05


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(16,16,bias=False).requires_grad_(False)
        module = LoRAModule('tiny', self.base, lora_dim=3, alpha=3)
        module.apply_to()
        self.adapter = module
        self.unet_loras = [module]

    def forward(self,x):
        return self.base(x)


def make_engine(diagnostics):
    torch.manual_seed(50)
    network = TinyNetwork()
    d = SpanCritic(16,width=12,layers=1,heads=3)
    rows = []
    for i in range(3):
        x = torch.randn(1,4+i,16)
        base = network.base.org_forward(x) if hasattr(network.base,'org_forward') else network.adapter.org_forward(x)
        rows.append(dict(x=x,base=base.detach(),real=torch.randn_like(x)*.3,
                         condition=torch.randn_like(x),end_teacher=torch.zeros(x.shape[1]),
                         policy_teacher=torch.randn(x.shape[1],4)))
    def forward(row, policy):
        hidden = network(row['x'])
        return dict(fake=hidden-row['base'],end=hidden[0].mean(-1),policy=hidden[0,:,:4])
    recipe=Recipe(schedule_origin=0,schedule_horizon=10,diagnostics_every=diagnostics,
                  fm_gradient_limit=.01,effective_relative_limit=.02)
    return GANEngine(network,d,forward,rows,recipe)


def test_diagnostics_do_not_change_updates_and_limits_are_real():
    one, two = make_engine(0), make_engine(1)
    a, b = one.update([0,1,2]), two.update([0,1,2])
    for p,q in zip(one.network.parameters(),two.network.parameters()):
        torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert b['collapse'] is None and a['gradients'] is None
    assert set(b['gradients']['norms']) == {'adversarial','fm','end','full_width','policy'}
    assert b['fm_gradient']['applied_norm'] <= .01
    assert b['effective_limit']['actual_norm'] <= b['effective_limit']['maximum']
    assert any(p.grad is not None and p.grad.norm() > 0 for p in two.network.adapter.parameters())


def test_versioned_resume_restores_annealing_sampler_optimizers_and_ema(tmp_path):
    from conceptmod.textsliders.gan_v2 import state
    a,b=make_engine(1),make_engine(1)
    sa,sb=state.RowSampler(3,2,91),state.RowSampler(3,2,91)
    ea,eb=EffectiveEMA(rank=4),EffectiveEMA(rank=4)
    ea.update(a.network,0);eb.update(b.network,0)
    history_a=[];history_b=[]
    for _ in range(4):
        history_a.append(a.update(sa.next()));ea.update(a.network,a.completed)
    for _ in range(2):
        history_b.append(b.update(sb.next()));eb.update(b.network,b.completed)
    signature={'fixed_schedule':dict(origin=0,horizon=10),'test':1}
    path=tmp_path/'state.pt';state.save(path,b,sb,eb,signature,history_b)
    c=make_engine(1);sc=state.RowSampler(3,2,91);ec=EffectiveEMA(rank=4)
    history_c=state.restore(path,c,sc,ec,signature)
    for _ in range(2):
        history_c.append(c.update(sc.next()));ec.update(c.network,c.completed)
    assert history_a==history_c
    for key,value in a.network.state_dict().items():torch.testing.assert_close(value,c.network.state_dict()[key],atol=0,rtol=0)
    for key,value in ea.inference_state().items():torch.testing.assert_close(value,ec.inference_state()[key],atol=0,rtol=0)
    with pytest.raises(ValueError,match='configuration'):
        state.restore(path,c,sc,ec,dict(signature,fixed_schedule=dict(origin=0,horizon=20)))


def test_sampler_does_not_drop_the_last_partial_batch():
    from conceptmod.textsliders.gan_v2.state import RowSampler
    sampler=RowSampler(7,3,1)
    batches=[sampler.next() for _ in range(3)]
    assert [len(b) for b in batches]==[3,3,1]
    assert sorted(sum(batches,[]))==list(range(7))


def test_teacher_geometry_and_student_policy_alignment(tmp_path,monkeypatch):
    from conceptmod.textsliders.gan_v2 import data
    from test_lm_gan_trainer_integration import _TinyLM
    lm=_TinyLM().eval().requires_grad_(False)
    monkeypatch.setattr(data,'validate_prompts',lambda rows:None)
    monkeypatch.setattr(data.legacy,'_assemble',lambda caption,lyrics:caption)
    def tokenize(tokenizer,text,device):
        values=[1,10,11,12] if text=='neutral' else [2,2,10,11,12]
        tokens=torch.tensor([values]);return tokens,torch.ones_like(tokens)
    monkeypatch.setattr(data.legacy,'_tokenize',tokenize)
    monkeypatch.setattr(data.legacy,'_assert_last_token_is_audio_start',lambda *a,**k:None)
    def span(nt,nm,pt,pm,tokenizer,lyrics,**kwargs):
        assert lyrics=='new words'
        return torch.tensor([[0,1,1,0]]),torch.tensor([[0,0,1,1,0]])
    monkeypatch.setattr(data.legacy,'_assert_lyric_span',span)
    monkeypatch.setattr(data.legacy,'_frame_margins',lambda lm,hidden:hidden[0].float().mean(-1))
    monkeypatch.setattr(data,'allowed_logits',lambda lm,hidden:hidden[...,:4].float())
    rows=data.prepare_rows(lm,object(),[dict(neutral='neutral',positive='positive',lyrics='new words')],
        model_dir=tmp_path,cache_dir=tmp_path/'cache',device=torch.device('cpu'),frames=0,seeds=(7,),policy_stride=2)
    from conceptmod.textsliders.lora import LoRANetwork
    net=LoRANetwork(lm,rank=2,alpha=2,target_replace=['Qwen3Attention'],prefix='lora_te',train_method='full')
    result=data.StudentForward(lm,net,torch.device('cpu'))(rows[0],True)
    assert torch.count_nonzero(result['fake'])==0
    assert result['policy'].shape==rows[0]['policy_teacher'].shape
    assert result['continuation'].shape==rows[0]['continuation_teacher'].shape
    assert len(rows)==1 and rows[0]['history_seed']==7


def test_diversity_detects_duplicate_collapse_even_with_a_noisy_outlier():
    import numpy as np
    from conceptmod.textsliders.gan_v2.metrics import compare_diversity
    rng=np.random.default_rng(407)
    reference=rng.normal(size=(8,32))
    collapsed=np.tile(reference[0],(8,1));collapsed[-1]=reference[-1]*100
    result=compare_diversity(collapsed,reference,reference)
    assert result['status']=='collapse_suspected'
    assert result['candidate']['central_radius']<1e-10
    assert compare_diversity(reference,reference,reference)['status']=='no_collapse_detected'
    assert compare_diversity(reference[:2],reference[:2],reference[:2])['status']=='insufficient_evidence'


def test_quality_gate_keeps_absolute_condition_and_rejects_failures():
    from conceptmod.textsliders.gan_v2.metrics import clip_diagnostics,quality_decision
    neutral=dict(rms=.1,duration=20.,lyrics=.9,concept=-.2,hf14k_fraction=.01,clipped_fraction=0.)
    positive=dict(neutral,concept=.1)
    candidate=dict(neutral,concept=-.05)
    diagnostics=clip_diagnostics(candidate,neutral,positive)
    assert diagnostics['relative_description_gain']>0
    assert diagnostics['absolute_description_margin']<0
    assert not diagnostics['reaches_positive_description_reference']
    clips=[dict(prompt=str(i),diagnostics=diagnostics) for i in range(8)]
    diversity={str(i):dict(status='no_collapse_detected') for i in range(8)}
    assert quality_decision(clips,diversity)['decision']=='quality_unvalidated'
    bad=clip_diagnostics(dict(candidate,rms=0.,lyrics=0.),neutral,positive)
    bad_clips=[dict(prompt=str(i),diagnostics=bad) for i in range(8)]
    assert quality_decision(bad_clips,diversity)['decision']=='reject_candidate'


def test_training_and_evaluation_lyrics_must_be_disjoint():
    from conceptmod.textsliders.gan_v2.data import check_disjoint
    with pytest.raises(ValueError,match='overlap'):
        check_disjoint([dict(lyrics='same')],[dict(lyrics='same')])


def test_final_scheduled_update_receives_the_floor():
    engine=make_engine(0);engine.completed=9
    result=engine.update([0,1,2])
    assert result['step']==10 and result['lr_scale']==.05


def test_mixed_rank_renderer_detaches_the_old_adapter_exactly():
    from analysis.gan_bcap.render_v2 import detach_network
    torch.manual_seed(7)
    base=nn.Linear(8,8,bias=False)
    original=base.forward
    inputs=torch.randn(2,8);expected=base(inputs)
    for rank in (2,4,2):
        adapter=LoRAModule('test',base,lora_dim=rank,alpha=rank);adapter.apply_to()
        with torch.no_grad():adapter.lora_up.weight.normal_()
        assert not torch.equal(base(inputs),expected)
        detach_network(SimpleNamespace(unet_loras=[adapter]))
        assert base.forward==original
        torch.testing.assert_close(base(inputs),expected,rtol=0,atol=0)
