import torch
import pytest
from analysis.gan_bcap.objective_20260905.conditional import ConditionalMetric, conditional_energy, backtrack


def test_match_is_stationary_for_both_players_and_at_zero():
    torch.manual_seed(21)
    d = ConditionalMetric(8,hidden=4).double()
    target = torch.randn(3,8,dtype=torch.float64)
    context = torch.randn_like(target)
    fake = target.clone().requires_grad_(True)
    loss = conditional_energy(d,fake,target,context,scale=1.)
    grads = torch.autograd.grad(loss,[fake,*d.parameters()])
    assert loss == 0
    assert all(torch.count_nonzero(g)==0 for g in grads)


def test_raw_skip_detects_errors_even_when_metric_is_constant():
    d = ConditionalMetric(8,hidden=4)
    with torch.no_grad():
        for p in d.parameters(): p.zero_()
    fake = torch.ones(3,8,requires_grad=True)
    loss = conditional_energy(d,fake,torch.zeros_like(fake),torch.ones_like(fake),scale=1.)
    assert loss>1
    assert torch.autograd.grad(loss,fake)[0].min()>0


def test_metric_projection_bounds_every_linear_operator():
    d = ConditionalMetric(8,hidden=4)
    with torch.no_grad():
        for p in d.parameters(): p.mul_(100)
    d.project()
    for module in d.modules():
        if isinstance(module,torch.nn.Linear):
            assert torch.linalg.matrix_norm(module.weight,ord=2)<=1.000001


def test_line_search_shortens_overshoot_and_restores_rejection():
    p = torch.nn.Parameter(torch.tensor(1.))
    result = backtrack([p],[p.detach().clone()],[torch.tensor(-3.)],lambda:p.square(),1.)
    assert result['fraction']==.25 and p==0 and result['loss']==0
    original = p.detach().clone()
    result = backtrack([p],[original],[torch.tensor(1.)],lambda:p.square(),0.)
    assert not result['accepted'] and p==original


def test_invalid_proposal_never_reaches_objective():
    p = torch.nn.Parameter(torch.tensor(1.))
    def fail(): raise AssertionError('Must not evaluate NaN state')
    result = backtrack([p],[p.detach().clone()],[torch.tensor(float('nan'))],fail,1.)
    assert not result['accepted'] and p==1


def test_perfect_conditional_logits_do_not_imply_perfect_guided_logits():
    # Inference applies the adapter on both branches. A conditional-only
    # objective is blind to this counterexample; CFG amplifies the omission.
    teacher=torch.zeros(2,2,dtype=torch.float64)
    student=teacher.clone();student[1]=torch.tensor([10.,-10.])
    assert torch.equal(student[0],teacher[0])
    def guided(pair):return pair[1]+1.5*(pair[0]-pair[1])
    logp=guided(teacher).log_softmax(-1)
    logq=guided(student).log_softmax(-1)
    assert (logp.exp()*(logp-logq)).sum()>4
    d=ConditionalMetric(2,hidden=4).double()
    assert conditional_energy(d,student,teacher,torch.ones_like(teacher),scale=1.)>1


def test_line_search_restores_parameters_when_evaluation_raises():
    p=torch.nn.Parameter(torch.tensor(1.))
    def fail():raise RuntimeError('Evaluator failed')
    with pytest.raises(RuntimeError,match='Evaluator failed'):
        backtrack([p],[p.detach().clone()],[torch.tensor(0.)],fail,1.)
    assert p==1


def test_no_first_order_game_rotation_at_a_conditional_match():
    torch.manual_seed(1)
    d=ConditionalMetric(2,hidden=4).double()
    real=torch.randn(1,2,dtype=torch.float64)
    fake=real.clone().requires_grad_(True)
    loss=conditional_energy(d,fake,real,torch.ones_like(real),scale=1.)
    gradient=torch.autograd.grad(loss,fake,create_graph=True)[0]
    mixed=torch.autograd.grad(gradient.sum(),list(d.parameters()),allow_unused=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in mixed)
