"""Counterexamples the research distribution objective must distinguish."""
import pytest
import torch

from analysis.gan_bcap.objective_20260905.distribution import energy_distance, paired_energy, rbf_mmd,paired_mmd


def test_conditional_correspondence_cannot_be_replaced_by_marginal_matching():
    real = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    swap = real.flip(0)
    assert energy_distance(swap,real,unbiased=False).abs() < 1e-12
    assert paired_energy(swap,real) > 3


def test_exact_conditional_match_is_stationary_including_the_origin():
    for real in (torch.zeros(3,7),torch.randn(3,7)):
        fake = real.clone().requires_grad_(True)
        loss = paired_energy(fake,real)
        assert loss == 0
        assert torch.equal(torch.autograd.grad(loss,fake)[0],torch.zeros_like(fake))


def test_smoothed_score_is_differentiable_and_bounded_at_large_error():
    torch.manual_seed(1)
    real = torch.randn(3,7,dtype=torch.float64)
    fake = torch.randn(3,7,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:paired_energy(x,real),fake)
    huge = (1e6*fake.detach()).requires_grad_(True)
    grad = torch.autograd.grad(paired_energy(huge,real),huge)[0]
    assert grad.norm() <= 2/(21**.5) + 1e-12


def test_vstat_and_ustat_have_different_contracts():
    x = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    assert rbf_mmd(x,x,unbiased=False).abs() < 1e-12
    assert rbf_mmd(x,x,unbiased=True) < 0
    # The U statistic assumes independent draws, and cannot estimate a
    # within-population pair term from only one sample.
    with pytest.raises(ValueError):
        energy_distance(x[:1],x)


def test_empirical_distribution_distance_rejects_mean_collapse():
    x = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    assert energy_distance(torch.zeros_like(x),x,unbiased=False) > .9
    assert rbf_mmd(torch.zeros_like(x),x,unbiased=False) > .1


def test_calibration_is_fixed_and_scale_covariant():
    a,b = torch.randn(2,3,dtype=torch.float64),torch.randn(2,3,dtype=torch.float64)
    assert torch.allclose(paired_energy(a,b,scale=2.),paired_energy(3*a,3*b,scale=6.))
    with pytest.raises(ValueError):
        paired_energy(a,b,scale=torch.tensor(1.,requires_grad=True))
    with pytest.raises(ValueError):
        paired_energy(a,b,scale=0.)


def test_conditional_kernel_is_the_same_mmd_on_singletons():
    a=torch.tensor([[.2,-.7]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.3,1.]],dtype=torch.float64)
    assert torch.allclose(paired_mmd(a,b),rbf_mmd(a/(2**.5),b/(2**.5),unbiased=False))
    assert torch.autograd.gradcheck(lambda x:paired_mmd(x,b),a)
    match=b.clone().requires_grad_(True)
    value=paired_mmd(match,b)
    assert value==0 and torch.count_nonzero(torch.autograd.grad(value,match)[0])==0


def test_energy_powers_retain_proper_empirical_match_and_reject_invalid_power():
    a=torch.tensor([[-1.],[1.]],dtype=torch.float64)
    for power in (.25,.5,1.,1.5):
        assert energy_distance(a,a,power=power,unbiased=False).abs()<1e-12
        assert energy_distance(torch.zeros_like(a),a,power=power,unbiased=False)>0
    with pytest.raises(ValueError):energy_distance(a,a,power=2.)
