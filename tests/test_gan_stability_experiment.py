import torch

from analysis.gan_bcap.stability_experiment import limit_loss_input_gradient


def test_feature_gradient_cap_preserves_value_and_calibrated_bound():
    x = torch.tensor([3., 4.], requires_grad=True)
    loss = x.square().sum()
    limited, before, factor = limit_loss_input_gradient(loss, x, input_scale=2., cap=1.)
    gradient = torch.autograd.grad(limited, x)[0]
    assert torch.equal(loss.detach(), limited.detach())
    assert before == 20. and factor == .05
    assert torch.allclose(gradient.norm() * 2., torch.tensor(1.))
    assert torch.allclose(gradient / gradient.norm(), x / x.norm())


def test_inactive_cap_returns_the_original_loss_and_gradient():
    x = torch.tensor([.01, .02], requires_grad=True)
    loss = x.square().sum()
    limited, _, factor = limit_loss_input_gradient(loss, x, input_scale=.5, cap=1.)
    assert limited is loss and factor == 1.
    assert torch.equal(torch.autograd.grad(limited, x)[0], 2 * x)
