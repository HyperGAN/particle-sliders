import torch
import torch.nn.functional as F

from conceptmod.textsliders.lm_gan import feature_mean_surrogate, weighted_feature_mean, delayed_cosine_scale


def test_delayed_cosine_schedule():
    assert delayed_cosine_scale(0, 101) == 1.0
    assert delayed_cosine_scale(60, 101) == 1.0
    assert abs(delayed_cosine_scale(80, 101) - 0.525) < 1e-6
    assert delayed_cosine_scale(100, 101) == 0.05


def test_sequential_fm_matches_full_batch_value_and_gradient():
    torch.manual_seed(14)
    fake = torch.randn(4, 9, requires_grad=True)
    real = torch.randn(4, 9)
    weights = torch.tensor([0.05, 0.15, 0.3, 0.5])
    fm = weighted_feature_mean(fake, weights)
    rm = weighted_feature_mean(real, weights)
    exact = F.mse_loss(fm, rm)
    expected = torch.autograd.grad(exact, fake)[0]
    sequential = sum(weights[i] * feature_mean_surrogate(fake[i:i+1], fm, rm) for i in range(4))
    actual = torch.autograd.grad(sequential, fake)[0]
    torch.testing.assert_close(sequential, exact)
    torch.testing.assert_close(actual, expected)


def test_matching_modes_are_stationary_not_pulled_to_centroid():
    fake = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], requires_grad=True)
    mean = fake.detach().mean(dim=0)
    loss = sum(feature_mean_surrogate(fake[i:i+1], mean, mean) / 2 for i in range(2))
    grad = torch.autograd.grad(loss, fake)[0]
    assert loss.item() == 0
    assert torch.count_nonzero(grad).item() == 0
