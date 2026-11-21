import pytest
import torch

from particle_sliders.prompt_steering import PromptSteering, psr_location_penalty, psr_mse_loss


def test_psr_uses_relu_and_per_example_strength_with_exact_off_positions():
    probe = PromptSteering(2)
    with torch.no_grad():
        probe.probe.weight.copy_(torch.tensor([[1., 0.]]))
        probe.probe.bias.fill_(-1)
        probe.direction.weight.copy_(torch.tensor([[0., 2.]]))
    residual = torch.tensor([[[3, 1], [4, 1]], [[0, 1], [5, 1]]], dtype=torch.bfloat16)
    mask = torch.tensor([[True, False], [True, True]])
    edited, coefficients = probe(residual, torch.tensor([0.5, -1.]), mask=mask)
    torch.testing.assert_close(coefficients, torch.tensor([[2., 0.], [0., 4.]]))
    torch.testing.assert_close(edited, torch.tensor([[[3, 3], [4, 1]], [[0, 1], [5, -7]]],
                                                   dtype=residual.dtype))
    off, _ = probe(residual, 0, mask=mask)
    assert torch.equal(off, residual)


def test_psr_mse_averages_examples_equally_and_sums_layers():
    target = torch.zeros((2, 3, 1), requires_grad=True)
    edited = torch.tensor([[[3.], [9.], [9.]], [[1.], [1.], [9.]]], requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss = psr_mse_loss([edited, edited * 2], [target, target], mask)
    assert loss.item() == 25
    loss.backward()
    assert target.grad is None
    assert torch.count_nonzero(edited.grad[~mask]) == 0
    assert psr_location_penalty(torch.tensor([[0.25, 0.25], [1., 1.]])).item() == 0.25
    with pytest.raises(ValueError, match="valid aligned position"):
        psr_mse_loss([edited], [target], torch.zeros_like(mask))


def test_a_psr_backpropagates_through_edited_layers_of_frozen_model():
    first, second = PromptSteering(2), PromptSteering(2)
    for module in (first, second):
        with torch.no_grad():
            module.probe.weight.zero_()
            module.probe.bias.fill_(0.5)
            module.direction.weight.copy_(torch.tensor([[0., 1.]]))
    frozen = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    with torch.no_grad():
        frozen.weight.copy_(torch.eye(2) * 2)
    plain = torch.tensor([[[1., 0.25], [2., 0.5]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    target_first = plain + torch.tensor([0., 1.5])
    target_second = frozen(target_first) + torch.tensor([0., 2.0])
    optimizer = torch.optim.SGD([*first.parameters(), *second.parameters()], lr=0.01)

    def objective():
        h1, lambda1 = first(plain, 1, mask=mask)
        h2, lambda2 = second(frozen(h1), 1, mask=mask)
        return (psr_mse_loss([h1, h2], [target_first, target_second], mask)
                + psr_location_penalty(lambda1) + psr_location_penalty(lambda2))

    loss = objective()
    loss.backward()
    for module in (first, second):
        assert torch.count_nonzero(module.probe.weight.grad)
        assert torch.count_nonzero(module.direction.weight.grad)
    assert frozen.weight.grad is None
    optimizer.step()
    assert objective() < loss
