"""Internal guards shared by caller-owned adversarial training loops."""
from contextlib import contextmanager

import torch


@contextmanager
def preserve_parameter_flags(module):
    """Restore caller-declared trainability after a temporary GAN phase freeze."""
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    try:
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def require_connected_response(response, parameters):
    """Reject missing native adversarial paths before an optimizer can mutate."""
    if not response.requires_grad:
        raise ValueError("detached generator response; adversarial path is inactive")
    gradients = torch.autograd.grad(response.sum(), tuple(parameters),
                                    retain_graph=True, allow_unused=True)
    if any(gradient is None for gradient in gradients):
        raise ValueError("generator trainable or routed bank is disconnected from response")
