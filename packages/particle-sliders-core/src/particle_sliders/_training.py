"""Internal guards shared by caller-owned adversarial training loops."""
from contextlib import contextmanager

import torch


_FAILED_UPDATE = '_particle_sliders_failed_update'


def require_healthy_update(*owners):
    """A failed update may have changed weights before its counter committed."""
    if any(getattr(owner, _FAILED_UPDATE, False) for owner in owners):
        raise RuntimeError('failed training update; restore a full checkpoint or recreate the game')


def clear_update_failure(*owners):
    """Only a successful full-state restore may re-enable existing owners."""
    for owner in owners:
        setattr(owner, _FAILED_UPDATE, False)


@contextmanager
def guard_training_update(*owners):
    """Latch any interrupted D/G update, including optimizer/hook failures.

    This does not roll back weights or change update mathematics. Optimizers
    carry the latch for historical function-based games with no game object.
    """
    require_healthy_update(*owners)
    try:
        yield
    except BaseException:
        for owner in owners:
            setattr(owner, _FAILED_UPDATE, True)
        raise


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
    parameters = tuple(parameters)
    # Reentrant checkpoint backward reconstructs closed-over model parameters;
    # autograd.grad cannot observe those edges from the outer checkpoint node.
    # A retained ordinary backward supports both checkpoint implementations.
    # Preserve declared buffers and any other visible leaf inputs exactly.
    leaves = {id(parameter): parameter for parameter in parameters}
    if response.is_leaf:
        leaves[id(response)] = response
    pending = [response.grad_fn] if response.grad_fn is not None else []
    visited = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        leaf = getattr(node, "variable", None)
        if isinstance(leaf, torch.Tensor) and leaf.is_leaf:
            leaves[id(leaf)] = leaf
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    buffers = [(leaf, leaf.grad) for leaf in leaves.values()]
    try:
        for leaf, _ in buffers:
            leaf.grad = None
        response.sum().backward(retain_graph=True)
        if any(parameter.grad is None for parameter in parameters):
            raise ValueError("generator trainable or routed bank is disconnected from response")
    finally:
        for leaf, gradient in buffers:
            leaf.grad = gradient
