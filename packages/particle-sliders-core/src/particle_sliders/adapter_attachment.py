"""Failure rollback for adapter construction on an existing native host."""
from contextlib import contextmanager
from functools import wraps
from inspect import signature

import torch
from torch import nn


@contextmanager
def attachment_transaction(roots):
    """Restore original module ownership, forwards, grad flags and RNG on failure.

    This protects adapter attachment, which creates new parameters rather than
    updating base tensor values. It is not a weight-update transaction and must
    not overlap another owner mutating the same host.
    """
    modules, parameters, seen, devices = [], [], set(), set()
    for root in roots:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            attributes = dict(vars(module))
            containers = [(value, value.copy()) for value in attributes.values()
                          if isinstance(value, (dict, list, set))]
            modules.append((module, attributes, containers))
        for parameter in root.parameters():
            parameters.append((parameter, parameter.requires_grad))
            if parameter.device.type == 'cuda':
                devices.add(parameter.device.index)
    cpu_rng = torch.get_rng_state()
    cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
    try:
        yield
    except BaseException:
        for module, attributes, containers in reversed(modules):
            for container, contents in containers:
                container.clear()
                if isinstance(container, list):
                    container.extend(contents)
                else:
                    container.update(contents)
            vars(module).clear()
            vars(module).update(attributes)
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        torch.set_rng_state(cpu_rng)
        for device, state in cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        raise


def atomic_attachment(argument):
    """Guard a native module argument, or an explicit sequence of named hosts."""
    def decorate(function):
        parameters = signature(function)

        @wraps(function)
        def guarded(*args, **kwargs):
            bound = parameters.bind(*args, **kwargs)
            value = bound.arguments[argument]
            if isinstance(value, nn.Module):
                roots = [value]
            else:
                value = tuple(value)
                bound.arguments[argument] = value
                roots = [module for _, module in value]
            with attachment_transaction(roots):
                return function(*bound.args, **bound.kwargs)

        return guarded
    return decorate
