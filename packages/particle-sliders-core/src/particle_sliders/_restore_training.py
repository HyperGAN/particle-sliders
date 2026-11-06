"""Rollback for historical full-training-state loaders, without cloning models."""
from contextlib import contextmanager
from copy import deepcopy
import random
import sys

import torch


_FAILED = '_particle_sliders_failed_restore'
_MISSING = object()


def require_healthy_restore(*owners):
    if any(getattr(owner, _FAILED, False) for owner in owners):
        raise RuntimeError('failed training restore rollback; restore a full checkpoint or recreate the game')


def validate_resume_rng(torch_rng=None, python_rng=None, cuda_rng=None):
    """Reject malformed CPU state / incompatible CUDA topology before mutation."""
    if torch_rng is not None:
        torch.Generator().set_state(torch_rng.cpu())
    if python_rng is not None:
        random.Random().setstate(python_rng)
    if cuda_rng:
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError('Resume requires the same number of visible CUDA devices')
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
               or value.ndim != 1 for value in cuda_rng):
            raise ValueError('Invalid saved CUDA RNG state')


def _unique(values):
    return list({id(value): value for value in values if value is not None}.values())


def _copy_attribute(value):
    return value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)


def _quarantine(owners, optimizers):
    for owner in _unique([*owners, *optimizers]):
        setattr(owner, _FAILED, True)
    for optimizer in optimizers:
        # Function-based legacy loops have no owning game object to gate.
        # Keep their optimizer unusable even if a caller catches this error.
        if not hasattr(optimizer, '_particle_sliders_restore_guard'):
            optimizer._particle_sliders_restore_guard = optimizer.register_step_pre_hook(
                lambda opt, args, kwargs: require_healthy_restore(opt))


@contextmanager
def restore_training_state(*, modules=(), optimizers=(), stateful=(), attributes=(), owners=()):
    """Apply a live restore; on error restore every captured component and RNG.

    Snapshots contain owned module tensors and optimizer state, never a cloned
    backend or pretrained model. A failed rollback preserves the primary error,
    attempts the remaining restorers, and quarantines optimizer updates/saves.
    """
    modules, optimizers, stateful = map(_unique, (modules, optimizers, stateful))
    owners = _unique([*owners, *optimizers])
    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    attributes = [*attributes, *((opt.record, key) for opt in optimizers if hasattr(opt, 'record')
                                 for key in ('_thermal_scale', '_in_penalty', '_last_d_logits'))]
    module_states = [(module, deepcopy(module.state_dict())) for module in modules]
    optimizer_states = [(opt, deepcopy(opt.state_dict())) for opt in optimizers]
    other_states = [(obj.load_state_dict, deepcopy(obj.state_dict())) if hasattr(obj, 'state_dict')
                    else (obj.set_state, obj.get_state().clone()) for obj in stateful]
    saved_attributes = [(obj, key, _copy_attribute(getattr(obj, key)) if hasattr(obj, key) else _MISSING)
                        for obj, key in attributes]
    previous_failed = [(owner, getattr(owner, _FAILED, False)) for owner in owners]
    native = _unique(child for module in modules for child in module.modules())
    modes = [(module, module.training) for module in native]
    checkpoint = [(module, key, getattr(module, key, _MISSING)) for module in native
                  for key in ('gradient_checkpointing', '_gradient_checkpointing_func')]
    parameters = _unique(parameter for module in modules for parameter in module.parameters())
    flags = [(parameter, parameter.requires_grad, parameter.grad,
              parameter.grad.detach().clone() if parameter.grad is not None else None)
             for parameter in parameters]
    buffers = [(module, dict(module._buffers), set(module._non_persistent_buffers_set)) for module in native]
    buffer_values = [(buffer, buffer.detach().clone()) for buffer in _unique(
        buffer for _, registered, _ in buffers for buffer in registered.values())]
    # Official critic optimizer loads publish an ephemeral device-local channel.
    channel = getattr(sys.modules.get('particlegan.bispf'), '_LAST_THERMAL_SCALE', None)
    devices = {parameter.device for parameter in parameters}
    thermal = {device: channel.get(device, _MISSING) for device in devices} if channel is not None else {}

    def attribute(obj, key, value):
        if value is _MISSING:
            if hasattr(obj, key):
                delattr(obj, key)
        else:
            setattr(obj, key, value)

    def gradient(parameter, flag, original, value):
        parameter.requires_grad_(flag)
        if original is not None:
            original.copy_(value)
        parameter.grad = original

    def buffer_registration(module, registered, nonpersistent):
        module._buffers.clear()
        module._buffers.update(registered)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(nonpersistent)

    def buffer_value(buffer, value):
        if (buffer.shape, buffer.dtype, buffer.device) == (value.shape, value.dtype, value.device):
            buffer.copy_(value)
        else:
            buffer.data = value.clone()

    def restore_thermal():
        for device, value in thermal.items():
            if value is _MISSING:
                channel.pop(device, None)
            else:
                channel[device] = value

    try:
        yield
    except BaseException as primary:
        actions = [
            *(lambda obj=obj, state=state: obj.load_state_dict(state) for obj, state in module_states),
            *(lambda obj=obj, state=state: obj.load_state_dict(state) for obj, state in optimizer_states),
            *(lambda load=load, state=state: load(state) for load, state in other_states),
            *(lambda obj=obj, key=key, value=value: attribute(obj, key, value)
              for obj, key, value in saved_attributes),
            *(lambda obj=obj, key=key, value=value: attribute(obj, key, value)
              for obj, key, value in checkpoint),
            *(lambda module=module, mode=mode: setattr(module, 'training', mode) for module, mode in modes),
            *(lambda parameter=p, flag=flag, original=original, value=value:
              gradient(parameter, flag, original, value) for p, flag, original, value in flags),
            *(lambda module=module, registered=registered, nonpersistent=nonpersistent:
              buffer_registration(module, registered, nonpersistent)
              for module, registered, nonpersistent in buffers),
            *(lambda buffer=buffer, value=value: buffer_value(buffer, value) for buffer, value in buffer_values),
            restore_thermal,
            lambda: random.setstate(python_rng), lambda: torch.set_rng_state(torch_rng),
            *([lambda: torch.cuda.set_rng_state_all(cuda_rng)] if cuda_rng is not None else []),
        ]
        failed = False
        with torch.no_grad():
            for action in actions:
                try:
                    action()
                except BaseException as rollback:
                    failed = True
                    primary.add_note('training restore rollback failed: ' + type(rollback).__name__ + ': ' + str(rollback))
        if failed:
            _quarantine(owners, optimizers)
        else:
            for owner, previous in previous_failed:
                setattr(owner, _FAILED, previous)
        raise
    else:
        for owner in owners:
            setattr(owner, _FAILED, False)
