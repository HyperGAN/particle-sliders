"""Shared restoration primitives; numerical engines own their recovery policy."""

import torch

_RESTORATION_FAILED = '_particle_sliders_restoration_failed'


def _restore_all(actions, *, primary=None, on_failure=None):
    """Attempt every action; preserve only the explicitly supplied operation error.

    Ambient exceptions may belong to an unrelated caller's ``except`` block.
    A normal cleanup failure must still be raised in that case.
    """
    failures = []
    for action in actions:
        try:
            action()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        # A product context can retain the body exception as the primary error.
        # Propagate restoration failure explicitly, rather than parsing notes.
        setattr(primary if primary is not None else failures[0], _RESTORATION_FAILED, True)
        if on_failure is not None:
            on_failure()
        if primary is not None:
            for failure in failures:
                primary.add_note('native restoration failed: ' + type(failure).__name__ + ': ' + str(failure))
        else:
            first, *others = failures
            for failure in others:
                first.add_note('additional native restoration failure: ' + type(failure).__name__ + ': ' + str(failure))
            raise first


def native_cleanup(root):
    """Capture borrowed module modes/gradient flags/checkpoint settings before binding.

    Factories call this before changing a supplied model, and compose the returned
    callback with removal of their owned hooks. It is safe to call repeatedly.
    """
    checkpoint = checkpoint_cleanup(root)
    modes = [(module, module.training) for module in root.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in root.parameters()]
    def restore():
        _restore_all([checkpoint,
            *(lambda module=module, mode=mode: setattr(module, 'training', mode) for module, mode in modes),
            *(lambda parameter=parameter, grad=grad: parameter.requires_grad_(grad) for parameter, grad in gradients)])
    return restore


def checkpoint_cleanup(root):
    """Capture caller-owned checkpoint hooks before a native adapter binds them."""
    keys = ('gradient_checkpointing', '_gradient_checkpointing_func')
    saved = [(module, {key: getattr(module, key) for key in keys if hasattr(module, key)})
             for module in root.modules()]
    def restore():
        actions = []
        for module, previous in saved:
            for key in keys:
                if key in previous:
                    actions.append(lambda module=module, key=key, value=previous[key]: setattr(module, key, value))
                elif hasattr(module, key):
                    actions.append(lambda module=module, key=key: delattr(module, key))
        _restore_all(actions)
    return restore



def _rng_devices(*values):
    """Discover explicit CUDA devices, including nested native arguments/staging."""
    devices, seen = set(), set()
    def visit(value, device_key=False):
        if not isinstance(value, (str, int, float, bool, type(None), torch.device)):
            if id(value) in seen:
                return
            seen.add(id(value))
        if isinstance(value, torch.device) or device_key and isinstance(value, str):
            device = torch.device(value)
            if device.type == 'cuda':
                devices.add(device.index if device.index is not None else torch.cuda.current_device())
        elif isinstance(value, torch.Tensor):
            if value.is_cuda:
                devices.add(value.device.index)
        elif isinstance(value, torch.nn.Module):
            for tensor in (*value.parameters(), *value.buffers()):
                visit(tensor)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, key in ('device', 'torch_device', 'execution_device', 'text_device'))
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif hasattr(value, 'device'):
            visit(value.device, True)
    for value in values:
        visit(value)
    return sorted(devices)


