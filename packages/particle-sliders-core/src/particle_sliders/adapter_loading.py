"""Validate an entire adapter checkpoint before copying any parameter."""
from collections.abc import Mapping

import torch


def load_adapter_states(entries):
    """Load one or more named adapter subsets atomically, without base copies.

    Each entry is ``(module, checkpoint_state, expected_adapter_keys)``. Base
    weights may be omitted, but the adapter subset must be complete and exact.
    """
    plans = []
    for module, state, keys in entries:
        current = module.state_dict()
        keys = set(keys)
        if not keys or not isinstance(state, Mapping) or set(state) != keys:
            raise ValueError('adapter checkpoint keys differ from the attached adapter')
        for key in keys:
            value, expected = state[key], current[key]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()
                    or not torch.isfinite(value.to(dtype=expected.dtype)).all()):
                raise ValueError('invalid adapter checkpoint tensor: ' + key)
        plans.append((module, state, {key: current[key].detach().clone() for key in keys}))
    if not plans:
        raise ValueError('no attached adapter state to load')
    try:
        for module, state, _ in plans:
            module.load_state_dict(state, strict=False)
    except BaseException:
        # Copy directly into state views: rollback must not re-enter the loader
        # hook which may have failed after it copied a preceding parameter.
        with torch.no_grad():
            for module, _, previous in reversed(plans):
                current = module.state_dict()
                for key, value in previous.items():
                    current[key].copy_(value)
        raise


def optional_peft():
    """Only a genuinely absent optional PEFT package permits custom fallback."""
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as error:
        if error.name != 'peft':
            raise
        return None
    return LoraConfig, get_peft_model
