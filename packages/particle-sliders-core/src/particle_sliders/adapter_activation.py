"""Restore effective adapter settings around nested, single-owner scopes."""


def push_activation(owner, modules, attributes):
    """Save exact values/objects before a scope changes any adapter settings."""
    saved, seen = [], set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        for name in attributes:
            if hasattr(module, name):
                value = getattr(module, name)
                saved.append((module, name, value, dict(value) if isinstance(value, dict) else None))
    stack = getattr(owner, '_activation_stack', None)
    if stack is None:
        stack = []
        owner._activation_stack = stack
    stack.append(saved)


def restore_activation(owner):
    """Restore the latest scope, including dictionary identity and contents."""
    stack = getattr(owner, '_activation_stack', None)
    if not stack:
        raise RuntimeError('adapter activation scope is not entered')
    for module, name, value, contents in reversed(stack.pop()):
        if contents is not None:
            value.clear()
            value.update(contents)
        setattr(module, name, value)


def activation_is_entered(owner):
    return bool(getattr(owner, '_activation_stack', None))
