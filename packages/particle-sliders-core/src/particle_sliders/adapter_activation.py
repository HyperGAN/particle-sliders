"""Restore effective adapter settings around nested, single-owner scopes."""
import asyncio
from contextlib import contextmanager
import threading


class _ActivationGuard:
    """An idle adapter remains deepcopy/pickle compatible."""
    def __init__(self):
        self.lock = threading.Lock()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.current_thread(), task


@contextmanager
def activation_mutation(owner):
    """Reject competing thread/task owners without blocking an event loop."""
    guard = vars(owner).get('_activation_guard')
    if guard is None:
        guard = vars(owner).setdefault('_activation_guard', _ActivationGuard())
    if not guard.lock.acquire(blocking=False):
        raise RuntimeError('adapter activation is owned by another operation')
    try:
        if activation_is_entered(owner) and owner._activation_owner != _owner():
            raise RuntimeError('adapter activation is owned by another thread or async task')
        yield
    finally:
        guard.lock.release()


def push_activation(owner, modules, attributes):
    """Save exact values/objects before a scope changes any adapter settings."""
    with activation_mutation(owner):
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
        owner._activation_owner = _owner()
        stack.append(saved)


def restore_activation(owner):
    """Restore the latest scope, including dictionary identity and contents."""
    with activation_mutation(owner):
        stack = getattr(owner, '_activation_stack', None)
        if not stack:
            raise RuntimeError('adapter activation scope is not entered')
        for module, name, value, contents in reversed(stack[-1]):
            if contents is not None:
                value.clear()
                value.update(contents)
            setattr(module, name, value)
        stack.pop()
        if not stack:
            del owner._activation_owner


def activation_is_entered(owner):
    return bool(getattr(owner, '_activation_stack', None))
