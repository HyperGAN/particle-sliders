"""Dependency-light training dispatch; product imports occur only in workers."""
from .request import PROFILE_SCHEMA, REQUEST_SCHEMA, resolve_request, load_profile


def execute_resolved_request(request, *, on_checkpoint=None, on_progress=None):
    from .local import execute_resolved_request as execute
    return execute(request, on_checkpoint=on_checkpoint, on_progress=on_progress)


__all__ = ['PROFILE_SCHEMA', 'REQUEST_SCHEMA', 'resolve_request', 'load_profile', 'execute_resolved_request']
