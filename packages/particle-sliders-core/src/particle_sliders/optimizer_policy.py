"""Versioned optimizer selection/observability; matrix mathematics stays upstream.

Only the exact modern PG release admitted by pg09 may construct this recipe.
Auto selects Gram only where upstream supports it, never a different GAN law.
"""
from copy import deepcopy
from dataclasses import replace

POLICY = 'particle-sliders-dense-policy-v1'
GRAM_PROFILE = 'gram_fp64_svd_diagnostic_v2'
GUARDED_PROFILE = 'gram_fp64_guarded_v1'
PROFILES = ('auto', 'default', GRAM_PROFILE, GUARDED_PROFILE)


def configure_dense_profile(recipe, requested='auto'):
    """Resolve allocation-free, preserving every other upstream Recipe field."""
    if requested not in PROFILES:
        raise ValueError('unknown dense optimizer profile; use auto, default, ' + GRAM_PROFILE)
    eligible = (recipe.optimizer_family == 'dualnorm' and recipe.optimizer_momentum == 0
                and recipe.optimizer_smoothing == .001
                and recipe.optimizer_svd_backend in ('native', 'batched'))
    if requested == 'auto':
        selected = GRAM_PROFILE if eligible else 'default'
    else:
        selected = requested
    if selected != 'default' and not eligible:
        raise ValueError('Gram requires upstream full DualNorm, zero momentum, smoothing .001, native/batched SVD')
    resolved = replace(recipe, optimizer_dense_profile=selected)
    declaration = dict(schema=POLICY, requested=requested, resolved=selected,
        eligible=eligible, reason=('supported upstream dense profile' if eligible else
          'optimizer law does not support Gram; original upstream law retained'),
        scope='FP32 dense 2D DualNorm leaves; vectors, row-normalized banks and convolutions retain upstream paths',
        differentiated_solver=False)
    return resolved, declaration


def optimizer_diagnostics(optimizer):
    """Detached telemetry; upstream resets counters on successful state restore."""
    stats = getattr(optimizer, 'dense_profile_stats', None)
    result = deepcopy(stats) if stats is not None else {
        'profile': 'default', 'available': False,
        'reason': 'upstream optimizer has no dense-profile telemetry'}
    result['eligible_dense_leaves'] = sum(
        parameter.ndim == 2 and str(parameter.dtype) == 'torch.float32'
        and parameter.device.type in ('cpu', 'cuda')
        for group in optimizer.param_groups
        if group.get('algorithm') == 'dualnorm' and 'dualnorm_convolution' not in group
        for parameter in group['params'])
    result['upstream_groups'] = [dict(role=group.get('role'), algorithm=group.get('algorithm'),
        convolution=group.get('dualnorm_convolution'), leaves=len(group['params']))
        for group in optimizer.param_groups]
    return result
