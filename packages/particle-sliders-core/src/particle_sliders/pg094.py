"""Explicit ParticleGAN 0.9.4 public engine/profile import for the core 0.12 line.

Core 0.8.1 remains the separately pinned 0.9.1 cohort line. No old packet or
unreviewed overlay is admitted here. No GAN implementation is duplicated.
"""
from .pg09 import (
    ConditionalPG09 as ConditionalPG094, PG_SOURCE_PROFILE, PG_VERSION,
    PG_WHEEL_SHA256, validate_pg09_domain,
)

__all__ = ['ConditionalPG094', 'PG_SOURCE_PROFILE', 'PG_VERSION', 'PG_WHEEL_SHA256',
           'validate_pg09_domain']


def engine_options(options):
    """Validate explicit product configuration and return the PG constructor keys."""
    if options.get('pg_version', PG_VERSION) != PG_VERSION:
        raise ValueError('this core release requires ParticleGAN ' + PG_VERSION)
    if (options.get('pg_source_profile', PG_SOURCE_PROFILE),
            options.get('penalty_input_domain', 'auto')) != (PG_SOURCE_PROFILE, 'auto'):
        raise ValueError('this engine admits only ' + PG_SOURCE_PROFILE + '/auto')
    return dict(pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain='auto')


__all__.append('engine_options')
