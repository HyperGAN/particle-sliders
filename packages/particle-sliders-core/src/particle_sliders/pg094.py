"""Explicit ParticleGAN 0.9.4 public engine/profile import for the core 0.12 line.

Core 0.8.1 remains the separately pinned 0.9.1 cohort line. No old packet or
unreviewed overlay is admitted here. No GAN implementation is duplicated.
"""
from .pg09 import (
    ConditionalPG09 as ConditionalPG094, PG_SOURCE_PROFILE, PG_VERSION,
    PG_WHEEL_SHA256, PG_CONTINUOUS_SOURCE_PROFILE, PG_CONTINUOUS_COMMIT,
    validate_pg09_domain,
)

__all__ = ['ConditionalPG094', 'PG_SOURCE_PROFILE', 'PG_VERSION', 'PG_WHEEL_SHA256',
           'validate_pg09_domain', 'PG_CONTINUOUS_SOURCE_PROFILE', 'PG_CONTINUOUS_COMMIT']


def engine_options(options):
    """Validate explicit product configuration and return the PG constructor keys."""
    if options.get('pg_version', PG_VERSION) != PG_VERSION:
        raise ValueError('this core release requires ParticleGAN ' + PG_VERSION)
    profile = options.get('pg_source_profile', PG_SOURCE_PROFILE)
    domain = options.get('penalty_input_domain', 'auto')
    if (profile, domain) not in ((PG_SOURCE_PROFILE, 'auto'), (PG_CONTINUOUS_SOURCE_PROFILE, 'continuous')):
        raise ValueError('this engine requires an explicitly qualified source/domain pair')
    return dict(pg_source_profile=profile, penalty_input_domain=domain)


__all__.append('engine_options')
