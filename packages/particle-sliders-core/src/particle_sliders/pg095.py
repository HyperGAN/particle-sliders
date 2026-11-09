"""Explicit ParticleGAN 0.9.5 public engine/profile import for the core 0.13 line.

Core 0.8.1 remains the separately pinned 0.9.1 cohort line. No old packet or
unreviewed overlay is admitted here. No GAN implementation is duplicated.
"""
from .pg09 import (
    ConditionalPG09 as ConditionalPG095, PG_SOURCE_PROFILE, PG_VERSION,
    PG_WHEEL_SHA256, PG_SOURCE_COMMIT,
    validate_pg09_domain, CONVND_SOURCE_PROFILE, _profile_version, _DOMAIN_PAIRS,
)

__all__ = ['ConditionalPG095', 'PG_SOURCE_PROFILE', 'PG_VERSION', 'PG_WHEEL_SHA256',
           'validate_pg09_domain', 'PG_SOURCE_COMMIT']


def engine_options(options):
    """Validate explicit product configuration and return the PG constructor keys."""
    profile = options.get('pg_source_profile', PG_SOURCE_PROFILE)
    expected_version = _profile_version(profile)
    if options.get('pg_version', expected_version) != expected_version:
        raise ValueError('exact ParticleGAN version for this profile is ' + expected_version)
    domain = options.get('penalty_input_domain', 'auto')
    if (profile, domain) not in _DOMAIN_PAIRS:
        raise ValueError('this engine requires an explicitly qualified source/domain pair')
    return dict(pg_source_profile=profile, penalty_input_domain=domain)


__all__.append('engine_options')

__all__.append("CONVND_SOURCE_PROFILE")
