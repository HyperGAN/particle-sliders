"""Preserved name for the separately pinned core0.11/PG0.9.3 cohort.

Core0.12 uses the exact PG0.9.4 engine in pg094. Refuse old explicit entrypoints
instead of silently mapping a PG0.9.3 request onto a different source cohort.
Use core0.11.0 at its original commit for existing PG0.9.3 checkpoints.
"""
PG_VERSION = '0.9.3'
PG_SOURCE_PROFILE = 'release-0.9.3'
PG_WHEEL_SHA256 = 'fd7ee424d4c67e1dc3cf0b9dd345222049a67e8c293c30c8a3c42d54805affec'
_MESSAGE = 'PG0.9.3 requires its pinned core0.11.0 cohort; core0.12 uses pg094 and cannot rebind old checkpoints'


class ConditionalPG093:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(_MESSAGE)


def engine_options(options):
    raise ValueError(_MESSAGE)


def validate_pg09_domain(*args, **kwargs):
    raise ValueError(_MESSAGE)


__all__ = ['ConditionalPG093', 'PG_SOURCE_PROFILE', 'PG_VERSION', 'PG_WHEEL_SHA256',
           'validate_pg09_domain', 'engine_options']
