"""Historical PG0.9.5 entrypoint; use its original core0.13 source for old runs."""
PG_VERSION = '0.9.5'
PG_SOURCE_PROFILE = 'release-0.9.5'
PG_WHEEL_SHA256 = 'c1c0de7a635ac1b1634ba158eed050b6f6aed993e01f9b463435430c77d846ae'
_MESSAGE = 'PG0.9.5 requires its pinned core0.13 cohort; core0.14 uses pg0101 and cannot rebind old checkpoints'

class ConditionalPG095:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(_MESSAGE)

def engine_options(options):
    raise ValueError(_MESSAGE)

def validate_pg09_domain(*args, **kwargs):
    raise ValueError(_MESSAGE)

__all__ = ['ConditionalPG095', 'PG_SOURCE_PROFILE', 'PG_VERSION', 'PG_WHEEL_SHA256',
           'validate_pg09_domain', 'engine_options']
