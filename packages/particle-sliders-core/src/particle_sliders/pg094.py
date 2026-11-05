"""Historical PG0.9.4 entrypoint; use its original core0.12 source for old runs."""
PG_VERSION = '0.9.4'
PG_SOURCE_PROFILE = 'release-0.9.4'
PG_WHEEL_SHA256 = 'ad21eeaea861cf0582fc39aa96eb3fd8ab8cf21e8e0f0179f0597a9c42eb99b6'
_MESSAGE = 'PG0.9.4 requires its pinned core0.12 cohort; core0.13 uses pg095 and cannot rebind old checkpoints'

class ConditionalPG094:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(_MESSAGE)

def engine_options(options):
    raise ValueError(_MESSAGE)

def validate_pg09_domain(*args, **kwargs):
    raise ValueError(_MESSAGE)

__all__ = ['ConditionalPG094', 'PG_SOURCE_PROFILE', 'PG_VERSION', 'PG_WHEEL_SHA256',
           'validate_pg09_domain', 'engine_options']
