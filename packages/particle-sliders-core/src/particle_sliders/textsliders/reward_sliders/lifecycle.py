"""Share the pilot's process lock across subsequent GPU stages."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def exclusive_stage(run):
    with (Path(run)/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
