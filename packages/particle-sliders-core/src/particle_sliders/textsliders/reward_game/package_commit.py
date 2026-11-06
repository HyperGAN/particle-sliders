"""Build a package privately; only a complete directory becomes public."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

from .core import IntegrityError, locked


@contextmanager
def staged_package(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with locked(destination.parent / ('.' + destination.name + '.publish.lock'), blocking=True):
        if destination.exists() or destination.is_symlink():
            raise IntegrityError('Completed package name already exists')
        staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
        try:
            yield staging
            # Flush only our staged files, never follow retained evidence links.
            for base, dirs, files in os.walk(staging, followlinks=False):
                for name in files:
                    path = Path(base) / name
                    if not path.is_symlink():
                        with path.open('rb') as stream:
                            os.fsync(stream.fileno())
                descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            if destination.exists() or destination.is_symlink():
                raise IntegrityError('Completed package name already exists')
            os.rename(staging, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
