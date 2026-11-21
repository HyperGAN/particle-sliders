"""Identify whether ``particle_sliders`` was imported from the installed core.

A regular wheel serves the package from site-packages. A strict PEP 660 editable
install (the build backend refuses any other editable mode) serves it from a
link tree named by the distribution's ``__editable__`` path file; that tree
contains only the reviewed shared modules, like the wheel. A plain source
checkout on ``sys.path`` matches neither location.
"""
from importlib.metadata import distribution
from pathlib import Path


def _package_roots(installed):
    roots = [Path(installed.locate_file(''))]
    for file in installed.files or ():
        if file.name.startswith('__editable__.') and file.suffix == '.pth':
            for line in Path(installed.locate_file(file)).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith(('#', 'import ')):
                    roots.append(Path(line))
    return [(root / 'particle_sliders').resolve() for root in roots]


def loaded_from_installed_core():
    import particle_sliders
    installed = distribution('particle-sliders-core')
    if installed.version != particle_sliders.__version__:
        return False
    # Resolve the directory, not __init__.py: strict editable trees link files.
    return Path(particle_sliders.__file__).parent.resolve() in _package_roots(installed)
