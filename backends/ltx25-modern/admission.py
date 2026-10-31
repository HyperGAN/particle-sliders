"""Generated package admission anchors; never admits a source checkout or overlay."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import particle_sliders

PRODUCT_SOURCES = {}
CORE_FILES = {}
CORE_COMMIT = '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
PRODUCT_ROOT = Path(__file__).resolve().parent
CORE_ROOT = Path(particle_sliders.__file__).resolve().parent


def _files(root, suffixes):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*')) if path.is_file()
            and '__pycache__' not in path.parts and path.suffix in suffixes}


def admit():
    if (version('particle-sliders-core'), version('particle-sliders-ltx25-modern-preview')) != ('0.12.0', '0.2.0'):
        raise ValueError('exact core/product version differs from dummy preview')
    core = _files(CORE_ROOT, ('.py', '.json', '.hndl'))
    if core != CORE_FILES:
        raise ValueError('exact core runtime source closure differs')
    product = _files(PRODUCT_ROOT, ('.py',))
    admission = product.pop('_admission.py', None)
    if product != PRODUCT_SOURCES or admission is None:
        raise ValueError('exact product runtime source closure differs')
    return dict(core_commit=CORE_COMMIT, core_sources=core,
                product_sources=product, admission_sha256=admission,
                qualification='dummy-preview-only')
