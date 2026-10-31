"""Cooperative source/cache/origin checks, not a Python security boundary.

Checks installed source files and executable bytecode caches before this
binding imports the core. Previously executed or subsequently monkeypatched
in-memory code cannot be attested by on-disk hashes; do not claim otherwise.
"""
import hashlib
from importlib.metadata import distribution
from importlib.util import MAGIC_NUMBER, cache_from_source, find_spec, source_hash
import json
import marshal
from pathlib import Path
import struct
import sys

CORE_PYTHON_SHA256 = '7c3b92f06012188b8d9bb6c389dec33fd3ea864c620d92272721a434114828cb'
CORE_ASSETS_SHA256 = '54265b65989481bb8ea38e35828c0539b7ea101f484a6b558b50b89e27aa31df'


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(files):
    return _sha(json.dumps(files, sort_keys=True).encode())


def _check_cache(source):
    """Reject a source-divergent pyc that this interpreter could execute."""
    cached = Path(cache_from_source(str(source)))
    if not cached.is_file():
        return
    value = cached.read_bytes()
    if len(value) < 16 or value[:4] != MAGIC_NUMBER:
        return  # The source loader ignores this cache too.
    flags, = struct.unpack('<I', value[4:8])
    if flags & ~3:
        return
    data = source.read_bytes()
    if flags & 1:
        if flags & 2 and value[8:16] != source_hash(data):
            return
    else:
        stamp, length = struct.unpack('<II', value[8:16])
        if (stamp, length) != (int(source.stat().st_mtime) & 0xffffffff, len(data) & 0xffffffff):
            return
    try:
        cached_code = marshal.loads(value[16:])
        current_code = compile(data, str(source), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
    except (ValueError, TypeError, EOFError, SyntaxError) as exc:
        raise ValueError('Bonsai executable bytecode cache is invalid: ' + str(source)) from exc
    if cached_code != current_code:
        raise ValueError('Bonsai executable bytecode cache differs from source: ' + str(source))


def _root(project, package, expected_version):
    info = distribution(project)
    if info.version != expected_version:
        raise ValueError('Bonsai requires exact ' + project + '==' + expected_version)
    root = Path(info.locate_file(package)).resolve()
    expected = root / '__init__.py'
    spec = find_spec(package)
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise ValueError('Bonsai import origin differs from installed distribution: ' + package)
    for name, module in tuple(sys.modules.items()):
        if name == package or name.startswith(package + '.'):
            origin = getattr(module, '__file__', None)
            module_spec = getattr(module, '__spec__', None)
            spec_origin = getattr(module_spec, 'origin', None)
            if (not origin or not spec_origin or Path(origin).resolve() != Path(spec_origin).resolve()
                    or not Path(origin).resolve().is_relative_to(root)):
                raise ValueError('Bonsai preloaded import origin differs: ' + name)
    return root


def require_source_environment():
    """Check cooperative installed origins, disk sources and usable pycs.

    This neither proves arbitrary loaded code nor prevents later in-memory
    mutation. It rejects ordinary path shadows and stale executable caches.
    """
    core = _root('particle-sliders-core', 'particle_sliders', '0.12.0')
    sources = {str(p.relative_to(core)): _sha(p.read_bytes()) for p in sorted(core.rglob('*.py'))}
    if _digest(sources) != CORE_PYTHON_SHA256:
        raise ValueError('Bonsai core sources differ; use the original pinned environment')
    assets = {str(p.relative_to(core)): _sha(p.read_bytes()) for p in sorted(core.rglob('*'))
              if p.is_file() and p.suffix in ('.json', '.hndl')}
    if _digest(assets) != CORE_ASSETS_SHA256:
        raise ValueError('Bonsai core graph/source manifests differ')
    hndl = _root('hndl', 'hndl', '0.10.0')
    hg = _root('hypergan', 'hypergan', '2.0.0')
    pg = _root('particlegan', 'particlegan', '0.9.4')
    manifest = json.loads((core / 'fibo_hndl_010_sources.json').read_text())
    if {str(p.relative_to(hndl)): _sha(p.read_bytes()) for p in sorted(hndl.rglob('*.py'))} != manifest['hndl']:
        raise ValueError('Bonsai HNDL on-disk source closure differs')
    if any(_sha((hg / name).read_bytes()) != digest for name, digest in manifest['hypergan'].items()):
        raise ValueError('Bonsai HyperGAN on-disk builder closure differs')
    release = json.loads((core / 'pg094_release.json').read_text())
    if {str(p.relative_to(pg)): _sha(p.read_bytes()) for p in sorted(pg.rglob('*.py'))} != release['python_sources']:
        raise ValueError('Bonsai PG on-disk source closure differs')
    for root in (core, hndl, hg, pg):
        for source in root.rglob('*.py'):
            _check_cache(source)
    return dict(scope='installed on-disk sources, executable caches and cooperative import origins; '
                'not attestation of arbitrary in-memory code or a security sandbox')
