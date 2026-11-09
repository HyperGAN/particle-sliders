"""A wheel rebuild cannot adopt stale product outputs or symlink aliases."""
import ast
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from test_namespace_installation import run


# (packaged file, dotted module) pairs for deliberate guarded (function, try or
# if) references to modules kept out of the core wheel. None exist today; never
# list an import that runs unconditionally at module import time.
OPTIONAL_EXTERNAL = frozenset()


@pytest.fixture(scope='module')
def stale_build_wheel(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('stale-build')
    project = Path(__file__).resolve().parents[1]
    copy = tmp_path / 'core'
    shutil.copytree(project, copy, ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
    stale = copy / 'build/lib'
    (stale / 'particle_sliders/textsliders').mkdir(parents=True)
    (stale / 'particle_sliders/native_lora.py').write_text('# old allowed output\n')
    (stale / 'particle_sliders/music3_native.py').symlink_to('native_lora.py')
    (stale / 'particle_sliders/rogue_product').symlink_to(
        copy / 'src/particle_sliders/textsliders', target_is_directory=True)
    (stale / 'particle_sliders/textsliders/leak.py').write_text('raise RuntimeError("stale product")\n')
    (stale / 'particle_sliders_legacy_imports.pth').write_text('import conceptmod\n')
    wheels = tmp_path / 'wheels'
    run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-index',
         '--no-build-isolation', '--wheel-dir', str(wheels), str(copy)], tmp_path)
    wheel, = wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_rebuild_removes_stale_product_files_and_aliases(stale_build_wheel):
    names = set(stale_build_wheel)
    assert 'particle_sliders/initialization.py' in names
    assert 'particle_sliders/native_lora.py' in names
    assert 'particle_sliders/native_lora_compat.py' in names
    assert 'particle_sliders/candidate_failure.py' in names
    assert 'particle_sliders/music3_native.py' not in names
    assert not any(name.startswith('particle_sliders/textsliders/') for name in names)
    assert not any(name.startswith('particle_sliders/rogue_product/') for name in names)
    assert 'particle_sliders_legacy_imports.pth' not in names


def _imports(tree):
    """Yield (node, guarded): guarded means inside a def, class body, try or if."""
    def visit(node, guarded):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child, guarded
            nested = guarded or isinstance(child, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Try, ast.If))
            if hasattr(ast, 'TryStar') and isinstance(child, ast.TryStar):
                nested = True
            yield from visit(child, nested)
    yield from visit(tree, False)


def _bound_names(source):
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split('.')[0] for alias in node.names)
    return names


def test_packaged_intra_package_imports_resolve_inside_the_wheel(stale_build_wheel):
    files = {name for name in stale_build_wheel if name.endswith('.py') and '.dist-info/' not in name}

    def module_file(parts):
        path = '/'.join(parts)
        for candidate in (path + '.py', path + '/__init__.py'):
            if candidate in files:
                return candidate
        return None

    missing = []
    for name in sorted(files):
        parts = name[:-3].split('/')
        package = parts[:-1] if parts[0] == 'particle_sliders' else None
        tree = ast.parse(stale_build_wheel[name], name)
        for node, guarded in _imports(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [(alias.name.split('.'), None) for alias in node.names
                           if alias.name.split('.')[0] == 'particle_sliders']
            elif node.level:
                if package is None:
                    continue
                base = package[:len(package) - node.level + 1]
                module = base + (node.module.split('.') if node.module else [])
                targets = [(module, [alias.name for alias in node.names])]
            elif node.module and node.module.split('.')[0] == 'particle_sliders':
                targets = [(node.module.split('.'), [alias.name for alias in node.names])]
            for module, imported in targets:
                unresolved = [module[:end] for end in range(1, len(module) + 1) if module_file(module[:end]) is None]
                if not unresolved and imported:
                    found = module_file(module)
                    if found.endswith('/__init__.py'):
                        bound = _bound_names(stale_build_wheel[found])
                        unresolved = [module + [item] for item in imported if item != '*'
                                      and item not in bound and module_file(module + [item]) is None]
                for target in unresolved:
                    key = (name, '.'.join(target))
                    if guarded and key in OPTIONAL_EXTERNAL:
                        continue
                    missing.append((name, node.lineno, '.'.join(target), 'guarded' if guarded else 'module-level'))
    assert not missing, missing
