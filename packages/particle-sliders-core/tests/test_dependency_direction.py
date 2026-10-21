"""Enforce product -> slider core -> upstream GAN dependencies without native imports."""
import ast
from pathlib import Path

import pytest
import particle_sliders

PRODUCTS = {'iris_sliders', 'qwen_image_sliders', 'krea2', 'fibo_particle_slider',
            'moss_sliders', 'anima', 'anima_sliders', 'lumen_studio', 'supra',
            'yue2_sliders', 'sana_sliders'}
# One exact pre-existing campaign bridge is inventoried, not a migrated session.
# It stays outside the new image/lifecycle dependency closure and active old pins
# are untouched. Remove this exception when its product injects a slider factory.
LEGACY_REVERSE_IMPORTS = {('pg09_moss_campaign.py', 'moss_sliders.lora', ('Slider',))}


def reverse_imports(source, filename):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module, names = node.module or '', tuple(alias.name for alias in node.names)
            modules = [(module, names)]
        elif isinstance(node, ast.Import):
            modules = [(alias.name, ()) for alias in node.names]
        else:
            continue
        for module, names in modules:
            name = module.split('.')[0]
            if name in PRODUCTS or name.endswith(('_sliders', '_particle_slider')) and name != 'particle_sliders':
                found.add((filename, module, names))
    return found


def test_core_has_no_uninventoried_product_imports():
    root = Path(particle_sliders.__file__).parent
    actual = set()
    for source in root.glob('*.py'):
        actual |= reverse_imports(source.read_text(), source.name)
    assert actual == (LEGACY_REVERSE_IMPORTS if (root / 'pg09_moss_campaign.py').exists() else set())
    assert all(file not in ('lifecycle.py', 'pg09_images.py', 'qwen_native.py', 'fibo_native.py')
               for file, _, _ in actual)


@pytest.mark.parametrize('source', [
    'from iris_sliders.train import build_game', 'import krea2',
    'from qwen_image_sliders.projection import predict', 'import future_model_sliders',
])
def test_guard_detects_new_reverse_edges(source):
    assert reverse_imports(source, 'new_backend.py')


def test_core_distribution_does_not_depend_on_products():
    from importlib.metadata import requires
    dependencies = requires('particle-sliders-core') or []
    forbidden = ('iris-particle', 'qwen-image-particle', 'krea2-particle',
                 'fibo-particle', 'moss-sliders', 'anima-particle', 'yue2-particle')
    assert not any(any(name in dependency.lower() for name in forbidden) for dependency in dependencies)
    assert any(dependency.lower().startswith('particlegan==0.9.1') for dependency in dependencies)
