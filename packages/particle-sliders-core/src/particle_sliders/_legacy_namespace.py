"""Resolve legacy research imports to a single canonical module identity."""

from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
import sys


class _AliasLoader(Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        module = import_module(self.canonical)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module):
        # importlib assigns the alias spec during loading; keep introspection,
        # relative imports, reload and pickle anchored to the canonical name.
        module.__spec__ = self.original_spec

    def get_code(self, fullname):
        # Preserve python -m conceptmod.textsliders.<command> while users migrate.
        spec = util.find_spec(self.canonical)
        return spec.loader.get_code(self.canonical)


class _LegacyFinder(MetaPathFinder):
    _particle_sliders_legacy = True

    def find_spec(self, fullname, path=None, target=None):
        if not any(fullname == root or fullname.startswith(root + ".")
                   for root in ("conceptmod.textsliders", "conceptmod.notrigger")):
            return None
        canonical = "particle_sliders" + fullname[len("conceptmod"):]
        spec = util.find_spec(canonical)
        if spec is None:
            return None
        alias = util.spec_from_loader(
            fullname, _AliasLoader(canonical),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None,
        )
        alias.has_location = spec.has_location
        alias.cached = spec.cached
        return alias


def install():
    if not any(getattr(finder, "_particle_sliders_legacy", False)
               for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
