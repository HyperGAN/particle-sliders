"""Resolve legacy research imports to a single canonical module identity."""

from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
import sys


class _AliasLoader(Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        # Let importlib initialize a temporary alias module. Returning the
        # canonical object here would briefly overwrite its __spec__ while a
        # concurrent canonical import can observe it.
        return None

    def exec_module(self, module):
        sys.modules[module.__name__] = import_module(self.canonical)

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


class _CompatibilityRootLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__doc__ = "Legacy research namespace; use particle_sliders for new code."


class _MissingRootFinder(MetaPathFinder):
    _particle_sliders_missing_root = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "conceptmod":
            return util.spec_from_loader(fullname, _CompatibilityRootLoader(), is_package=True)


def install():
    if not any(getattr(finder, "_particle_sliders_legacy", False)
               for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
    if not any(getattr(finder, "_particle_sliders_missing_root", False)
               for finder in sys.meta_path):
        # Existing regular/namespace/editable conceptmod providers get first
        # refusal. Only synthesize a root when no other finder supplies one.
        sys.meta_path.append(_MissingRootFinder())
