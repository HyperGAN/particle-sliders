"""Opt-in named PEFT scaling; no eager PEFT import or historical scaler changes."""
from contextlib import contextmanager
import hashlib
import math
from numbers import Real
from pathlib import Path

from torch import nn


def _finite_scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label + ' must be a finite real scalar')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(label + ' must be a finite real scalar') from error
    if not math.isfinite(result):
        raise ValueError(label + ' must be a finite real scalar')
    return result


def _module_schema(module):
    """Bind ownership/topology, not evolving trained values or mode/grad flags."""
    return (id(module), type(module), module.extra_repr(),
            tuple((name, id(value), tuple(value.shape), value.dtype, value.device)
                  for name, value in module.named_parameters()),
            tuple((name, id(child), type(child), child.extra_repr())
                  for name, child in module.named_modules()))


def _same_scaling(actual, expected):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return False
    try:
        return all(_finite_scalar(value, 'scaling') == expected[name] for name, value in actual.items())
    except ValueError:
        return False


class _PEFTScaleContext:
    def __init__(self, roots, adapter_names, layer_type, linear_type):
        self._roots, self._names = roots, adapter_names
        self._layer_type, self._linear_type = layer_type, linear_type
        self._inventory = self._discover()
        self._plan = []
        if not self._inventory:
            raise ValueError('selected PEFT adapters have no supported leaves')
        for path, layer in self._inventory:
            schema = self._inspect(path, layer)
            factors = {name: _finite_scalar(layer.scaling[name], path + '.' + name + ' scaling')
                       for name in self._names}
            fixed = {name: value for name, value in layer.scaling.items() if name not in self._names}
            self._plan.append((path, layer, schema, factors, fixed))

    def _discover(self):
        result, seen = [], set()
        for prefix, root in self._roots:
            for path, module in root.named_modules():
                if isinstance(module, self._layer_type) and id(module) not in seen:
                    seen.add(id(module))
                    result.append((prefix + ('.' + path if path else ''), module))
        if len({path for path, _ in result}) != len(result):
            raise ValueError('PEFT native paths must uniquely identify distinct layers')
        return tuple(result)

    def _inspect(self, path, layer):
        if not isinstance(layer.scaling, dict) or not layer.scaling:
            raise ValueError(path + ': PEFT scaling must be a nonempty dictionary')
        if any(not isinstance(name, str) or not name for name in layer.scaling):
            raise ValueError(path + ': PEFT scaling keys must be nonempty adapter names')
        keys = set(getattr(layer, 'scaling', {})) | set(getattr(layer, 'r', {}))
        missing = set(self._names) - keys
        if missing:
            raise ValueError(path + ': selected PEFT adapters missing from layer: ' + ', '.join(sorted(missing)))
        if type(layer) is not self._linear_type or type(layer.get_base_layer()) is not nn.Linear:
            raise ValueError(path + ': selected adapters require stock PEFT Linear on torch.nn.Linear')
        if layer.merged:
            raise ValueError(path + ': merged PEFT layers are unsupported; unmerge before binding')
        if layer.disable_adapters:
            raise ValueError(path + ': disabled PEFT adapters are unsupported; enable before binding')
        active = tuple(layer.active_adapters)
        if (any(getattr(layer, 'use_dora', {}).values())
                or getattr(layer, 'lora_variant', {})):
            raise ValueError(path + ': PEFT LoRA variants are unsupported in the selected scope')
        for name in self._names:
            if name not in layer.lora_A or name not in layer.lora_B or name not in layer.scaling:
                raise ValueError(path + ': incomplete selected PEFT A/B/scaling entries: ' + name)
            if name not in active:
                raise ValueError(path + ': selected PEFT adapter is inactive: ' + name)
        for name, value in layer.scaling.items():
            _finite_scalar(value, path + '.' + name + ' scaling')
        schema = (_module_schema(layer), tuple(sorted(layer.scaling)), active,
                  tuple(sorted(layer.r.items())), tuple(sorted(layer.lora_alpha.items())))
        return schema

    def _preflight(self):
        inventory = self._discover()
        if len(inventory) != len(self._inventory) or any(
                path != old_path or layer is not old_layer
                for (path, layer), (old_path, old_layer) in zip(inventory, self._inventory)):
            raise ValueError('PEFT layer topology changed; create a new scale scope')
        for path, layer, schema, _, fixed in self._plan:
            if self._inspect(path, layer) != schema:
                raise ValueError(path + ': PEFT adapter topology or activation changed; create a new scale scope')
            if any(layer.scaling[name] != value for name, value in fixed.items()):
                raise ValueError(path + ': unselected PEFT scaling changed; create a new scale scope')

    def metadata(self):
        """Include this receipt in product metadata to bind source and selection."""
        self._preflight()
        return dict(protocol='particle-sliders-peft-adapter-scale-v1',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            adapter_names=list(self._names),
            targets=[dict(path=path, selected_baseline_scaling=dict(factors),
                          fixed_scaling=dict(fixed), active_adapters=list(schema[2]))
                     for path, _, schema, factors, fixed in self._plan])

    @contextmanager
    def __call__(self, scale):
        scale = _finite_scalar(scale, 'adapter scale')
        self._preflight()  # Validate the entire scope before touching any layer.
        updates = []
        for path, layer, _, factors, _ in self._plan:
            previous = layer.scaling
            updated = dict(previous)
            for name, factor in factors.items():
                updated[name] = _finite_scalar(factor * scale, path + '.' + name + ' scaled value')
            updates.append((layer, previous, updated, dict(previous), dict(updated)))
        installed = []
        try:
            for entry in updates:
                installed.append(entry)  # Retain rollback even if assignment raises.
                entry[0].scaling = entry[2]
            yield
        finally:
            error = None
            try:
                self._preflight()
            except BaseException as failure:
                error = failure
            for layer, previous, updated, saved_previous, expected in reversed(installed):
                try:
                    if not _same_scaling(previous, saved_previous) or not _same_scaling(updated, expected):
                        error = error or RuntimeError('PEFT scaling values changed inside scope; use one scale owner')
                except BaseException as failure:
                    error = error or failure
                try:
                    if layer.scaling is updated:
                        layer.scaling = previous
                    elif layer.scaling is not previous:
                        error = error or RuntimeError('PEFT scaling ownership changed inside scope; external replacements were not overwritten')
                except BaseException as failure:
                    error = error or failure
            if error is not None:
                raise error


def peft_adapter_scale_context(roots, *, adapter_names):
    """Bind an explicit named scale scope on already attached PEFT Linear LoRA.

    ``roots`` contains ``(native_path, nn.Module)`` pairs, deduplicated by layer
    identity. Every requested name must occur on EVERY PEFT leaf under those
    roots; pass narrower roots for intentional partial placement. Unselected
    adapter entries are untouched. Each entry scales from its
    binding-time factor, preserving alpha/rank or RS-LoRA factors and avoiding
    multiplication by an outer scope. Exit restores the exact prior dictionary.

    The returned callable also exposes ``metadata()`` for product receipts. Bind
    once after attachment/device placement, before scale overrides. It changes
    no parameters, activation, modes or gradient flags. Mode/grad changes are
    allowed; adapter topology/activation changes require a fresh scope.

    This controls scale, not checkpoint scheduling: pass it to AdapterParameters
    and use the existing captured-scale checkpoint owner. Uncaptured checkpoint
    replay after scope exit is unsupported. Concurrent use or external scaling
    dictionary replacement inside a scope is unsupported. PEFT is an optional
    native dependency, imported only when this function is called.
    """
    if isinstance(adapter_names, (str, bytes)):
        raise ValueError('adapter_names must be a nonempty collection of explicit names')
    try:
        names = tuple(adapter_names)
    except TypeError as error:
        raise ValueError('adapter_names must be a nonempty collection of explicit names') from error
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('adapter_names must be nonempty, unique, nonempty strings')
    try:
        roots = tuple(roots)
    except TypeError as error:
        raise ValueError('roots must contain nonempty native-path/module pairs') from error
    if not roots or any(not isinstance(entry, (tuple, list)) or len(entry) != 2
                        or not isinstance(entry[0], str) or not entry[0]
                        or not isinstance(entry[1], nn.Module) for entry in roots):
        raise ValueError('roots must contain nonempty native-path/module pairs')
    roots = tuple((path, module) for path, module in roots)
    from peft.tuners.lora.layer import Linear, LoraLayer
    return _PEFTScaleContext(roots, tuple(sorted(names)), LoraLayer, Linear)
