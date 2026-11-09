"""Explicit fresh-only, parameter-scoped access to the admitted PG initializer.

Nothing calls this helper automatically. Callers must supply truthful fresh/restore
context before constructing optimizers; a module cannot reveal arbitrary prior
``load_state_dict`` calls. See SCOPED-INITIALIZATION.md for the supported scopes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path

import torch
from torch import nn
from particlegan import init as pg_init

__all__ = ["ComponentInitialization", "InitializationReceipt", "initialize_fresh_components"]

_ADMITTED_INIT_SHA256 = "3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b"
_MARKER = "_particle_sliders_fresh_initialization_sha256"
_METHODS = {"keep", "deterministic_orthogonal", "sample_distributions_v1",
            "xavier_uniform_zero_bias_v1", "identity_linear_v1"}
_LEAVES = {nn.Module, nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
           nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d}


@dataclass(frozen=True)
class ComponentInitialization:
    """One explicitly owned leaf, or a whole preserved ``keep`` scope.

    ``parameter_seeds`` names EVERY nonempty trainable parameter, including KEEP
    and protected zero-up values. ``distributions`` locally overrides official
    declarations for QR/sampling; plain Module holders require complete specs.
    Methods have no default. No module/bank/optimizer is constructed by this API.
    """
    module: nn.Module
    method: str
    parameter_seeds: Mapping[str, int] = field(default_factory=dict)
    distributions: Mapping[str, object] = field(default_factory=dict)
    gain: float = 1.0


@dataclass(frozen=True)
class InitializationReceipt:
    """Canonical immutable JSON; ``as_dict`` returns an independent copy."""
    canonical_json: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def as_dict(self) -> dict:
        return json.loads(self.canonical_json)


class _DeclaredParameters(nn.Module):
    def __init__(self, parameters, declarations):
        super().__init__()
        self._declarations = declarations
        for name, parameter in parameters.items():
            self.register_parameter(name, parameter)


# Only this private holder is registered. Host layer declarations are never patched.
pg_init.register(_DeclaredParameters, lambda module: module._declarations)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tensor_sha(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _source(path):
    data = Path(path).read_bytes()
    return {"file": Path(path).name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _spec(spec):
    if spec is pg_init.KEEP:
        return {"kind": "KEEP"}
    if isinstance(spec, pg_init.Uniform):
        values = (spec.low, spec.high)
        valid = spec.low < spec.high
        result = {"kind": "Uniform", "low": spec.low, "high": spec.high}
    elif isinstance(spec, (pg_init.Normal, pg_init.R2Normal)):
        values = (spec.mean, spec.std)
        valid = spec.std > 0
        result = {"kind": type(spec).__name__, "mean": spec.mean, "std": spec.std}
    else:
        raise ValueError("every trainable parameter requires Uniform, Normal, R2Normal or KEEP")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or not valid:
        raise ValueError("distribution bounds must be finite with positive width/std")
    return result


def _linear_holder(original, parameters):
    # Bypass reset_parameters: no global RNG draw and no original/backbone deepcopy.
    holder = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(holder)
    holder.in_features, holder.out_features = original.in_features, original.out_features
    for name in ("weight", "bias"):
        holder.register_parameter(name, parameters.get(name))
    return holder


def initialize_fresh_components(
    components: Mapping[str, ComponentInitialization], *, cohort_id: str,
    source_identity: Mapping[str, object], completed_steps: int,
    restored_from: str | None, optimizers: Sequence,
) -> InitializationReceipt:
    """Stage all owned trainables, then commit in place with a provenance receipt.

    Refuses restored/nonzero/optimizer contexts and repeat calls on marked scopes.
    These are enforceable supplied-context checks, not a universal history detector.
    Validation/staging failures leave original values, global RNG and markers alone.
    Tensor copies retain Parameter identity; arbitrary device/copy failures are not
    promised to be transactional (versions can advance even during rollback).
    """
    if type(completed_steps) is not int or completed_steps != 0 or restored_from is not None:
        raise ValueError("initialization is fresh-only: completed_steps=0 and restored_from=None required")
    if not isinstance(optimizers, Sequence) or len(optimizers):
        raise ValueError("initialize before constructing optimizers; optimizers must be empty")
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("a nonempty fresh cohort_id is required")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise ValueError("explicit caller source_identity is required")
    source_identity = json.loads(_canonical(dict(source_identity)))
    if not isinstance(components, Mapping) or not components:
        raise ValueError("components must be a nonempty named mapping")
    if any(not isinstance(n, str) or not n or not isinstance(c, ComponentInitialization)
           for n, c in components.items()):
        raise ValueError("components require nonempty names and ComponentInitialization values")
    version = metadata.version("particlegan")
    pg_source = _source(pg_init.__file__)
    if version != "0.9.5" or pg_source["sha256"] != _ADMITTED_INIT_SHA256:
        raise ValueError("this helper admits the recorded ParticleGAN 0.9.5 initializer source only")

    # Validate ownership globally, including KEEP/frozen parameters and buffers.
    seen_modules, seen_parameters, storages, buffers = set(), set(), set(), set()
    for name, policy in components.items():
        module = policy.module
        if not isinstance(module, nn.Module) or policy.method not in _METHODS:
            raise ValueError(f"invalid module/method for {name}")
        if hasattr(module, _MARKER) or id(module) in seen_modules:
            raise ValueError(f"already initialized or overlapping scope: {name}")
        seen_modules.add(id(module))
        if any(hasattr(child, _MARKER) for child in module.modules()):
            raise ValueError(f"scope contains an already initialized component: {name}")
        for parameter_name, parameter in module.named_parameters(remove_duplicate=False):
            if hasattr(parameter, _MARKER):
                raise ValueError(f"already initialized parameter: {name}.{parameter_name}")
            if id(parameter) in seen_parameters:
                raise ValueError(f"aliased/overlapping parameter: {name}.{parameter_name}")
            seen_parameters.add(id(parameter))
            if isinstance(parameter, nn.parameter.UninitializedParameter) or parameter.device.type == "meta" or parameter.layout != torch.strided:
                raise ValueError(f"materialized strided parameter required: {name}.{parameter_name}")
            if parameter.grad is not None:
                raise ValueError(f"existing gradients are not fresh: {name}.{parameter_name}")
            if parameter.numel():
                key = (str(parameter.device), parameter.untyped_storage().data_ptr())
                if key in storages:
                    raise ValueError(f"shared parameter storage: {name}.{parameter_name}")
                storages.add(key)
        for buffer in module.buffers():
            if buffer.device.type == "meta" or buffer.layout != torch.strided:
                raise ValueError("materialized strided buffers required")
            if buffer.numel():
                buffers.add((str(buffer.device), buffer.untyped_storage().data_ptr()))
    if storages & buffers:
        raise ValueError("parameter/buffer shared storage is unsupported")

    staged, records = [], {}
    for component_name, policy in components.items():
        module, method = policy.module, policy.method
        if not isinstance(policy.parameter_seeds, Mapping) or not isinstance(policy.distributions, Mapping):
            raise ValueError("parameter_seeds/distributions must be named mappings")
        if type(policy.gain) not in (int, float) or not math.isfinite(policy.gain) or policy.gain <= 0:
            raise ValueError("gain must be finite and positive")
        if method != "xavier_uniform_zero_bias_v1" and policy.gain != 1:
            raise ValueError("gain applies only to Xavier")
        if method == "keep":
            if policy.parameter_seeds or policy.distributions:
                raise ValueError("keep scopes require no seeds/distributions")
            records[component_name] = {"method": method, "module_type": type(module).__qualname__}
            continue
        if type(module) not in _LEAVES or any(module.children()):
            raise ValueError("mutating scopes must be supported parameter-only leaves; declare composite leaves separately")
        originals = {n: p for n, p in module.named_parameters() if p.requires_grad and p.numel()}
        if not originals:
            raise ValueError("mutating scope has no nonempty trainable parameters; use keep")
        if set(policy.parameter_seeds) != set(originals) or any(type(s) is not int or not 0 <= s < 2**63 for s in policy.parameter_seeds.values()):
            raise ValueError("parameter_seeds must give every nonempty trainable name one integer in [0,2**63)")
        if set(policy.distributions) - set(originals):
            raise ValueError("distributions must name nonempty trainable parameters only")
        declared = pg_init.declarations(module)
        specs = {n: policy.distributions.get(n, declared.get(n)) for n in originals}
        serialized_specs = {n: _spec(s) for n, s in specs.items()}
        if method in {"xavier_uniform_zero_bias_v1", "identity_linear_v1"}:
            if type(module) is not nn.Linear or policy.distributions:
                raise ValueError("literal Linear methods require an exact Linear and no distribution overrides")
            if any(not p.requires_grad for p in module.parameters()):
                raise ValueError("literal Linear methods require wholly owned trainable leaves; use keep or explicit sampled/QR scopes for partial freezing")
            if method == "identity_linear_v1" and module.in_features != module.out_features:
                raise ValueError("identity requires one square Linear")
        cpu_parameters, detail, generators = {}, {}, {}
        for index, (name, parameter) in enumerate(originals.items()):
            if not parameter.is_contiguous() or not parameter.is_floating_point() or parameter.is_complex():
                raise ValueError("owned trainables must be contiguous real floating parameters")
            before = parameter.detach().cpu().clone()
            if not torch.isfinite(before).all():
                raise ValueError("nonfinite constructor values are not supported")
            spec = specs[name]
            if isinstance(spec, pg_init.R2Normal) and parameter.ndim != 2:
                raise ValueError("R2Normal requires a 2D existing particle table")
            # Never turn neutral zero-up/explicit KEEP into a nonzero adapter.
            protected = spec is pg_init.KEEP or bool(torch.all(before == 0))
            alias = name if method in {"identity_linear_v1", "xavier_uniform_zero_bias_v1"} else f"p{index}"
            cpu_parameters[alias] = nn.Parameter(before.clone(), requires_grad=not protected)
            detail[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                            "seed": policy.parameter_seeds[name], "host_declaration": serialized_specs[name],
                            "protected": protected, "before_sha256": _tensor_sha(before)}
            if protected:
                operation = {"kind": "keep", "reason": "explicit-KEEP" if spec is pg_init.KEEP else "zero-constructor"}
            elif method == "xavier_uniform_zero_bias_v1":
                bound = policy.gain * math.sqrt(6 / (module.in_features + module.out_features))
                operation = {"kind": "Uniform", "low": -bound, "high": bound} if name == "weight" else {"kind": "zero-bias"}
            elif method == "identity_linear_v1":
                operation = {"kind": "identity" if name == "weight" else "zero-bias"}
            else:
                operation = {"kind": method, "distribution": serialized_specs[name]}
            detail[name]["effective_operation"] = operation
            staged.append((parameter, before, cpu_parameters[alias]))
            random_draw = not protected and (method == "sample_distributions_v1" or
                                           method == "xavier_uniform_zero_bias_v1" and name == "weight")
            if random_draw:
                g = torch.Generator(device="cpu").manual_seed(policy.parameter_seeds[name])
                generators[alias] = g
                detail[name]["generator_before_sha256"] = _tensor_sha(g.get_state())
        if method in {"xavier_uniform_zero_bias_v1", "identity_linear_v1"}:
            holder = _linear_holder(module, cpu_parameters)
        else:
            holder = _DeclaredParameters(cpu_parameters, {f"p{i}": specs[n] for i, n in enumerate(originals)})
        if method == "deterministic_orthogonal":
            pg_init.deterministic_orthogonal_(holder, strict=True,
                parameter_seeds={f"p{i}": policy.parameter_seeds[n] for i, n in enumerate(originals) if cpu_parameters[f"p{i}"].requires_grad})
        else:
            pg_init.initialize_(holder, method=method, parameter_generators=generators,
                                gain=policy.gain, strict=True)
        for index, name in enumerate(originals):
            alias = name if method in {"identity_linear_v1", "xavier_uniform_zero_bias_v1"} else f"p{index}"
            value = cpu_parameters[alias]
            if not torch.isfinite(value).all():
                raise ValueError("initializer produced nonfinite values")
            detail[name]["after_sha256"] = _tensor_sha(value)
            if alias in generators:
                detail[name]["generator_after_sha256"] = _tensor_sha(generators[alias].get_state())
        records[component_name] = {"method": method, "module_type": type(module).__qualname__,
                                   "gain": policy.gain, "parameters": detail}

    document = {"schema": "particle-sliders-scoped-fresh-initialization-v1", "cohort_id": cohort_id,
                "source_identity": source_identity, "completed_steps": completed_steps, "restored_from": restored_from,
                "particlegan": {"version": version, "initializer_source": pg_source},
                "helper_source": _source(__file__), "components": records}
    receipt = InitializationReceipt(_canonical(document))
    # All validation/staging precedes mutation. Roll back values on a copy exception.
    with torch.no_grad():
        copied = []
        try:
            for original, before, value in staged:
                copied.append((original, before))
                original.copy_(value)
        except Exception:
            for original, before in copied:
                original.copy_(before)
            raise
    for policy in components.values():
        setattr(policy.module, _MARKER, receipt.sha256)
        for parameter in policy.module.parameters():
            setattr(parameter, _MARKER, receipt.sha256)
    return receipt
