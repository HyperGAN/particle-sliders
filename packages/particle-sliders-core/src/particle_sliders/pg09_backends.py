"""Frozen-model response adapters. Losses and optimizers belong to the shared core.

No model loader runs on import. Existing loaders/LoRA implementations are reused;
these adapters never call the legacy MSE trainers or merge adapter weights.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import torch
from torch import nn


class AdapterParameters(nn.Module):
    """Register only adapter leaves, preserving their identity and target names.

    The owner keeps ``scaled`` active through backward. Per-forward checkpoint
    wrappers additionally capture the scale so signed rows can share a batch.
    """

    def __init__(self, named, scale_context):
        super().__init__()
        named = list(named)
        if not named or len({id(p) for _, p in named}) != len(named):
            raise ValueError("adapter parameters must be nonempty and unique")
        if any(not p.requires_grad or p.dtype != torch.float32 for _, p in named):
            raise ValueError("adapter leaves must be trainable float32 tensors")
        self.names = tuple(n for n, _ in named)
        self.weights = nn.ParameterList([p for _, p in named])
        self._scale_context = scale_context
        self._particle_index = next((i for i, (n, _) in enumerate(named) if n == "particles"), None)
        self.scale = 1.0

    @property
    def particles(self):
        if self._particle_index is None:
            raise AttributeError("this adapter has no routed particle bank")
        return self.weights[self._particle_index]

    @contextmanager
    def scaled(self, scale):
        scale = float(scale)
        if not isfinite(scale):
            raise ValueError("adapter scale must be finite")
        previous = self.scale
        with self._scale_context(scale):
            self.scale = scale
            try:
                yield self
            finally:
                self.scale = previous


def _roots(backend):
    if hasattr(backend, "_adapted_nn_modules"):
        return [(str(i), m) for i, m in enumerate(backend._adapted_nn_modules())]
    return [("transformer", backend.transformer)]


def _lora_scale_context(roots):
    """Restore exact heterogeneous scales, including nested zero/negative calls.

    Local LoRALinear and PEFT LoraLayer carry the existing adapters. PEFT scale
    dictionaries are snapshotted; alpha/rank factors are retained exactly.
    No enable/disable call changes requires_grad flags during a frozen forward.
    """
    layers = []
    seen = set()
    for _, root in roots:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, "multiplier") and hasattr(module, "down") and hasattr(module, "up"):
                layers.append((module, "multiplier", float(module.multiplier)))
            elif hasattr(module, "lora_A") and hasattr(module, "scaling"):
                if getattr(module, "disable_adapters", False):
                    raise ValueError("enable the selected PEFT adapter before binding")
                layers.append((module, "scaling", dict(module.scaling)))
    if not layers:
        raise ValueError("no supported local or PEFT LoRA layers")

    @contextmanager
    def scaled(value):
        previous = []
        try:
            for module, attr, initial in layers:
                old = getattr(module, attr)
                previous.append((module, attr, dict(old) if isinstance(old, dict) else old))
                if attr == "scaling":
                    module.scaling = {k: v * value for k, v in initial.items()}
                else:
                    module.multiplier = initial * value
            yield
        finally:
            for module, attr, old in reversed(previous):
                setattr(module, attr, old)
    return scaled


lora_scale_context = _lora_scale_context


def _adapter_parameters(backend):
    roots = _roots(backend)
    requested = {id(p) for p in backend.trainable_parameters()}
    for _, root in roots:
        root.eval()
    named = []
    seen = set()
    for prefix, root in roots:
        for name, p in root.named_parameters():
            if id(p) in requested and id(p) not in seen:
                named.append((f"{prefix}.{name}", p))
                seen.add(id(p))
            elif p.requires_grad and id(p) not in requested:
                raise ValueError(f"unowned trainable base parameter: {prefix}.{name}")
    if seen != requested:
        raise ValueError("trainable adapter leaves missing from model roots")
    return AdapterParameters(named, _lora_scale_context(roots))


@dataclass(frozen=True)
class FlowSchedule:
    """Explicit Euler intervals and model-native times; no implicit time guessing."""
    timesteps: tuple[float, ...]
    deltas: tuple[float, ...]

    def __post_init__(self):
        if not self.timesteps or len(self.timesteps) != len(self.deltas):
            raise ValueError("one timestep per Euler interval required")
        if any(not isfinite(v) for v in self.timesteps + self.deltas):
            raise ValueError("finite schedule required")
        if any(v == 0 for v in self.deltas) or any(v * self.deltas[0] <= 0 for v in self.deltas):
            raise ValueError("Euler intervals must share a nonzero direction")

    @classmethod
    def increasing_linear(cls, steps):
        """Uniform t=0 to t=1 trajectory with positive Euler intervals."""
        if type(steps) is not int or steps < 1:
            raise ValueError("positive integer steps required")
        return cls(tuple(i / steps for i in range(steps)), tuple(1. / steps for _ in range(steps)))

    @classmethod
    def from_scheduler(cls, scheduler):
        """Read already-configured official FlowMatchEuler intervals without shifting twice."""
        times = tuple(float(v) for v in scheduler.timesteps.detach().cpu())
        sigmas = tuple(float(v) for v in scheduler.sigmas.detach().cpu())
        if len(sigmas) != len(times) + 1:
            raise ValueError("scheduler must expose one terminal sigma")
        return cls(times, tuple(b - a for a, b in zip(sigmas, sigmas[1:])))

    @classmethod
    def linear(cls, steps, *, time_scale):
        if type(steps) is not int or steps < 1 or not isfinite(time_scale) or time_scale <= 0:
            raise ValueError("positive steps and model-native time scale required")
        sigmas = torch.linspace(1, 0, steps + 1, dtype=torch.float64).tolist()
        return cls(tuple(s * time_scale for s in sigmas[:-1]),
                   tuple(b - a for a, b in zip(sigmas, sigmas[1:])))


@dataclass(frozen=True)
class ResponseRow:
    neutral: Any
    positive: Any
    negative: Any
    latent: torch.Tensor
    timestep: torch.Tensor | None = None
    schedule: FlowSchedule | None = None
    scale: float = 1.0
    strength: float = 1.0
    guidance: float = 1.0


@dataclass(frozen=True)
class PreparedResponse:
    row: ResponseRow
    target: torch.Tensor
    neutral: torch.Tensor


class FrozenResponseAdapter:
    """Single interface consumed by the reusable adversarial game.

    ``prepare`` caches only frozen teacher responses. ``generated`` re-evaluates
    adapted conditioning each time. ``render`` uses the identical denoise/CFG
    implementation and an injected VAE decoder; it never invokes an unrelated
    pipeline with different adapter branches or negative-layer skipping.
    """
    family = "abstract"
    time_coordinates = "abstract"

    def __init__(self, network, *, decoder: Callable | None = None, model_identity=None,
                 trajectory_direction="decreasing", teacher_convention="positive"):
        from copy import deepcopy
        if trajectory_direction not in ("increasing", "decreasing"):
            raise ValueError("trajectory_direction must be increasing or decreasing")
        if teacher_convention not in ("positive", "signed_difference"):
            raise ValueError("teacher_convention must be positive or signed_difference")
        self.trajectory_direction = trajectory_direction
        self.teacher_convention = teacher_convention
        self.network = network
        self.decoder = decoder
        if any(p.device.type != "cpu" for p in network.parameters()) and not model_identity:
            raise ValueError("live backend requires verified staged-model identity")
        self.model_identity = deepcopy(model_identity or {"qualification": "CPU fixture only"})

    def _velocity(self, condition, latent, timestep):
        raise NotImplementedError

    def velocity(self, condition, latent, timestep, *, scale=1.0):
        with self.network.scaled(scale):
            return self._velocity(condition, latent, timestep).float()

    def cfg(self, row, latent, timestep, *, scale):
        # Student-only conditional training means the negative branch stays frozen.
        positive = self.velocity(row.neutral, latent, timestep, scale=scale)
        if row.guidance == 1:
            return positive
        # Frozen weights still differentiate through trajectory latents.
        negative = self.velocity(row.negative, latent, timestep, scale=0.)
        return negative + row.guidance * (positive - negative)

    def _advance(self, latent, velocity, delta):
        return latent + delta * velocity

    def _response(self, row, *, condition=None, scale=1.0):
        if condition is not None:
            from dataclasses import replace
            row = replace(row, neutral=condition)
        if row.schedule is None:
            return self.cfg(row, row.latent, row.timestep, scale=scale)
        latent = row.latent.clone()
        for time, delta in zip(row.schedule.timesteps, row.schedule.deltas):
            timestep = latent.new_tensor(time)
            latent = self._advance(latent, self.cfg(row, latent, timestep, scale=scale), delta)
        return latent

    def prepare(self, row):
        from dataclasses import replace
        if (not isfinite(row.scale) or not isfinite(row.strength) or row.strength < 0 or
                not isfinite(row.guidance) or row.guidance < 0):
            raise ValueError("finite scale/guidance and nonnegative strength required")
        if row.latent.ndim < 2 or row.latent.shape[0] != 1 or not torch.isfinite(row.latent).all():
            raise ValueError("finite single-example latent required")
        if (row.timestep is None) == (row.schedule is None):
            raise ValueError("choose one velocity timestep or a trajectory schedule")
        if row.schedule is not None:
            if self.trajectory_direction == 'increasing' and any(d <= 0 for d in row.schedule.deltas):
                raise ValueError("increasing trajectory requires positive Euler intervals")
            if self.trajectory_direction == 'decreasing' and any(d >= 0 for d in row.schedule.deltas):
                raise ValueError("decreasing trajectory requires negative Euler intervals")
        if row.timestep is not None and not torch.isfinite(row.timestep).all():
            raise ValueError("finite timestep required")
        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, dict):
                return {key: snapshot(v) for key, v in value.items()}
            if isinstance(value, tuple):
                return tuple(snapshot(v) for v in value)
            if isinstance(value, list):
                return [snapshot(v) for v in value]
            return value
        row = replace(row, neutral=snapshot(row.neutral), positive=snapshot(row.positive),
                      negative=snapshot(row.negative))
        # Caller mutation cannot change the cached input trajectory.
        row = replace(row, latent=row.latent.detach().clone(),
                      timestep=row.timestep.detach().clone() if row.timestep is not None else None)
        with torch.no_grad(), self.network.scaled(0.):
            neutral = self._response(row, scale=0.)
            positive = self._response(row, condition=row.positive, scale=0.)
            target = positive
            if self.teacher_convention == "signed_difference":
                if row.schedule is not None or row.guidance != 1:
                    raise ValueError("signed-difference teacher requires raw velocity; CFG is inference-only")
                negative = self._response(row, condition=row.negative, scale=0.)
                target = neutral + row.scale * row.strength * (positive - negative)
            elif row.scale != 1 or row.strength != 1:
                raise ValueError("positive teacher requires unipolar targets at scale +1")
        target, neutral = target.detach(), neutral.detach()
        if not torch.isfinite(target).all() or not torch.isfinite(neutral).all():
            raise ValueError("nonfinite frozen response")
        return PreparedResponse(row, target, neutral)

    def generated(self, prepared):
        value = self._response(prepared.row, scale=prepared.row.scale)
        if value.shape != prepared.target.shape or not torch.isfinite(value).all():
            raise ValueError("generated response differs in shape or is nonfinite")
        return value

    @torch.no_grad()
    def render(self, row, *, scale=1.0):
        if row.schedule is None:
            raise ValueError("render requires an explicit denoising schedule")
        if self.decoder is None:
            raise ValueError("bind the staged model's VAE decoder before rendering")
        return self.decoder(self._response(row, scale=scale))

    def metadata(self):
        return dict(protocol="frozen-response-pg090-v1", family=self.family,
                    model_identity=self.model_identity,
                    target_convention=("neutral + sign*strength*(positive-negative)" if self.teacher_convention == "signed_difference"
                                       else "frozen positive response"),
                    trajectory_direction=self.trajectory_direction,
                    teacher_convention=self.teacher_convention,
                    checkpoint_scale_capture="per-forward captured context",
                    time_coordinates=self.time_coordinates,
                    student_conditioning="neutral", cfg_adapter_branch="conditional_only",
                    negative_layer_skip=False, adapter_names=list(self.network.names),
                    adapter_dtype="float32", quality_qualified=False)


class _RepeatableScaleContext:
    """A checkpoint context may be entered for several autograd.grad replays."""
    def __init__(self, network, scale):
        self.network, self.scale = network, scale
        self._active = []

    def __enter__(self):
        context = self.network.scaled(self.scale)
        value = context.__enter__()
        self._active.append(context)
        return value

    def __exit__(self, *exc):
        return self._active.pop().__exit__(*exc)


def _checkpoint_contexts(network, scale):
    return _RepeatableScaleContext(network, scale), _RepeatableScaleContext(network, scale)


class AdapterScaleCheckpoint:
    """Undo handle for one eagerly installed, scale-capturing checkpoint.

    Use as a context manager or call ``close()`` after the owning operation.
    Closing twice is safe. A different owner's replacement is never overwritten.
    """

    def __init__(self, module, forward, saved):
        self._module, self._forward, self._saved = module, forward, saved
        self._closed = False
        self._context_depth = 0

    def close(self):
        if self._closed:
            return
        values = vars(self._module)
        if (values.get("forward") is not self._forward or
                values.get("_pg09_scale_checkpoint") is not True):
            raise RuntimeError("checkpoint ownership changed; restore the owned wrapper before closing")
        _restore_checkpoint_attributes(self._module, self._saved)
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("checkpoint handle is already closed")
        self._context_depth += 1
        return self

    def __exit__(self, *exc):
        self._context_depth -= 1
        if not self._context_depth:
            self.close()


def _restore_checkpoint_attributes(module, saved):
    # Restore instance ownership exactly: a class-defined forward must remain a
    # descriptor, rather than becoming a permanent bound-method override.
    values = vars(module)
    for name in ("forward", "_pg09_scale_checkpoint"):
        if name in saved:
            values[name] = saved[name]
        else:
            values.pop(name, None)


def checkpoint_with_adapter_scale(module, network):
    """Checkpoint a block with the forward scale captured for recomputation.

    Install instead of an existing checkpoint wrapper. Double wrapping is refused.
    Returns an eagerly installed :class:`AdapterScaleCheckpoint` undo handle.
    Existing callers may ignore the return and retain the prior installation
    lifetime. New owners should use a context manager or register ``close``.
    Caller owns which blocks to checkpoint; no global service/model mutation.
    """
    from torch.utils.checkpoint import checkpoint
    if getattr(module, "_pg09_scale_checkpoint", False):
        raise ValueError("block already checkpointed with adapter scale")
    original = module.forward
    saved = {name: vars(module)[name] for name in ("forward", "_pg09_scale_checkpoint")
             if name in vars(module)}

    def forward(*args, **kwargs):
        scale = network.scale
        return checkpoint(original, *args, use_reentrant=False,
                          context_fn=lambda: _checkpoint_contexts(network, scale), **kwargs)

    try:
        module.forward = forward
        module._pg09_scale_checkpoint = True
        return AdapterScaleCheckpoint(module, forward, saved)
    except BaseException:
        _restore_checkpoint_attributes(module, saved)
        raise


def bind_native_gradient_checkpointing(root, network):
    """Use the model's public checkpoint hook with captured adapter scales.

    Enabled Diffusers and Transformers checkpointing need captured contexts:
    conditioner LoRA and later trajectory steps can otherwise recompute at a
    frozen scale. Native enablement remains caller-owned.
    """
    import inspect
    from torch.utils.checkpoint import checkpoint
    if not getattr(root, "is_gradient_checkpointing", False):
        return

    def checkpoint_fn(function, *args, **kwargs):
        scale = network.scale
        return checkpoint(function, *args, use_reentrant=False,
                          context_fn=lambda: _checkpoint_contexts(network, scale), **kwargs)

    enable = getattr(root, "enable_gradient_checkpointing", None)
    if callable(enable):
        enable(gradient_checkpointing_func=checkpoint_fn)
        return
    enable = getattr(root, "gradient_checkpointing_enable", None)
    if callable(enable):
        if "gradient_checkpointing_kwargs" not in inspect.signature(enable).parameters:
            raise ValueError("Transformers checkpointing requires the public "
                             "gradient_checkpointing_enable(gradient_checkpointing_kwargs=...) API")
        setter = getattr(root, "_set_gradient_checkpointing", None)
        if callable(setter) and "value" in inspect.signature(setter).parameters:
            # Transformers explicitly ignores kwargs for this legacy format.
            raise ValueError("legacy Transformers checkpointing ignores gradient_checkpointing_kwargs; "
                             "use a model with the current public checkpoint API")
        enable(gradient_checkpointing_kwargs={
            "use_reentrant": False,
            # Transformers passes these kwargs to torch.checkpoint at each
            # forward. Capture that forward's scale in repeatable contexts,
            # rather than capturing the scale at backend-binding time.
            "context_fn": lambda: _checkpoint_contexts(network, network.scale),
        })
        return
    raise ValueError("enabled checkpointing has no replaceable public hook: expected "
                     "Diffusers enable_gradient_checkpointing(gradient_checkpointing_func=...) or "
                     "Transformers gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)")


def response_signature(adapter, prepared, *, model_fingerprint):
    """Bind cached responses to actual conditioning, times, frozen model and state."""
    from dataclasses import asdict
    import hashlib
    import json
    if not isinstance(model_fingerprint, str) or not model_fingerprint:
        raise ValueError("verified staged-model fingerprint required")

    def encode(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            return dict(shape=list(value.shape), dtype=str(value.dtype),
                        sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        return value
    payload = dict(backend=adapter.metadata(), model_fingerprint=model_fingerprint,
                   rows=[encode(dict(row=asdict(p.row), target=p.target, neutral=p.neutral))
                         for p in prepared])
    return dict(payload=payload, sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())


def save_adapter(adapter, path, *, model_fingerprint):
    """Portable FP32 adapter state; separate from the core's full optimizer packet."""
    import json
    from safetensors.torch import save_file
    if not isinstance(model_fingerprint, str) or not model_fingerprint:
        raise ValueError("verified staged-model fingerprint required")
    metadata = dict(backend=adapter.metadata(), model_fingerprint=model_fingerprint)
    save_file({k: v.detach().cpu().clone() for k, v in adapter.network.state_dict().items()},
              str(path), metadata={"pg09_backend": json.dumps(metadata, sort_keys=True)})


def load_adapter(adapter, path, *, model_fingerprint):
    """Validate the entire export before changing any adapter leaf."""
    import json
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata = json.loads((file.metadata() or {}).get('pg09_backend', '{}'))
    expected = dict(backend=adapter.metadata(), model_fingerprint=model_fingerprint)
    if metadata != expected:
        raise ValueError("backend protocol, frozen model or adapter target names differ")
    state = load_file(str(path), device='cpu')
    current = adapter.network.state_dict()
    if (state.keys() != current.keys() or any(
            t.shape != current[k].shape or t.dtype != torch.float32 or not torch.isfinite(t).all()
            for k, t in state.items())):
        raise ValueError("adapter export keys, shapes, dtype or finite values differ")
    adapter.network.load_state_dict(state, strict=True)
    return metadata
