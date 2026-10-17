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
    def supra(cls, steps):
        """Supra Hub convention: t=0 noise -> t=1 data, x += v/steps."""
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

    def __init__(self, network, *, decoder: Callable | None = None, model_identity=None):
        from copy import deepcopy
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
        if row.timestep is not None and not torch.isfinite(row.timestep).all():
            raise ValueError("finite timestep required")
        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
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
            if self.family == "moss":
                if row.schedule is not None or row.guidance != 1:
                    raise ValueError("MOSS signed teacher uses raw velocity; CFG is inference-only")
                negative = self._response(row, condition=row.negative, scale=0.)
                target = neutral + row.scale * row.strength * (positive - negative)
            elif row.scale != 1 or row.strength != 1:
                raise ValueError("Anima/Supra use unipolar positive targets at scale +1")
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
                    target_convention=("neutral + sign*strength*(positive-negative)" if self.family == "moss"
                                       else "frozen positive response"),
                    checkpoint_scale_capture="per-forward captured context",
                    time_coordinates=self.time_coordinates,
                    student_conditioning="neutral", cfg_adapter_branch="conditional_only",
                    negative_layer_skip=False, adapter_names=list(self.network.names),
                    adapter_dtype="float32", quality_qualified=False)


class AnimaResponseAdapter(FrozenResponseAdapter):
    family = "anima"
    time_coordinates = "scheduler training timestep; LiveAnimaBackend normalizes by num_train_timesteps"

    def __init__(self, backend, *, decoder=None, model_identity=None):
        self.backend = backend
        super().__init__(_adapter_parameters(backend), decoder=decoder, model_identity=model_identity)
        roots = list(_roots(backend))
        pipe = getattr(backend, "pipe", None)
        if pipe is not None and hasattr(pipe, "transformer"):
            roots.append(("pipe.transformer", pipe.transformer))
        seen = set()
        for _, root in roots:
            if id(root) not in seen:
                bind_native_gradient_checkpointing(root, self.network)
                seen.add(id(root))

    def _velocity(self, condition, latent, timestep):
        # No scale argument: the outer reversible context must survive backward.
        # Encoding is live, so conditioner LoRA is never replaced by frozen embeds.
        return self.backend.predict_v(condition, latent, timestep)


class SupraResponseAdapter(AnimaResponseAdapter):
    family = "supra"
    time_coordinates = "Supra flow time t=0 noise -> t=1 data; dx/dt=velocity"

    def _velocity(self, condition, latent, timestep):
        if torch.any((timestep < 0) | (timestep > 1)):
            raise ValueError("Supra time must be flow sigma in [0,1]")
        # LiveSupraBackend.predict_v resets scale to 1 internally. Reuse its frozen
        # encoding/context and native transformer directly under our scale scope.
        if hasattr(self.backend, "_context"):
            embeds, mask = self.backend._context(condition)
            hidden = latent.to(self.backend.device)
            embeds = embeds.to(hidden)
            if embeds.ndim == 2:
                embeds = embeds.unsqueeze(0)
            time = timestep.reshape(-1).to(hidden.device, torch.float32).expand(hidden.shape[0])
            mask = None if mask is None else mask.to(hidden.device)
            return self.backend.transformer(hidden, time, embeds, mask).float()
        embeds, _ = self.backend.encode_text(condition)
        return self.backend._forward(latent, timestep, embeds)


class MossResponseAdapter(FrozenResponseAdapter):
    family = "moss"
    time_coordinates = "official MOSS scheduler timestep = 1000*sigma"

    def __init__(self, transformer, slider, *, decoder=None, model_identity=None):
        self.transformer, self.slider = transformer, slider
        named = list(slider.named_parameters())
        if any(p.requires_grad for p in transformer.parameters()):
            raise ValueError("MOSS base transformer must be frozen")
        super().__init__(AdapterParameters(named, slider.at), decoder=decoder, model_identity=model_identity)

    def _velocity(self, condition, latent, timestep):
        embeds, mask = condition
        return self.transformer(latent, timestep, embeds, mask).sample.float()


def checkpoint_with_adapter_scale(module, network):
    """Checkpoint a block with the forward scale captured for recomputation.

    Install instead of an existing checkpoint wrapper. Double wrapping is refused.
    Caller owns which blocks to checkpoint; no global service/model mutation.
    """
    from torch.utils.checkpoint import checkpoint
    if getattr(module, "_pg09_scale_checkpoint", False):
        raise ValueError("block already checkpointed with adapter scale")
    original = module.forward

    def forward(*args, **kwargs):
        scale = network.scale
        return checkpoint(original, *args, use_reentrant=False,
                          context_fn=lambda: (network.scaled(scale), network.scaled(scale)), **kwargs)

    module.forward = forward
    module._pg09_scale_checkpoint = True


def bind_native_gradient_checkpointing(root, network):
    """Use the model's public checkpoint hook with captured adapter scales.

    Existing enabled Diffusers checkpointing needs this hook too: conditioner
    LoRA and later trajectory steps can otherwise recompute at a frozen scale.
    """
    from torch.utils.checkpoint import checkpoint
    if not getattr(root, "is_gradient_checkpointing", False):
        return
    enable = getattr(root, "enable_gradient_checkpointing", None)
    if not callable(enable):
        raise ValueError("enabled checkpointing has no replaceable public hook")

    def checkpoint_fn(function, *args, **kwargs):
        scale = network.scale
        return checkpoint(function, *args, use_reentrant=False,
                          context_fn=lambda: (network.scaled(scale), network.scaled(scale)), **kwargs)

    # A backend whose native checkpoint hook cannot be replaced must fail here.
    enable(gradient_checkpointing_func=checkpoint_fn)


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


class AnimaRuntimeResponseAdapter(FrozenResponseAdapter):
    """Reuse lumen_studio's TurboRuntime, actual ParticleAdapter and Mixer.

    No copied router/particle prior. The supplied adapter must already be attached
    to this runtime under ``name``. The same actual bank is exposed to core.
    """
    family = "anima_turbo_runtime"
    time_coordinates = "official scheduler timestep; runtime.predict divides by training timestep count"

    def __init__(self, runtime, adapter, name):
        if runtime.mixer.adapters.get(name) is not adapter:
            raise ValueError("attach this actual particle adapter to runtime.mixer first")
        if any(p.requires_grad for p in runtime.transformer.parameters()):
            raise ValueError("Anima runtime base must be frozen")
        self.runtime, self.adapter, self.name = runtime, adapter, name
        network = AdapterParameters(adapter.named_parameters(),
                                    lambda scale: runtime.mixer.scales({name: scale}))
        self.particles = adapter.particles  # the actual routed bank, never a replacement
        super().__init__(network, decoder=runtime.decode, model_identity=runtime.identity)
        bind_native_gradient_checkpointing(runtime.transformer, network)

    def _velocity(self, condition, latent, timestep):
        embedding = self.runtime.encode(condition) if isinstance(condition, str) else condition
        return self.runtime.predict(latent, timestep, embedding)

    def _advance(self, latent, velocity, delta):
        # Match TurboRuntime.step: scheduler arithmetic in float32 with the
        # official bf16 model-output rounding both before and after Euler.
        velocity = velocity.to(self.runtime.dtype)
        return (latent.float() + delta * velocity.float()).to(self.runtime.dtype)

    def metadata(self):
        return dict(super().metadata(), model_identity=self.runtime.identity,
                    particle_bank="existing lumen_studio ParticleAdapter.particles",
                    trajectory_rounding=str(self.runtime.dtype))


class SupraRuntimeResponseAdapter(SupraResponseAdapter):
    """Reuse supra.runtime.SupraRuntime's frozen text cache, DiT, LoRA and VAE."""
    family = "supra_runtime"

    def __init__(self, runtime, *, model_identity=None):
        from types import SimpleNamespace
        self.runtime = runtime
        backend = SimpleNamespace(transformer=runtime.model, device=runtime.device,
                                  trainable_parameters=runtime.parameters, _context=runtime.encode)
        super().__init__(backend, decoder=self._decode, model_identity=model_identity)

    def _velocity(self, condition, latent, timestep):
        # Preserve product autocast while controlling conditional/negative scales
        # separately, unlike legacy runtime.velocity's concatenated CFG branches.
        from contextlib import nullcontext
        context = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if self.runtime.device.type == 'cuda' else nullcontext())
        with context:
            return super()._velocity(condition, latent, timestep)

    def _decode(self, latent):
        from contextlib import nullcontext
        context = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if self.runtime.device.type == 'cuda' else nullcontext())
        with context:
            return self.runtime.vae.decode(latent / .18215).sample
