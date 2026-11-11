"""Native packed Flux forward and ordinary LoRA, without GAN math or reshaping.

This is an explicit prerequisite proof, not a migrated/shared trainer. The
historical trainer predicts scheduler-step latents; this opt-in seam observes
raw packed velocity, matching the shared positive-response target convention.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn
from particle_sliders.pg09_backends import FrozenResponseAdapter

from . import COHORT
from .provenance import ROOT, checked_builder, check_native


class PackedTrainingUnavailable(ValueError):
    """Shared admission does not support the product's native packed response."""


class FluxSourceMismatch(RuntimeError):
    """The installed ParticleGAN version/sources cannot support this preflight."""


def preflight_training(response_shape, *, model=None, cohort=COHORT):
    """Check native shape without loading or mutating an optional staged model.

    Do not flatten [1,T,64], squeeze T, or turn tokens into channels/spatial axes
    to satisfy an unrelated domain. Probe the existing continuous_field rank
    gate with the unchanged rank3 shape; this does not declare packed tokens
    to be a supported field or invent a new shared domain.
    """
    if cohort != COHORT:
        raise ValueError("unsupported Flux prerequisite cohort")
    if model is not None:
        check_native(model)
        if model.config.in_channels != 64 or model.config.out_channels not in (None, 64):
            raise ValueError("staged Flux packed64 input/output config required")
    checked_builder()
    shape = tuple(response_shape)
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 64 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("native Flux response must be [1,image_tokens,64]")
    from particle_sliders.pg094 import validate_pg09_domain
    try:
        validate_pg09_domain(shape, response_domain="continuous_field")
    except RuntimeError as exc:
        raise FluxSourceMismatch(
            "Flux preflight requires the exact ParticleGAN0.9.4 release source; "
            "installed version/source verification failed: " + str(exc)) from exc
    except ValueError as exc:
        raise PackedTrainingUnavailable(
            "core0.12/PG0.9.4 continuous_field rank gate rejects native packed rank3 responses; "
            "Flux shared training/resume/EMA/export remain unavailable. "
            "No hooks or weights were changed; no response axes were recast.") from exc
    # A future core must be separately source-qualified; its new API is not
    # anticipated here and a changed source closure is rejected above.
    raise PackedTrainingUnavailable("packed-domain lifecycle/critic qualification is still required")


def _source(cls):
    return hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()


class HNDLLoRA(nn.Module):
    """Exactly the ordinary bias-free down/up branch used by LoRAModule.

    The host retains its own forward; a removable public forward hook adds the
    branch. B and the fixed native token dimension remain separate graph axes.
    """
    def __init__(self, host, rank, alpha, builder, identity, *, tokens, _rng=None):
        super().__init__()
        self.tokens = tokens
        with torch.random.fork_rng(devices=[]):
            if _rng is not None:
                torch.set_rng_state(_rng.get_state())
            try:
                self.graph = builder((ROOT / "lora.hndl").read_text(),
                    input_shape={"value": ["B", tokens, host.in_features]},
                    output_shape=["B", tokens, host.out_features],
                    parameters={"rank": rank, "output_width": host.out_features})
                nn.init.kaiming_uniform_(self.graph["down"].weight, a=1)
                nn.init.zeros_(self.graph["up"].weight)
            finally:
                if _rng is not None:
                    _rng.set_state(torch.get_rng_state())
        self.scale = alpha / rank
        self._identity = deepcopy(identity)

    def extra_repr(self):
        return json.dumps(dict(self._identity, alpha_over_rank=self.scale, native_tokens=self.tokens), sort_keys=True)

    def forward(self, value):
        if value.ndim != 3 or value.shape[1] != self.tokens:
            raise ValueError("Flux attention projections require native [B,T,D]")
        return self.graph(value=value.to(self.graph["down"].weight)) * self.scale


class FluxLoRA(nn.Module):
    """Borrow an already staged frozen native model and restore every hook/mode.

    Only historical full/Attention linear targets are supported: add_* text
    projections are excluded exactly as in conceptmod's LoRANetwork selector.
    No model parameters are registered in the adapter network.
    """
    def __init__(self, model, *, image_tokens, text_tokens, rank=4, alpha=1.):
        super().__init__()
        check_native(model)
        if any(type(v) is not int or v < 1 for v in (image_tokens, text_tokens)):
            raise ValueError("fixed positive native image/text token counts required")
        self.image_tokens, self.text_tokens = image_tokens, text_tokens
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("positive rank/alpha required")
        if model.config.in_channels != 64 or model.config.out_channels not in (None, 64):
            raise ValueError("Flux packed64 input/output config required")
        if getattr(model, "gradient_checkpointing", False):
            raise ValueError("checkpointed recomputation needs reviewed scale capture; unsupported")
        if any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("stage a frozen float32 model before binding")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("this prerequisite is CPU-qualified only")
        builder, identity = checked_builder()
        targets, seen = [], set()
        for name, module in model.named_modules():
            if module.__class__.__name__ == "FluxAttention":
                for child_name, child in module.named_modules():
                    if isinstance(child, nn.Linear) and "add_" not in child_name and id(child) not in seen:
                        seen.add(id(child))
                        targets.append((name + "." + child_name, child))
        if not targets:
            raise ValueError("no historical full Attention linear targets")
        self.target_names = tuple(name for name, _ in targets)
        # Keep the caller's CPU RNG unchanged on successful construction and
        # graph/initialization failure. The private fork still advances between
        # branches, preserving distinct sequential native initializations.
        with torch.random.fork_rng(devices=[]):
            branch_rng = torch.Generator().set_state(torch.get_rng_state())
            self.branches = nn.ModuleList([HNDLLoRA(host, rank, alpha, builder, identity,
                tokens=image_tokens + text_tokens if name.startswith("single_transformer_blocks.") else image_tokens,
                _rng=branch_rng)
                for name, host in targets])
        self.names = tuple(name for name, _ in self.named_parameters())
        object.__setattr__(self, "_model", model)
        self._targets = targets
        self._model_config = deepcopy(dict(model.config))
        self._base_ids = tuple(id(p) for p in model.parameters())
        self._handles = []
        self._modes = None
        self.strength = 0.

    @property
    def attached(self):
        return bool(self._handles)

    def attach(self):
        if self.attached:
            raise RuntimeError("Flux hooks already attached")
        check_native(self._model)
        checked_builder()
        if (dict(self._model.config) != self._model_config
                or tuple(id(p) for p in self._model.parameters()) != self._base_ids
                or any(p.requires_grad or p.dtype != torch.float32 or p.device.type != "cpu"
                       for p in self._model.parameters())
                or getattr(self._model, "gradient_checkpointing", False)):
            raise ValueError("staged Flux model changed before attachment")
        if "forward" in self._model.__dict__ or any(host._forward_hooks or host._forward_pre_hooks or "forward" in host.__dict__
               for _, host in self._targets):
            raise ValueError("Flux target already carries hooks or a replaced forward")
        self._modes = [(m, m.training) for m in self._model.modules()]
        try:
            self._model.eval()
            for (_, host), branch in zip(self._targets, self.branches):
                def hook(module, args, output, branch=branch):
                    if self.strength == 0:
                        return output
                    return output + (branch(args[0]) * self.strength).to(output)
                self._handles.append(host.register_forward_hook(hook))
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                original.add_note("Flux cleanup also failed: " + repr(cleanup))
            raise
        return self

    def close(self):
        errors, remaining = [], []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
                remaining.append(handle)
        self._handles = remaining
        if self._modes is not None:
            for module, training in self._modes:
                try:
                    module.training = training
                except BaseException as exc:
                    errors.append(exc)
            self._modes = None
        self.strength = 0.
        if errors:
            raise RuntimeError("Flux cleanup attempted every hook and mode; failures occurred") from errors[0]

    @contextmanager
    def borrowed(self):
        self.attach()
        try:
            yield self
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                original.add_note("Flux cleanup also failed: " + repr(cleanup))
            raise
        else:
            self.close()

    @contextmanager
    def scaled(self, scale):
        if not self.attached or not math.isfinite(scale):
            raise ValueError("attached bank and finite strength required")
        previous = self.strength
        self.strength = float(scale)
        try:
            yield self
        finally:
            self.strength = previous


class PackedFluxResponse(FrozenResponseAdapter):
    family = "flux-packed-prerequisite"
    time_coordinates = "raw scheduler t / 1000; raw packed velocity; no scheduler.step"

    def __init__(self, model, network, *, model_identity):
        if network._model is not model or not isinstance(model_identity, dict) or not model_identity:
            raise ValueError("bind the actual bank/model and explicit staged identity")
        self.transformer = model
        self._config = deepcopy(dict(model.config))
        self._model_source = _source(type(model))
        super().__init__(network, model_identity=model_identity)

    def _velocity(self, condition, latent, timestep):
        if dict(self.transformer.config) != self._config:
            raise ValueError("staged Flux config changed")
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (1, self.network.image_tokens, 64) or not torch.isfinite(latent).all():
            raise ValueError("finite native [1,image_tokens,64] latent required")
        required = {"embeds", "pooled", "text_ids", "image_ids", "guidance"}
        if not isinstance(condition, dict) or set(condition) != required:
            raise ValueError("complete native Flux conditioning dictionary required")
        c = condition
        if (not isinstance(timestep, torch.Tensor)
                or any(not isinstance(c[key], torch.Tensor) for key in required - {"guidance"})):
            raise ValueError("native conditioning and timestep tensors required")
        text_width = self.transformer.config.joint_attention_dim
        pool_width = self.transformer.config.pooled_projection_dim
        if (tuple(c["embeds"].shape) != (1, self.network.text_tokens, text_width)
                or tuple(c["pooled"].shape) != (1, pool_width)
                or tuple(c["text_ids"].shape) != (c["embeds"].shape[1], 3)
                or tuple(c["image_ids"].shape) != (latent.shape[1], 3)):
            raise ValueError("native text/image token axes or feature dimensions differ")
        guided = bool(self.transformer.config.guidance_embeds)
        if guided != (isinstance(c["guidance"], torch.Tensor)):
            raise ValueError("guidance tensor must match Flux model guidance_embeds")
        if guided and tuple(c["guidance"].shape) != (1,):
            raise ValueError("one native guidance value per row required")
        tensors = [latent, timestep] + [v for v in c.values() if isinstance(v, torch.Tensor)]
        if any(v.dtype != torch.float32 or v.device != latent.device or not torch.isfinite(v).all() for v in tensors):
            raise ValueError("native CPU proof requires finite colocated float32 tensors")
        if timestep.numel() != 1 or not 0 <= float(timestep) <= 1000:
            raise ValueError("one raw timestep in [0,1000] required")
        result = self.transformer(hidden_states=latent, timestep=timestep.reshape(1) / 1000,
            encoder_hidden_states=c["embeds"], pooled_projections=c["pooled"],
            txt_ids=c["text_ids"], img_ids=c["image_ids"], guidance=c["guidance"], return_dict=False)[0]
        if result.shape != latent.shape or not torch.isfinite(result).all():
            raise ValueError("native Flux packed output shape or values differ")
        return result

    def metadata(self):
        return dict(super().metadata(), cohort=COHORT, model_config=self._config,
            model_source_sha256=self._model_source, target_names=self.network.target_names,
            response_coordinates="native packed [1,image_tokens,64]; no pooling/reshape",
            shared_training_admitted=False)
