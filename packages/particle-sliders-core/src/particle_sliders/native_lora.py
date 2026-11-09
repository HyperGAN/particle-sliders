"""Generic LoRA construction and activation over product-owned named targets.

The original name-based selector remains available through a compatibility
adapter. New integrations pass ``LoRANetwork(..., target_plan=...)`` directly.
"""
# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Optional, List

import torch
import torch.nn as nn

from .adapter_activation import activation_is_entered, activation_mutation, push_activation, restore_activation
from .adapter_attachment import atomic_attachment

# Re-export the historical import surface without embedding model selection in
# the generic construction/activation path.
from .native_lora_compat import (
    DEFAULT_TARGET_REPLACE,
    LORA_PREFIX_UNET,
    TRAINING_METHODS,
    UNET_TARGET_REPLACE_MODULE_CONV,
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER,
    lora_module_kind as _legacy_module_kind,
    lora_target_plan as _legacy_target_plan,
)


def lora_init_weights_advanced(lora_down, lora_up, init_scale=0.01, sparsity=0.9):
    fan_in = lora_down.weight.shape[1]
    fan_out = lora_up.weight.shape[0]
    mid_dim = lora_down.weight.shape[0]

    # Initialize lora_down with an asymmetric pattern
    with torch.no_grad():
        lora_down.weight.zero_()
        for i in range(mid_dim):
            if i % 2 == 0:
                lora_down.weight[i, i % fan_in] = init_scale
            else:
                lora_down.weight[i, -(i % fan_in) - 1] = -init_scale

    # Initialize lora_up with a complementary asymmetric pattern
    with torch.no_grad():
        lora_up.weight.zero_()
        for i in range(fan_out):
            if i % 2 == 0:
                lora_up.weight[i, i % mid_dim] = init_scale
            else:
                lora_up.weight[i, -(i % mid_dim) - 1] = -init_scale


def lora_target_plan(root_module, target_replace_modules, *, prefix=LORA_PREFIX_UNET,
                     delimiter="_", train_method="full"):
    """Compatibility discovery; new integrations supply explicit named targets."""
    return _legacy_target_plan(root_module, target_replace_modules, prefix=prefix,
                               delimiter=delimiter, train_method=train_method)


def _validate_target_plan(root, target_plan, network):
    """Materialize and check all explicit targets before allocating adapters."""
    plan = tuple(target_plan)
    if not plan:
        raise ValueError("target_plan must contain at least one named host module")
    owned = {id(module) for module in root.modules()}
    names, modules = set(), set()
    for entry in plan:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("target_plan entries must be (export_name, host_module) pairs")
        name, module = entry
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("target_plan export names must be nonempty strings without '.'")
        if (name in names or name in {
            "unet_loras", "_activation_guard", "_activation_stack", "_activation_owner",
        } or hasattr(network, name)):
            raise ValueError(f"target_plan export name is duplicated or reserved: {name!r}")
        if id(module) not in owned:
            raise ValueError(f"target_plan host {name!r} does not belong to the supplied root")
        if id(module) in modules:
            raise ValueError(f"target_plan host module is repeated: {name!r}")
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            raise TypeError(f"target_plan host {name!r} must be Linear, Conv1d or Conv2d")
        if isinstance(module, (nn.Conv1d, nn.Conv2d)) and (
            module.groups != 1 or any(value != 1 for value in module.dilation)
            or module.padding_mode != "zeros"
        ):
            raise ValueError(f"target_plan convolution {name!r} requires groups=1, dilation=1 and zero padding mode")
        names.add(name)
        modules.add(id(module))
    return plan


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if isinstance(org_module, nn.Linear):
            kind = "linear"
        elif isinstance(org_module, nn.Conv1d):
            kind = "conv1d"
        elif isinstance(org_module, nn.Conv2d):
            kind = "conv2d"
        else:
            kind = _legacy_module_kind(org_module)

        if kind == "linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif kind == "conv1d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
            self.lora_down = nn.Conv1d(
                in_dim,
                self.lora_dim,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up = nn.Conv1d(self.lora_dim, out_dim, 1, bias=False)

        elif kind == "conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        #lora_init_weights_advanced(self.lora_down, self.lora_up, 5e-2, sparsity=0.9)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.seq_gain = None  # optional 1-D envelope over the sequence axis
        self.seq_gain_mode = "stretch"  # stretch | prefix
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _seq_gain_broadcast(self, x):
        gain = self.seq_gain
        if gain is None:
            return 1.0
        if x.dim() == 3:
            seq_dim, seq_len = 1, x.shape[1]
        elif x.dim() == 2:
            seq_dim, seq_len = 0, x.shape[0]
        else:
            return 1.0
        g = gain.to(device=x.device, dtype=x.dtype).reshape(-1)
        if g.numel() == 1:
            return g
        mode = getattr(self, "seq_gain_mode", "stretch")
        if mode == "prefix":
            if g.numel() >= seq_len:
                g = g[:seq_len]
            else:
                g = torch.cat((g, g[-1].expand(seq_len - g.numel())))
        elif g.numel() != seq_len:
            g = torch.nn.functional.interpolate(
                g.float().view(1, 1, -1),
                size=seq_len,
                mode="linear",
                align_corners=True,
            ).to(dtype=x.dtype).view(-1)
        shape = [1] * x.dim()
        shape[seq_dim] = seq_len
        return g.reshape(shape)

    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        # LoRA may be fp32/cuda while the host module is bf16 or CPU-offloaded.
        weight = self.lora_down.weight
        x_lora = x.to(device=weight.device, dtype=weight.dtype)
        delta = self.lora_up(self.lora_down(x_lora)).to(device=x.device, dtype=x.dtype)
        gain = self.multiplier * self.scale * self._seq_gain_broadcast(x)
        return self.org_forward(x) + delta * gain


class LoRANetwork(nn.Module):
    """Attach LoRA to an explicit sequence of ``(export_name, host_module)``.

    Names are checkpoint module names and are used verbatim, in plan order.
    Targets must belong to ``unet`` (the historical root argument name). The
    legacy selector arguments remain supported when ``target_plan`` is absent.
    """
    @atomic_attachment('unet')
    def __init__(
        self,
        unet: nn.Module,
        rank: int = 4,
        multiplier: float = 1.0,
        delimiter: str = "_",
        alpha: float = 1.0,
        target_replace = DEFAULT_TARGET_REPLACE,
        prefix=LORA_PREFIX_UNET,
        train_method: TRAINING_METHODS = "full",
        *,
        target_plan: Optional[Iterable[tuple[str, nn.Module]]] = None,
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.seq_gain = None
        self.seq_gain_mode = "stretch"
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha

        # LoRAのみ
        self.module = LoRAModule

        if target_plan is None:
            self.unet_loras = self.create_modules(
                prefix, unet, target_replace, delimiter, self.lora_dim,
                self.multiplier, train_method=train_method,
            )
        else:
            if (target_replace is not DEFAULT_TARGET_REPLACE or train_method != "full"
                    or prefix != LORA_PREFIX_UNET or delimiter != "_"):
                raise ValueError("target_plan cannot be combined with legacy selector/naming arguments")
            targets = _validate_target_plan(unet, target_plan, self)
            self.unet_loras = [
                self.module(name, module, self.multiplier, self.lora_dim, self.alpha)
                for name, module in targets
            ]
        #print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        # 適用する
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet


    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        delimiter: str,
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
    ) -> list:
        return [self.module(name, module, multiplier, rank, self.alpha)
                for name, module in lora_target_plan(root_module, target_replace_modules,
                    prefix=prefix, delimiter=delimiter, train_method=train_method)]

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def get_state_dict(self, dtype=None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        return state_dict

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

#         for key in list(state_dict.keys()):
#             if not key.startswith("lora"):
#                 # lora以外除外
#                 del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    def set_lora_slider(self, scale):
        with activation_mutation(self):
            self.lora_scale = scale

    def set_seq_gain(self, gain, mode: str = "stretch"):
        """Per-position envelope for the LoRA delta. None = uniform.

        mode='stretch' interpolates the curve onto the current sequence
        (transformer chunks). mode='prefix' takes the first S values
        (LM tokens, including a full-prefix refresh).
        """
        if mode not in ('stretch', 'prefix'):
            raise ValueError("Sequence gain mode must be 'stretch' or 'prefix'")
        if gain is None:
            tensor = None
        elif torch.is_tensor(gain):
            tensor = gain.detach().float().reshape(-1).cpu()
        else:
            tensor = torch.tensor(list(gain), dtype=torch.float32)
        if tensor is not None and (tensor.numel() == 0 or not torch.isfinite(tensor).all()):
            raise ValueError('Sequence gain must be non-empty and finite in FP32')
        with activation_mutation(self):
            self.seq_gain = tensor
            self.seq_gain_mode = mode
            if not activation_is_entered(self):
                for lora in self.unet_loras:
                    lora.seq_gain = tensor
                    lora.seq_gain_mode = mode

    def __enter__(self):
        push_activation(self, self.unet_loras, ('multiplier', 'seq_gain', 'seq_gain_mode'))
        try:
            for lora in self.unet_loras:
                lora.multiplier = 1.0 * self.lora_scale
                lora.seq_gain = self.seq_gain
                lora.seq_gain_mode = getattr(self, "seq_gain_mode", "stretch")
        except BaseException:
            restore_activation(self)
            raise
        return self

    def __exit__(self, exc_type, exc_value, tb):
        restore_activation(self)


def _is_lora_delta_layer(module) -> bool:
    return any(
        hasattr(module, name)
        for name in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    )


def apply_continuous_lora_scale(module, scale: float) -> int:
    """Multiply LoRA delta by ``scale`` (continuous 0..1).

    PEFT ``PeftModel.set_adapter_scale`` often no-ops or only toggles,
    which made live smile grids at 0.25 / 0.5 / 1.0 byte-identical.
    This writes the layer ``scaling`` dict (and ``lora_scale`` /
    ``multiplier`` when present) the same way PEFT
    ``rescale_adapter_scale`` does: ``scaling[name] = base[name] * scale``.
    ``base`` is snapshotted on first call (the init ``alpha/r``).
    Works on ``get_peft_model`` wrappers and on duck-typed mocks.
    """
    scale = float(scale)
    updated = 0
    for child in module.modules():
        is_lora = _is_lora_delta_layer(child)
        scaling = getattr(child, "scaling", None)
        if is_lora and isinstance(scaling, dict) and scaling:
            base = getattr(child, "_krea_base_scaling", None)
            if not isinstance(base, dict):
                base = {}
                child._krea_base_scaling = base
            # Existing entries may already be scaled. Only a newly attached
            # adapter supplies a new canonical baseline; never recapture peers.
            base.update({key: float(value) for key, value in scaling.items() if key not in base})
            for key in scaling:
                scaling[key] = base[key] * scale
            updated += 1
        if is_lora and hasattr(child, "lora_scale") and not callable(child.lora_scale):
            child.lora_scale = scale
            updated += 1
        if is_lora and hasattr(child, "multiplier") and not callable(child.multiplier):
            child.multiplier = scale
            updated += 1
    return updated


# Public predicate for native integrations; keep the historical private spelling.
is_lora_delta_layer = _is_lora_delta_layer
