"""Historical name-based LoRA targeting for existing product integrations.

New products provide explicit ``LoRANetwork(target_plan=...)`` entries. This
adapter retains the old class/path filters and export names for callers that
have not adopted a reviewed new target plan. It does not import model packages.
"""
from typing import Literal


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",

]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]


def lora_module_kind(module):
    """Retain duck-typed layer construction for direct historical callers."""
    name = module.__class__.__name__
    if "Linear" in name:
        return "linear"
    if name == "Conv1d":
        return "conv1d"
    if "Conv" in name:
        return "conv2d"
    raise TypeError(f"Unsupported LoRA host type: {name}")


def lora_target_plan(root_module, target_replace_modules, *, prefix=LORA_PREFIX_UNET,
                     delimiter="_", train_method="full"):
    """Read native target names and modules without initialization or attachment."""
    targets = []
    names = []
    wrapped_ids = set()  # dedupe by module identity: overlapping target classes
    # (e.g. the root model + its blocks) must not wrap the same Linear twice
    for name, module in root_module.named_modules():
        if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
            if "attn2" in name or "time_embed" in name:
                continue
        elif train_method == "innoxattn":  # Cross Attention 以外学習
            if "attn2" in name:
                continue
        elif train_method == "selfattn":  # Self Attention のみ学習
            if "attn1" not in name:
                continue
        elif train_method == "xattn" or train_method == "xattn-strict":  # Cross Attention のみ学習
            if "attn2" not in name:
                continue
        elif train_method == "full":  # 全部学習
            pass
        else:
            raise NotImplementedError(
                f"train_method: {train_method} is not implemented."
            )
        if module.__class__.__name__ in target_replace_modules:
            for child_name, child_module in module.named_modules():
                if 'add_' in child_name:
                    continue
                if child_module.__class__.__name__ in [
                    "Linear",
                    "Conv1d",
                    "Conv2d",
                    "LoRACompatibleLinear",
                    "LoRACompatibleConv",
                ]:
                    if train_method == 'xattn-strict':
                        if 'out' in child_name:
                            continue
                    if train_method == 'noxattn-hspace':
                        if 'mid_block' not in name:
                            continue
                    if train_method == 'noxattn-hspace-last':
                        if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                            continue
                    if id(child_module) in wrapped_ids:
                        continue
                    # Skip empty path segments. The root model class is one of
                    # TARGET_REPLACE_FULL and its named_modules() name is "",
                    # which used to yield "lora_unet..transformer_blocks-0-..."
                    # -> a "lora_unet--" double delimiter. Because the identity
                    # dedupe lets the root claim every module first, *every*
                    # --targets full key came out double, disagreeing with the
                    # single-delimiter names --targets attn gives the very same
                    # attention modules -- and with the ComfyUI converter, which
                    # matches "lora_unet-".
                    lora_name = ".".join(p for p in (prefix, name, child_name) if p)
                    lora_name = lora_name.replace(".", delimiter)
                    if lora_name not in names:
                        wrapped_ids.add(id(child_module))
                        targets.append((lora_name, child_module))
                        names.append(lora_name)
    return targets
