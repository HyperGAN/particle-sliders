"""Native Qwen 2.1 sequence layout helpers; no dependency on a slider product."""
import math
import torch


def predict(model, kwargs):
    if isinstance(kwargs, list):
        # 2.1 RoPE ties image positions to prefix length. Per-row forwards preserve native layouts.
        return torch.cat([predict(model, row) for row in kwargs])
    # The native model emits the joint text/image sequence; the pipeline takes its target suffix.
    tokens = math.prod(kwargs["img_shapes"][0][-1])
    return model(**kwargs, return_dict=False)[0][:, -tokens:].float()


def with_encoding(kwargs, encoding):
    tokens = kwargs["hidden_states"].shape[1]
    image_mask = encoding["image_pad_mask"]
    if image_mask.any():
        raise ValueError("image-conditioned training is outside this protocol")
    return {
        **kwargs,
        "encoder_hidden_states": encoding["prompt_embeds"],
        "encoder_hidden_states_mask": encoding["prompt_embeds_mask"],
        "img_mask": torch.cat([image_mask, image_mask.new_ones(image_mask.shape[0], tokens // 4)], dim=1),
    }
