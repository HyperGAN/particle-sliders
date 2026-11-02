"""Ordinary additive LoRA gains for q/k/v/o projections, independent of depth."""
import math
import re


def fold(weights,gains):
    if len(gains)!=4 or any(not math.isfinite(g) or not 0<=g<=1.25 for g in gains):
        raise ValueError('Declare q/k/v/o gains in [0, 1.25]')
    result={key:value.clone() for key,value in weights.items()}
    for key,value in result.items():
        if key.endswith('.lora_up.weight'):
            match=re.fullmatch(r'lora_te-model-layers-(\d+)-self_attn-([qkvo])_proj\.lora_up\.weight',key)
            if not match or not 0<=int(match[1])<36:raise ValueError('Unexpected projection topology')
            result[key]=value*gains['qkvo'.index(match[2])]
    return result
