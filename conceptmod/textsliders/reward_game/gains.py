"""Construct ordinary rank-8 candidates from bounded depth-group gains.

This is direct adapter-parameter search, not GAN training. Selection must use a
declared training mini-bank before the one fixed winner enters development.
"""
import re


def fold(weights,gains):
    import math
    if len(gains)!=3 or any(not math.isfinite(g) or not 0<=g<=1 for g in gains):
        raise ValueError('Declare three bounded depth gains in [0, 1]')
    result={key:value.clone() for key,value in weights.items()}
    for key,value in result.items():
        if key.endswith('.lora_up.weight'):
            match=re.fullmatch(r'lora_te-model-layers-(\d+)-self_attn-[qkvo]_proj\.lora_up\.weight',key)
            if not match or not 0<=int(match[1])<36:raise ValueError('Unexpected projection topology')
            result[key]=value*gains[int(match[1])//12]
    return result


def training_objective(deltas):
    """Predeclared equal-family objective for a one-case-per-family pilot."""
    import math
    if len(deltas)!=4 or any(not math.isfinite(v) for v in deltas):
        raise ValueError('Four valid paired training CE deltas are required')
    return sum(deltas)/4 - sum(max(0.,-v) for v in deltas)/4
