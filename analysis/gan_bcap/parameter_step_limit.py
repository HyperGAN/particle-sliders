#!/usr/bin/env python3
"""Bound actual generator parameter steps in an explicitly signed lyric trial."""
from contextlib import contextmanager
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import torch
from torch.optim.adamw import AdamW

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.gan_bcap import lyric_preservation_experiment as lyric_driver
from analysis.gan_bcap.stability_experiment import game, sha

_lyric_hooks = lyric_driver.lyric_hooks


def bound_step(parameters, before, maximum):
    """Shorten the joint displacement, up to weight rounding; preserve inactive steps exactly."""
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('Maximum parameter-update norm must be finite and positive')
    with torch.no_grad():
        raw2 = sum((p.detach().double()-b.double()).square().sum() for p,b in zip(parameters,before))
        raw = float(raw2.sqrt())
        if not math.isfinite(raw):
            raise FloatingPointError('Non-finite optimizer displacement')
        factor = min(1., maximum / raw) if raw else 1.
        if factor < 1.:
            for p,b in zip(parameters,before):
                p.copy_(b + factor * (p-b))
        actual = float(sum((p.detach().double()-b.double()).square().sum()
                           for p,b in zip(parameters,before)).sqrt())
    return dict(proposed_update_norm=raw, factor=factor, actual_update_norm=actual)


@contextmanager
def limited_optimizer_steps(maximum, telemetry_path):
    """Patch only this process; the restored LoRA optimizer is explicitly marked."""
    path = Path(telemetry_path)
    if path.exists():
        raise ValueError('Use fresh step-limit telemetry')
    path.parent.mkdir(parents=True, exist_ok=True)
    original_restore, original_step = game.restore, AdamW.step
    marked = None

    def restored(*args, **kwargs):
        nonlocal marked
        history = original_restore(*args, **kwargs)
        marked = kwargs['optimizers']['lora']
        marked._bounded_parameters = [p for p in kwargs['modules']['lora'].parameters() if p.requires_grad]
        marked._bounded_step = len(history) + 1
        return history

    def step(optimizer, *args, **kwargs):
        if optimizer is not marked:
            return original_step(optimizer, *args, **kwargs)
        params = optimizer._bounded_parameters
        before = [p.detach().clone() for p in params]
        result = original_step(optimizer, *args, **kwargs)
        record = dict(step=optimizer._bounded_step, **bound_step(params,before,maximum))
        with path.open('a') as handle:
            handle.write(json.dumps(record,allow_nan=False)+'\n')
        optimizer._bounded_step += 1
        return result

    game.restore, AdamW.step = restored, step
    try:
        yield
    finally:
        game.restore, AdamW.step = original_restore, original_step
        if marked is not None:
            marked.step = original_step.__get__(marked, type(marked))
            del marked._bounded_parameters, marked._bounded_step


class StepLimitedHooks:
    def __init__(self, inner, maximum, telemetry_path):
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError('Maximum parameter-update norm must be finite and positive')
        self.inner, self.maximum = inner, maximum
        self.telemetry_path = Path(telemetry_path)
        self.config = inner.config
        self.config['parameter_step_limit'] = dict(maximum=maximum, implementation_sha256=sha(__file__),
            definition='Global L2 norm of the actual trainable LoRA parameter displacement after AdamW, up to parameter-dtype rounding.',
            optimizer_state='Adam moment updates are retained; only the parameter displacement is shortened.',
            limitation='This bounds a step in the current LoRA parameterization, not a policy KL, effective delta-W norm, or audio-quality change.')

    @contextmanager
    def installed(self):
        with limited_optimizer_steps(self.maximum, self.telemetry_path):
            with self.inner.installed():
                yield


def step_limited_hooks(*, maximum, **kwargs):
    inner = _lyric_hooks(**kwargs)
    path = Path(kwargs['telemetry_path'])
    return StepLimitedHooks(inner, maximum, path.with_name(path.stem+'_step_limit.jsonl'))


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--max-update-norm', type=float, required=True)
    args, remaining = p.parse_known_args()
    invocation = list(sys.argv)
    sys.argv = [__file__, *remaining]
    name = remaining[remaining.index('--name')+1]
    folder = ROOT/'models/gan-bcap-repair'/name
    folder.mkdir(exist_ok=True)
    shutil.copyfile(__file__, folder/'parameter_step_limit_source.py')
    (folder/'step_limit_invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    lyric_driver.lyric_hooks = lambda **kwargs: step_limited_hooks(maximum=args.max_update_norm, **kwargs)
    try:
        lyric_driver.main()
    finally:
        lyric_driver.lyric_hooks = _lyric_hooks


if __name__ == '__main__':
    main()
