"""Create one declared adapter-strength candidate without an optimizer update.

Alpha buffers and their sidecar value change together. The locked renderer still
uses scale +1. Original trainable tensors, full training state and provenance are
retained. This is checkpoint calibration, not an additional training run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file, save_file

from conceptmod.textsliders.gan_v2.data import ROOT, sha, validate_prompts


def calibrated_tensors(weights, sidecar, factor):
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Strength factor must be finite and positive')
    if sidecar.get('unit_scale', 1.) != 1.:
        raise ValueError('Parent must already use unit scale one')
    alpha = float(sidecar['alpha'])
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Parent alpha must be finite and positive')
    output, count = {}, 0
    for key, value in weights.items():
        if key.endswith('.alpha'):
            if value.numel() != 1 or float(value) != alpha:
                raise ValueError('Alpha tensor and sidecar differ')
            output[key] = (value.float()*factor).to(value.dtype)
            if float(output[key]) != alpha*factor:
                raise ValueError('Requested alpha is not exactly representable')
            count += 1
        else:
            output[key] = value.clone()
    if count < 1:
        raise ValueError('No adapter alpha buffers found')
    return output, dict(sidecar, alpha=alpha*factor, unit_scale=1.), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-weights', type=Path, required=True)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--factor', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.out.exists():
        raise ValueError('Use a fresh research directory')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    source = torch.load(args.source_state, map_location='cpu', weights_only=True)
    weights = load_file(str(args.source_weights))
    if weights.keys() != source['network'].keys() or any(not torch.equal(v, source['network'][k]) for k, v in weights.items()):
        raise ValueError('Scored parent weights differ from complete state')
    sidecar = json.loads(args.source_weights.with_suffix('.json').read_text())
    from conceptmod.textsliders import train_lm_slider_music3 as legacy
    prompts = Path(sidecar['prompts_file'])
    rows, _ = legacy._load_rows(prompts)
    validate_prompts(rows)
    tensors, metadata, count = calibrated_tensors(weights, sidecar, args.factor)
    if 'gan_v2' in metadata:
        metadata['parent_gan_training'] = metadata.pop('gan_v2')
    output = args.out/f'{args.out.name}_step{sidecar["steps"]}.safetensors'
    metadata.update(strength_calibration=dict(factor=args.factor,
        parent_weights=str(args.source_weights.resolve()), parent_sha256=sha(args.source_weights),
        optimizer_updates=0, changed='Alpha buffers and sidecar alpha only',
        quality_status='Unvalidated research candidate'), quality_status='unvalidated_research_candidate')
    save_file(tensors, str(output))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    shutil.copyfile(args.source_weights, args.out/'initial.safetensors')
    shutil.copyfile(args.source_weights.with_suffix('.json'), args.out/'initial.json')
    shutil.copyfile(args.source_state, args.out/'parent-state.pt')
    shutil.copyfile(prompts, args.out/'prompts.yaml')
    shutil.copytree(args.source_state.parent/'provenance', args.out/'parent-provenance')
    paths = [Path(__file__).resolve(), ROOT/'conceptmod/textsliders/lora.py',
        ROOT/'conceptmod/textsliders/gan_v2/data.py', ROOT/'conceptmod/textsliders/train_lm_slider_music3.py']
    sources = {}
    for path in paths:
        destination = args.out/'provenance'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[str(path)] = sha(path)
        if sha(destination) != sources[str(path)]:
            raise RuntimeError('Source changed during capture')
    manifest = dict(method='Adapter strength calibration', family='calibration', factor=args.factor,
        parent_weights=str(args.source_weights.resolve()), parent_weights_sha256=sha(args.source_weights),
        parent_state=str(args.source_state.resolve()), parent_state_sha256=sha(args.source_state),
        alpha_before=sidecar['alpha'], alpha_after=metadata['alpha'], alpha_buffer_count=count,
        source_step=sidecar['steps'], attempted_updates=0, training_seconds=0.,
        optimizer='No optimizer instantiated or updated; complete parent state archived separately',
        derivation='Each alpha buffer and sidecar alpha multiplied by the fixed factor; all trainable tensors bitwise retained',
        render_scale=1., prompts_sha256=sha(prompts), sources=sources,
        final_weights=str(output.resolve()), final_weights_sha256=sha(output),
        construction_seconds=time.monotonic()-started)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    torch.save(dict(schema='derived-strength-1', network=tensors, manifest=manifest,
        parent_state_file='parent-state.pt', parent_state_sha256=sha(args.out/'parent-state.pt'),
        optimizer=None, completed_optimizer_updates=0), args.out/'state.pt')
    reloaded = load_file(str(output))
    if any(not torch.equal(value, reloaded[key]) for key, value in tensors.items()):
        raise RuntimeError('Saved candidate tensors changed')
    for key in weights:
        if not key.endswith('.alpha') and not torch.equal(weights[key], reloaded[key]):
            raise RuntimeError('A trainable parent tensor changed')
    completion = dict(status='constructed', final_weights=str(output.resolve()),
        attempted_updates=0, accepted=0, training_seconds=0.,
        total_seconds=time.monotonic()-started, tensor_audit='passed')
    (args.out/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
