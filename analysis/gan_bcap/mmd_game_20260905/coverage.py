"""Freeze positive-scale parent histories before constructing the next objective."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import torch

from conceptmod.textsliders.gan_v2.data import digest, sha


def replace_history(original, frames, *, seed, path, parent_sha256, ended):
    """Retain condition and lyric positions; invalidate all old history targets."""
    if frames is None or frames.ndim != 3 or frames.shape[0] != 1:
        raise ValueError('Expected a nonempty single sampled audio history')
    if frames.shape[1] < 1 or frames.shape[-1] != original['prompt_embeds'].shape[-1]:
        raise ValueError('Invalid sampled history geometry')
    row = copy.deepcopy(original)
    branches, _, width = original['prompt_embeds'].shape
    row.update(frame_embeds=frames.detach().cpu().repeat(branches, 1, 1),
        history_seed=seed, history_cache=str(Path(path).resolve()), history_cache_sha256=sha(path),
        ended=bool(ended), history_origin='frozen scored parent at scale +1',
        history_parent_sha256=parent_sha256,
        fixture=digest([original['prompt_hash'], seed, parent_sha256, sha(path)]))
    # Teacher preparation checks shape parity. These placeholders explicitly
    # invalidate the old targets; they are recomputed before any loss is used.
    target_shape = (branches, original['real'].shape[1]+frames.shape[1], width)
    row['energy_neutral'] = torch.zeros(target_shape, dtype=torch.float32)
    row['energy_target'] = torch.zeros(target_shape, dtype=torch.float32)
    for key in ('end_teacher', 'policy_teacher', 'continuation_teacher'):
        row.pop(key, None)
    return row


@torch.no_grad()
def collect_parent_histories(lm, tokenizer, network, prepared, definitions, device, out, *,
                             parent_weights, frames_cap=250, seed=17):
    """Sampling is completed once with fixed parent weights, then archived.

    No optimizer runs during collection. These are additional conditioning
    histories for paired hidden matching, not independent fake MMD particles.
    """
    from conceptmod.textsliders import train_lm_slider_music3 as legacy
    from diffusers import MiniMaxMusic3RVQDepthDecoder

    out = Path(out)
    out.mkdir()
    parent_hash = sha(parent_weights)
    started = time.monotonic()
    lm.lm_head.to(device)
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        str(legacy.DEFAULT_MODEL/'rvq_depth_decoder'), torch_dtype=torch.bfloat16,
        local_files_only=True).to(device).eval().requires_grad_(False)
    legacy._set_scale(network, 1.)
    result, manifest = [], []
    try:
        for index, (original, definition) in enumerate(zip(prepared, definitions)):
            neutral = legacy._assemble(definition.get('neutral') or definition['target'], definition['lyrics'])
            tokens, _ = legacy._tokenize(tokenizer, neutral, device)
            history_seed = seed+index
            history, ended = legacy._preroll_frames(lm, depth, tokens, frames_cap, history_seed, device)
            if history is None:
                raise RuntimeError('Parent emitted no frames; preserve the failure instead of resampling')
            path = out/f'parent-history-row{index}-seed{history_seed}.pt'
            torch.save(dict(frame_embeds=history.cpu(), ended=bool(ended), seed=history_seed,
                frames_cap=frames_cap, parent_weights=str(Path(parent_weights).resolve()),
                parent_sha256=parent_hash, scale=1., prompt_hash=original['prompt_hash']), path)
            row = replace_history(original, history, seed=history_seed, path=path,
                                  parent_sha256=parent_hash, ended=ended)
            result.append(row)
            record = dict(row=index, seed=history_seed, frames=history.shape[1], ended=bool(ended),
                          path=str(path.resolve()), sha256=sha(path))
            manifest.append(record)
            print('FROZEN_PARENT_HISTORY', json.dumps(record), flush=True)
    finally:
        del depth
        lm.lm_head.to('cpu')
        torch.cuda.empty_cache()
    if len(result) != len(prepared) or len(result) != len(definitions):
        raise ValueError('History collection count differs from training conditions')
    (out/'manifest.json').write_text(json.dumps(dict(parent_sha256=parent_hash, records=manifest,
        seconds=time.monotonic()-started, frozen_before_optimization=True,
        sampling='Existing bf16 CFG/top-k/RVQ feedback sampler, scored parent at +1, no resampling',
        objective_scope='Frozen-history surrogate; no trajectory measure derivative or fake-fake term'),
        indent=2)+'\n')
    return result
