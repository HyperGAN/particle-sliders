"""Separate, explicitly typed rank-8 acoustic feed-forward adapter format.

This does not widen or modify the frozen language-model game validator.
"""
import math
from pathlib import Path
import sys
from .core import ROOT,read,sha,IntegrityError

STRUCTURE=dict(kind='transformer',rank=8,alpha=8,target_replace=['MiniMaxMusic3TransformerBlock'],
               prefix='lora_unet',delimiter='-',train_method='full',unit_scale=1.)


def checkpoint(path,multiplier=1.):
    if path is None:
        from .core import checkpoint as off
        return off(None,multiplier)
    import json
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0,str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    if not math.isfinite(float(multiplier)):raise IntegrityError('Nonfinite acoustic multiplier')
    path=Path(path).resolve();metadata=read(path.with_suffix('.json'));h=sha(path)
    if h!=metadata.get('weights_sha256') or any(metadata.get(k)!=v for k,v in STRUCTURE.items()):
        raise IntegrityError('Unsupported acoustic adapter format or changed weights')
    prompt=Path(metadata['prompts_file'])
    if not prompt.is_absolute():prompt=ROOT/prompt
    if _artist_name_hit('',json.dumps(metadata)) or _artist_name_hit('',prompt.read_text()):
        raise IntegrityError('Prohibited acoustic training provenance; strip and retrain')
    values=load_file(str(path));expected=set()
    for layer in range(36):
        projections={**{f'attn-{n}':(2048,2048) for n in ('to_q','to_k','to_v','to_out-0')},'ff_in':(16384,2048),'ff_out':(2048,8192)}
        for projection,(out_dim,in_dim) in projections.items():
            name=f'lora_unet-transformer_blocks-{layer}-{projection}'
            keys=[name+s for s in ('.lora_up.weight','.lora_down.weight','.alpha')];expected.update(keys)
            if not set(keys)<=set(values):raise IntegrityError('Missing acoustic feed-forward block tensors')
            up,down,alpha=[values[k] for k in keys]
            if up.shape!=(out_dim,8) or down.shape!=(8,in_dim) or alpha.numel()!=1 or float(alpha)!=8:
                raise IntegrityError('Wrong acoustic block dimensions or alpha')
            if projection.startswith('attn-') and bool(up.count_nonzero()):
                raise IntegrityError('Feed-forward-only experiment requires exactly zero attention up factors')
    if set(values)!=expected or not all(torch.isfinite(v).all() for v in values.values()):raise IntegrityError('Unexpected/nonfinite acoustic tensors')
    return dict(path=str(path),weights_sha256=h,structure={k:metadata[k] for k in STRUCTURE},multiplier=float(multiplier),
                prompt_sha256=sha(prompt),sidecar_sha256=sha(path.with_suffix('.json')),tensor_count=len(values))


def component(path,multiplier=1.):
    candidate=checkpoint(path,multiplier);p=Path(candidate['path'])
    return dict(candidate['structure'],weights=str(p),mtime=p.stat().st_mtime,multiplier=float(multiplier))
