"""Shared-geometry teachers and explicit, disjoint prompt/lyric fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from .. import train_lm_slider_music3 as legacy

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_prompts(rows):
    sys.path.insert(0, str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    if not rows:
        raise ValueError('No prompt rows')
    for i, row in enumerate(rows):
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('positive','lyrics')):
            raise ValueError(f'Incomplete prompt row {i}')
        if not (row.get('neutral') or row.get('target')):
            raise ValueError(f'Missing neutral prompt in row {i}')
        if _artist_name_hit('', json.dumps(row, ensure_ascii=False)):
            raise ValueError(f'Name validation rejected prompt row {i}')


def allowed_logits(lm, hidden):
    from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_END_TOKEN_ID
    semantic = hidden @ legacy._semantic_readout(lm).T
    end = hidden @ lm.lm_head.weight[_AUDIO_END_TOKEN_ID]
    return torch.cat([semantic.float(), end.float().unsqueeze(-1)], -1)


def gather(full, span_mask):
    return legacy._gather_span_last(full, span_mask.to(full.device)).float()


@torch.no_grad()
def prepare_rows(lm, tokenizer, rows, *, model_dir, cache_dir, device,
                 frames=250, seeds=(7,), policy_stride=4):
    """All references use the exact sequence geometry used by their student.

    Extra seeds add teacher-forced histories, not claims of differentiable
    free-running audio training. Those outputs are assessed by the audio gate.
    """
    validate_prompts(rows)
    if frames < 0 or not seeds or len(set(seeds)) != len(seeds) or policy_stride < 1:
        raise ValueError('Invalid history fixture settings')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    depth = None
    result = []
    # No adapter may be active while collecting teachers.
    for index, row in enumerate(rows):
        neutral = legacy._assemble(str(row.get('neutral') or row['target']), row['lyrics'])
        positive = legacy._assemble(row['positive'], row['lyrics'])
        nt, nm = legacy._tokenize(tokenizer, neutral, device)
        pt, pm = legacy._tokenize(tokenizer, positive, device)
        for name, tokens, mask in [('neutral',nt,nm),('positive',pt,pm)]:
            legacy._assert_last_token_is_audio_start(tokens,mask,tokenizer,where=f'v2 row {index} {name}')
        ns, ps = legacy._assert_lyric_span(nt,nm,pt,pm,tokenizer,row['lyrics'],where=f'v2 row {index}')
        ne, pe = lm.model.embed_tokens(nt), lm.model.embed_tokens(pt)
        for seed in seeds:
            history_seed = int(seed) + index
            path = legacy._endreg_cache_path(cache_dir, str(model_dir), neutral, frames, history_seed)
            if path.exists():
                blob = torch.load(path,map_location='cpu',weights_only=True)
            elif frames:
                if depth is None:
                    from diffusers import MiniMaxMusic3RVQDepthDecoder
                    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(str(Path(model_dir)/'rvq_depth_decoder'),
                        torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
                history, ended = legacy._preroll_frames(lm,depth,nt,frames,history_seed,device)
                blob = dict(frame_embeds=None if history is None else history.cpu(),ended=bool(ended))
                temporary = path.with_suffix('.tmp')
                torch.save(blob,temporary); temporary.replace(path)
            else:
                blob = dict(frame_embeds=None,ended=False)
            history = None if blob['frame_embeds'] is None else blob['frame_embeds'].to(device)
            _, neutral_end, nh = legacy._forward_teacher_forced(lm,ne,history)
            _, _, ph = legacy._forward_teacher_forced(lm,pe,history)
            neutral_span = gather(nh[:,:ne.shape[1]],ns)
            positive_span = gather(ph[:,:pe.shape[1]],ps)
            # Match the student's full-tail head GEMM before striding. In bf16,
            # projecting a strided subset can use a different numeric kernel.
            teacher_policy = allowed_logits(lm,ph[:,pe.shape[1]-1:])[0][::policy_stride]
            item = dict(prompt_index=index,history_seed=history_seed,prompt_hash=digest(row),
                lyric_hash=digest(row['lyrics']),fixture=digest([row,history_seed,frames,policy_stride]),
                prompt_embeds=ne.cpu(),frame_embeds=None if history is None else history.cpu(),
                neutral_span=neutral_span.cpu(),condition=neutral_span.cpu(),
                real=(positive_span-neutral_span).cpu(),span_mask=ns.cpu(),
                end_teacher=neutral_end.cpu(),policy_teacher=teacher_policy.cpu(),
                continuation_teacher=ph[:,pe.shape[1]-1::policy_stride].float().cpu(),
                policy_stride=policy_stride,ended=blob['ended'],history_cache=str(path),
                history_cache_sha256=sha(path) if path.exists() else None)
            result.append(item)
            print(f'Prepared prompt {index+1}/{len(rows)}, history seed {history_seed}, '
                  f'{0 if history is None else history.shape[1]} frames',flush=True)
    if depth is not None:
        del depth
        torch.cuda.empty_cache()
    return result


class StudentForward:
    def __init__(self,lm,network,device):
        self.lm,self.network,self.device=lm,network,device

    def __call__(self,row,with_policy):
        legacy._set_scale(self.network,1.)
        prompt=row['prompt_embeds'].to(self.device)
        history=None if row['frame_embeds'] is None else row['frame_embeds'].to(self.device)
        embeds=prompt if history is None else torch.cat([prompt,history],dim=1)
        hidden=self.lm.model(inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2],device=self.device,dtype=torch.long)).last_hidden_state
        fake=gather(hidden[:,:prompt.shape[1]],row['span_mask'])-row['neutral_span'].to(self.device)
        result={'fake':fake}
        if torch.is_grad_enabled():
            logits=allowed_logits(self.lm,hidden[:,prompt.shape[1]-1:])[0]
            result['end']=logits[:,-1]-logits[:,:-1].logsumexp(-1)
            result['continuation']=hidden[:,prompt.shape[1]-1::row['policy_stride']].float()
            if with_policy:
                result['policy']=logits[::row['policy_stride']]
        return result


def fixture_manifest(prepared):
    fields=('prompt_index','history_seed','prompt_hash','lyric_hash','fixture','history_cache_sha256','ended')
    return [{key: row[key] for key in fields} for row in prepared]


def check_disjoint(training, evaluation):
    train_lyrics={row['lyrics'].strip() for row in training}
    overlap=train_lyrics & {row['lyrics'].strip() for row in evaluation}
    if overlap:
        raise ValueError('Evaluation lyrics overlap the training split')
