#!/usr/bin/env python3
"""Free-run AR diagnostic for LM concept sliders.

planreg measures teacher-forced hidden drift on a *base* composition. The
documented failure mode is different: a prompt-last displacement flips an early
*sampled* token and the open-loop song re-rolls. This probe free-runs the AR
loop (CFG + top-k, same path as endreg pre-roll) at slider scales and reports:

  first_div   frame index where sampled semantic/end token first differs from
              the scale-0 roll (same seed). None if identical for `frames` steps.
  match_rate  fraction of frames whose sampled token equals scale-0's
  kl_tf       mean KL(base || adapted) of CFG-guided next-token dists while
              teacher-forcing the scale-0 frame history (planreg's trajectory)
  pdrift      MSE of frame hiddens vs scale-0 under that same teacher-force
              (the quantity --planreg_weight minimizes)

A working LM slider *should* diverge early on free-run — that is how
arrangement edits happen. Shipped energy-lm-v4 first_divs at frame 0–1 with
match_rate ≈ 0. Use this probe to *confirm the mechanism* (teacher-forced
pdrift falls with --planreg_weight while free-run still re-rolls) and to
compare +/− asymmetry, not as a pass/fail quality gate. Ears + axis probes
remain the acceptance tests.

  $PY scripts/probe_lm_free_run.py \\
      --weights models/dust-lm-v1-planreg03/dust-lm-v1-planreg03_last.safetensors \\
      --prompts_file conceptmod/textsliders/data/prompts-cand-dust-v1.yaml \\
      --scales=-1,1 --frames 40 --seed 7 --device 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.generate_listen import _sidecar  # noqa: E402
from conceptmod.textsliders.lora import LoRANetwork  # noqa: E402
from conceptmod.textsliders.train_lm_slider_music3 import (  # noqa: E402
    DEFAULT_MODEL,
    TARGET_REPLACE,
    _assemble,
    _forward_teacher_forced,
    _load_rows,
    _set_scale,
    _tokenize,
)


def _guided_logits(lm, last_hidden, vocab_mask):
    from diffusers.modular_pipelines.minimax_music3.encoders import _AR_CFG_SCALE, _AR_CFG_TOP_K

    logits = lm.lm_head(last_hidden).float()
    logits = logits.masked_fill(vocab_mask, -float("inf"))
    conditional, unconditional = logits[0:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
    threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
    guided = guided.masked_fill(conditional < threshold, -float("inf"))
    guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
    return guided


def _free_run(lm, depth_decoder, cond_ids, frames_cap: int, seed: int, device):
    """CFG free-run; returns sampled token ids [T], frame embeds [1,T,H] or None, ended."""
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CFG_TOKEN_ID,
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _SEMANTIC_VOCAB_SIZE,
        _embed_audio_frame,
        _generate_depth_codes,
        _sample_top_k,
    )

    shim = SimpleNamespace(
        language_model=lm,
        rvq_depth_decoder=depth_decoder,
        num_codebooks=int(depth_decoder.config.num_codebooks),
        audio_vocab_size=int(depth_decoder.config.audio_vocab_size),
    )
    unconditional_ids = cond_ids.clone()
    unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((cond_ids, unconditional_ids), dim=0)
    generator = torch.Generator(device).manual_seed(seed)

    output = lm.model(inputs_embeds=lm.model.embed_tokens(text_ids), use_cache=True)
    past_key_values = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device=device)
    vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[_AUDIO_END_TOKEN_ID] = False

    tokens: list[int] = []
    frame_embeds: list[torch.Tensor] = []
    ended = False
    while len(tokens) < frames_cap:
        guided = _guided_logits(lm, last_hidden, vocab_mask)
        sampled = _sample_top_k(guided, generator)
        token_id = int(sampled.item())
        tokens.append(token_id)
        if token_id == _AUDIO_END_TOKEN_ID:
            ended = True
            break
        semantic_code = sampled - _AUDIO_CODE_OFFSET
        frame_codes, _ = _generate_depth_codes(shim, last_hidden, semantic_code.repeat(2), generator)
        feedback = _embed_audio_frame(shim, frame_codes)
        frame_embeds.append(feedback[0:1])
        output = lm.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

    frames = torch.cat(frame_embeds, dim=1) if frame_embeds else None
    return tokens, frames, ended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="LM slider safetensors (+ sidecar)")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--scales", default="-1,1")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
    weights = Path(args.weights)
    meta = _sidecar(weights)
    scales = [float(p) for p in args.scales.split(",") if p.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from diffusers import MiniMaxMusic3RVQDepthDecoder

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_dir) / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm.to(device).eval().requires_grad_(False)
    depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        str(Path(args.model_dir) / "rvq_depth_decoder"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    depth_decoder.eval()

    rows, _ = _load_rows(Path(args.prompts_file))
    row = rows[int(args.row)]
    lyrics = str(row.get("lyrics") or "")
    neu_text = _assemble(str(row.get("neutral") or row["target"]), lyrics)
    cond_ids, _mask = _tokenize(tokenizer, neu_text, device)
    prompt_embeds = lm.model.embed_tokens(cond_ids)

    network = LoRANetwork(
        lm,
        rank=int(meta.get("rank", 8)),
        alpha=float(meta.get("alpha", 8.0)),
        multiplier=1.0,
        delimiter=str(meta.get("delimiter") or "-"),
        target_replace=list(meta.get("target_replace") or TARGET_REPLACE),
        prefix=str(meta.get("prefix") or "lora_te"),
        train_method=str(meta.get("train_method") or "full"),
    ).to(device)
    network.load_state_dict(load_file(str(weights), device="cpu"), strict=False)
    for p in network.parameters():
        p.requires_grad_(False)

    # Scale 0 reference free-run.
    _set_scale(network, 0.0)
    base_tokens, base_frames, base_ended = _free_run(
        lm, depth_decoder, cond_ids, int(args.frames), int(args.seed), device
    )
    _last0, base_margins, base_hidden_full = _forward_teacher_forced(lm, prompt_embeds, base_frames)
    base_hidden = base_hidden_full[:, prompt_embeds.shape[1] :].float()

    print(
        f"weights={weights.name}  frames={args.frames}  seed={args.seed}  "
        f"base_len={len(base_tokens)}  ended={base_ended}"
    )
    print(
        f"{'scale':>7} {'first_div':>10} {'match':>7} {'len':>5} {'ended':>5} "
        f"{'pdrift':>8} {'edrift':>8} {'kl_tf':>8}"
    )

    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _SEMANTIC_VOCAB_SIZE,
    )

    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device=device)
    vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[_AUDIO_END_TOKEN_ID] = False

    for scale in scales:
        _set_scale(network, float(scale))
        tokens, frames, ended = _free_run(
            lm, depth_decoder, cond_ids, int(args.frames), int(args.seed), device
        )
        first_div = None
        matched = 0
        n_cmp = min(len(tokens), len(base_tokens))
        for i in range(n_cmp):
            if tokens[i] == base_tokens[i]:
                matched += 1
            elif first_div is None:
                first_div = i
        match_rate = matched / max(n_cmp, 1)

        # Teacher-force base frames under this scale (planreg geometry).
        if base_frames is None:
            pdrift = edrift = kl_tf = 0.0
        else:
            _last, margins, hidden = _forward_teacher_forced(lm, prompt_embeds, base_frames)
            adapted_hidden = hidden[:, prompt_embeds.shape[1] :].float()
            pdrift = float(F.mse_loss(adapted_hidden, base_hidden).detach())
            edrift = float((margins - base_margins).abs().mean().detach())
            # Per-frame KL on stop+semantic band from frame-position hiddens.
            # Use lm_head rows for end + semantic codes only.
            weight = lm.lm_head.weight
            band = weight[
                list(range(_AUDIO_CODE_OFFSET, _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE))
                + [_AUDIO_END_TOKEN_ID]
            ]
            # hidden at positions that predict each next frame: prompt_last .. last-1
            pred_pos = hidden[:, prompt_embeds.shape[1] - 1 : -1]  # [1, T, H]
            base_pred = base_hidden_full[:, prompt_embeds.shape[1] - 1 : -1]
            if pred_pos.shape[1] == 0:
                kl_tf = 0.0
            else:
                logits_a = (pred_pos.float() @ band.T.float()).float()
                logits_b = (base_pred.float() @ band.T.float()).float()
                log_a = F.log_softmax(logits_a, dim=-1)
                p_b = F.softmax(logits_b, dim=-1)
                kl_tf = float((p_b * (p_b.clamp_min(1e-12).log() - log_a)).sum(dim=-1).mean())

        div_s = "—" if first_div is None else str(first_div)
        print(
            f"{scale:+7.2f} {div_s:>10} {match_rate:7.3f} {len(tokens):5d} {str(ended):>5} "
            f"{pdrift:8.5f} {edrift:8.4f} {kl_tf:8.4f}"
        )


if __name__ == "__main__":
    main()
