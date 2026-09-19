#!/usr/bin/env python3
"""Score an LM slider against on-manifold caption policies, not synthetic targets.

Train collapse/cos measure how well the LoRA hits neu±(pos-neg)/2. That target
is off the caption encodings; chasing it ranked lyric-distorting cells as
winners. This probe instead asks, at the first AR token (prompt-last):

  pole_kl   KL(encode(pos) || LoRA@+1)  and  KL(encode(neg) || LoRA@-1)
            lower = first-token policy matches the real dusty/glossy caption
  base_kl   KL(encode(neu) || LoRA@±1)
            lower = first token stays nearer the in-distribution song
  pm_kl     KL(LoRA@+1 || LoRA@-1)
            higher = the two directions are actually different policies
  cos_f     cosine of (h_lora - neu) to (pos-neu) / (neg-neu)
            the faithful hidden geometry (not the synthetic one)

Slider clips encode the *neutral* caption with the LoRA on, same as generate.

  $PY scripts/probe_lm_policy_ood.py \\
      --prompts_file conceptmod/textsliders/data/prompts-cand-dust-v1.yaml \\
      --weights models/dust-lm-v1-planreg03/dust-lm-v1-planreg03_last.safetensors \\
                 models/dust-lm-v1-faithful-kl-pole03/dust-lm-v1-faithful-kl-pole03_last.safetensors \\
      --device 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    _encode_static,
    _load_rows,
    _semantic_band_logits,
    _set_scale,
    _tokenize,
)


def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum(dim=-1).mean())


def _log_policy(lm, hidden: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = _semantic_band_logits(lm, hidden.unsqueeze(1)) / max(float(temperature), 1e-6)
    return F.log_softmax(logits.float(), dim=-1).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--row", type=int, default=-1, help="row index; -1 = average over all rows (the optimization target)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_dir) / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm.to(device).eval().requires_grad_(False)

    rows, _ = _load_rows(Path(args.prompts_file))
    eval_rows = rows if int(args.row) < 0 else [rows[int(args.row)]]

    def row_states(row: dict):
        lyrics = str(row.get("lyrics") or "")
        texts = {
            "neu": _assemble(str(row.get("neutral") or row["target"]), lyrics),
            "pos": _assemble(str(row["positive"]), lyrics),
            "neg": _assemble(str(row["negative"]), lyrics),
        }
        toks = {k: _tokenize(tokenizer, v, device) for k, v in texts.items()}
        with torch.no_grad():
            h_neu = _encode_static(lm, *toks["neu"])
            h_pos = _encode_static(lm, *toks["pos"])
            h_neg = _encode_static(lm, *toks["neg"])
        log_neu = _log_policy(lm, h_neu, args.temperature)
        log_pos = _log_policy(lm, h_pos, args.temperature)
        log_neg = _log_policy(lm, h_neg, args.temperature)
        axis_p = h_pos - h_neu
        axis_n = h_neg - h_neu
        synth_p = h_neu + 0.5 * (h_pos - h_neg)
        synth_n = h_neu - 0.5 * (h_pos - h_neg)
        return log_neu, log_pos, log_neg, h_neu, axis_p, axis_n, synth_p, synth_n, toks

    n_rows = len(eval_rows)
    print(
        f"{'cell':<28} {'pole_kl':>8} {'base_kl':>8} {'pm_kl':>8} "
        f"{'pm_caps':>8} {'realize':>7} "
        f"{'cos_f+':>7} {'cos_f-':>7} {'cos_s+':>7} {'cos_s-':>7}"
    )
    for weights in args.weights:
        path = Path(weights)
        meta = _sidecar(path)
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
        network.load_state_dict(load_file(str(path), device="cpu"), strict=False)
        for p in network.parameters():
            p.requires_grad_(False)
        unit = float(meta.get("unit_scale") or 1.0)

        acc = {"pole_kl": 0.0, "base_kl": 0.0, "pm_kl": 0.0, "pm_caps": 0.0,
               "cos_fp": 0.0, "cos_fn": 0.0, "cos_sp": 0.0, "cos_sn": 0.0}
        for row in eval_rows:
            log_neu, log_pos, log_neg, h_neu, axis_p, axis_n, synth_p, synth_n, toks = row_states(row)

            def hidden_at(scale: float) -> torch.Tensor:
                _set_scale(network, float(scale) * unit)
                with torch.no_grad():
                    return _encode_static(lm, *toks["neu"])

            h_p = hidden_at(1.0)
            h_m = hidden_at(-1.0)
            log_p = _log_policy(lm, h_p, args.temperature)
            log_m = _log_policy(lm, h_m, args.temperature)
            acc["pole_kl"] += 0.5 * (_kl(log_pos, log_p) + _kl(log_neg, log_m))
            acc["base_kl"] += 0.5 * (_kl(log_neu, log_p) + _kl(log_neu, log_m))
            acc["pm_kl"] += 0.5 * (_kl(log_p, log_m) + _kl(log_m, log_p))
            # The contrast the PAIR itself contains (base-model policy distance
            # between the two real caption encodes). pm_kl / pm_caps = fraction
            # of the true contrast the fader realizes -- comparable across pairs.
            acc["pm_caps"] += 0.5 * (_kl(log_pos, log_neg) + _kl(log_neg, log_pos))
            acc["cos_fp"] += F.cosine_similarity(h_p - h_neu, axis_p, dim=-1).mean().item()
            acc["cos_fn"] += F.cosine_similarity(h_m - h_neu, axis_n, dim=-1).mean().item()
            acc["cos_sp"] += F.cosine_similarity(h_p - h_neu, synth_p - h_neu, dim=-1).mean().item()
            acc["cos_sn"] += F.cosine_similarity(h_m - h_neu, synth_n - h_neu, dim=-1).mean().item()
        pole_kl, base_kl, pm_kl = (acc[k] / n_rows for k in ("pole_kl", "base_kl", "pm_kl"))
        pm_caps = acc["pm_caps"] / n_rows
        cos_fp, cos_fn = acc["cos_fp"] / n_rows, acc["cos_fn"] / n_rows
        cos_sp, cos_sn = acc["cos_sp"] / n_rows, acc["cos_sn"] / n_rows
        _set_scale(network, 0.0)
        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        realize = pm_kl / max(pm_caps, 1e-8)
        print(
            f"{path.parent.name:<28} {pole_kl:8.4f} {base_kl:8.4f} {pm_kl:8.4f} "
            f"{pm_caps:8.4f} {realize:7.3f} "
            f"{cos_fp:7.3f} {cos_fn:7.3f} {cos_sp:7.3f} {cos_sn:7.3f}"
        )


if __name__ == "__main__":
    main()
