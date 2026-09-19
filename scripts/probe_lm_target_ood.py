#!/usr/bin/env python3
"""Quantify OOD gap of synthetic symmetric LM targets vs on-manifold pole encodings.

Reports how far `neu ± (pos-neg)/2` sits from `encode(pos/neg)` — the condavg
hypothesis: if the gap is large, training against the synthetic target pushes
the adapter off the caption manifold.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.train_lm_slider_music3 import (  # noqa: E402
    DEFAULT_MODEL,
    _assemble,
    _encode_static,
    _load_rows,
    _tokenize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--common_beta", type=float, default=0.0)
    parser.add_argument("--target_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_dir) / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm.to(device).eval().requires_grad_(False)

    rows, _ = _load_rows(Path(args.prompts_file))
    beta = float(args.common_beta)
    scale = float(args.target_scale)
    print(
        f"{'row':>4} {'gap+/||neu||':>12} {'cos+':>7} {'gap-/||neu||':>12} {'cos-':>7} "
        f"{'||axis||/n':>10} {'raw_cos':>8}"
    )
    for index, row in enumerate(rows):
        lyrics = str(row.get("lyrics") or "")
        texts = {
            "neutral": _assemble(str(row.get("neutral") or row["target"]), lyrics),
            "positive": _assemble(str(row["positive"]), lyrics),
            "negative": _assemble(str(row["negative"]), lyrics),
        }
        toks = {k: _tokenize(tokenizer, v, device) for k, v in texts.items()}
        with torch.no_grad():
            pos = _encode_static(lm, *toks["positive"])
            neg = _encode_static(lm, *toks["negative"])
            neu = _encode_static(lm, *toks["neutral"])
        axis = (pos - neg) / 2.0 * scale
        common = (pos + neg) / 2.0 - neu
        synth_p = neu + axis + beta * common
        synth_n = neu - axis + beta * common
        neu_n = torch.norm(neu).clamp_min(1e-6)
        gap_p = (torch.norm(synth_p - pos) / neu_n).item()
        gap_n = (torch.norm(synth_n - neg) / neu_n).item()
        cos_p = F.cosine_similarity(synth_p - neu, pos - neu, dim=-1).mean().item()
        cos_n = F.cosine_similarity(synth_n - neu, neg - neu, dim=-1).mean().item()
        axis_n = (torch.norm(axis) / neu_n).item()
        raw_cos = F.cosine_similarity(pos - neu, neg - neu, dim=-1).mean().item()
        print(
            f"{index:4d} {gap_p:12.4f} {cos_p:7.3f} {gap_n:12.4f} {cos_n:7.3f} "
            f"{axis_n:10.4f} {raw_cos:8.3f}"
        )


if __name__ == "__main__":
    main()
