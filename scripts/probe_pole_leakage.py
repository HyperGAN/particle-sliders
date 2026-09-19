#!/usr/bin/env python3
"""Measure +/- direction leakage of an LM slider checkpoint.

For one prompt row: encode the neutral caption under base, LoRA@+1 and LoRA@-1
(prompt-last hidden state), then report the geometry of the two applied
displacements d+ = h(+1)-h(neu) and d- = h(-1)-h(neu):

  cos(d+, d-)    want strongly NEGATIVE for a bipolar slider (opposite
                 directions); near 0 or positive = the poles share a direction
                 — pushing the fader down moves the music like pushing it up.
                 This is the joy-lm-v6 failure (train collapse -0.20).
  leak_frac      signed projection of d+ onto d-: fraction of the positive
                 displacement that IS the negative displacement. >0 = leak.
  span_ratio     |d+ - d-| / |d+| : how much true bipolar span is realized.

Usage:
  python scripts/probe_pole_leakage.py models/joy-lm-v6 \
      --prompts_file conceptmod/textsliders/data/prompts-joy-v7.yaml \
      --device 1 [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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
from conceptmod.textsliders.generate_gender_stack import _apply, _wrap  # noqa: E402


def json_sidecar(path: Path) -> dict:
    if path.is_dir():
        cands = [c for c in sorted(path.glob("*_last.json")) if "comfyui" not in c.name] or sorted(path.glob("*_last.json"))
        path = cands[0]
    return json.loads(path.read_text(encoding="utf-8"))


def measure(lm, tokenizer, device, net, row: dict) -> dict:
    lyrics = str(row.get("lyrics") or "")
    cells = {
        "neu": _assemble(str(row.get("neutral") or row["target"]), lyrics),
        "pos": _assemble(str(row["positive"]), lyrics),
        "neg": _assemble(str(row["negative"]), lyrics),
    }

    def encode(text: str) -> torch.Tensor:
        ids, mask = _tokenize(tokenizer, text, device)
        return _encode_static(lm, ids, mask)

    with torch.no_grad():
        h_neu = encode(cells["neu"]).float()[0]
        h_pos_cap = encode(cells["pos"]).float()[0]
        h_neg_cap = encode(cells["neg"]).float()[0]
        with _apply((net, +1.0)):
            h_plus = encode(cells["neu"]).float()[0]
        with _apply((net, -1.0)):
            h_minus = encode(cells["neu"]).float()[0]

    d_plus = h_plus - h_neu
    d_minus = h_minus - h_neu
    axis_cap = h_pos_cap - h_neg_cap

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a @ b) / max(a.norm().item() * b.norm().item(), 1e-8))

    return {
        "dplus_norm": float(d_plus.norm() / max(h_neu.norm().item(), 1e-8)),
        "dminus_norm": float(d_minus.norm() / max(h_neu.norm().item(), 1e-8)),
        "cos_dplus_dminus": cos(d_plus, d_minus),
        "leak_frac": float((d_plus @ d_minus) / max(d_minus.norm().item() ** 2, 1e-8)),
        "span_ratio": float(torch.norm(d_plus - d_minus) / max(d_plus.norm().item(), 1e-8)),
        "cos_dplus_caption_axis": cos(d_plus, axis_cap),
        "cos_dminus_caption_axis": cos(d_minus, axis_cap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_dir) / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm.to(device).eval().requires_grad_(False)

    sidecar = json_sidecar(args.checkpoint)
    weights = Path(sidecar.get("weights") or sidecar.get("checkpoint") or "")
    if not weights.is_absolute():
        weights = _REPO_ROOT / weights
    net = _wrap(
        lm,
        list(sidecar.get("target_replace") or ["Qwen3Attention"]),
        str(sidecar.get("prefix") or "lora_te"),
        int(sidecar.get("rank", 8)),
        float(sidecar.get("alpha", 8.0)),
        device,
        weights,
        train_method=str(sidecar.get("train_method") or "full"),
    )

    rows, _meta = _load_rows(Path(args.prompts_file))
    out = measure(lm, tokenizer, device, net, rows[min(args.row, len(rows) - 1)])
    out["checkpoint"] = str(args.checkpoint)

    if args.json:
        print(json.dumps(out))
        return
    print(f"checkpoint            : {out['checkpoint']}")
    print(f"|d+| / |h_neu|        : {out['dplus_norm']:.4f}")
    print(f"|d-| / |h_neu|        : {out['dminus_norm']:.4f}")
    print(f"cos(d+, d-)           : {out['cos_dplus_dminus']:+.4f}   (want <= -0.80; ~0/positive = leak)")
    print(f"leak_frac (d+ on d-)  : {out['leak_frac']:+.4f}   (>0 means + moves like -)")
    print(f"|d+-d-| / |d+|        : {out['span_ratio']:.4f}")
    print(f"cos(d+, caption axis) : {out['cos_dplus_caption_axis']:+.4f}")
    print(f"cos(d-, caption axis) : {out['cos_dminus_caption_axis']:+.4f}")


if __name__ == "__main__":
    main()
