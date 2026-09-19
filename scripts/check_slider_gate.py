#!/usr/bin/env python3
"""Acceptance gate for trained slider checkpoints.

Replaces train_v6_lib.sh's `collapse_ok` (which only required collapse < 0 and
therefore passed yearn-lm-v6 while it was rewriting songs: cos+ −0.13,
edrift ~2 nats).

LM gates (sidecar *_last.json):
    symmetric : cos_pos >= 0.85 AND cos_neg >= 0.85 AND collapse <= -0.8
                AND max(edrift_p, edrift_n) <= 0.10
    faithful  : cos_pos >= 0.60 AND cos_neg >= 0.60 AND collapse <= 0
                AND max(edrift) <= 0.10   (dust ears-winner ships at collapse
                -0.15; antisymmetry is not the contract there)
    edge      : + side only (minus is an intentional no-op)

Leakage gate (--leakage_prompts FILE --device N): encodes the neutral caption
under LoRA +/-1 and requires the two displacements to be genuinely opposite
and on-axis (calibrated 2026-08-23 against energy-lm-v7/v4 and gender-lm-v4 at
cos(d+,d-) -0.94..-0.97, axis tracking +/-0.96..0.98; joy-lm-v6 fails at
-0.43 / +0.32 / -0.03 — the minus side was orthogonal to somber):

    cos(d+, d-) <= -0.80 AND cos(d+, axis) >= 0.50 AND |cos(d-, axis)| >= 0.50

TF gates (eval block of *_last.json):
    eval.cos >= 0.45 AND eval.collapse <= -0.5 AND axis_tracking_low not true

Usage:
  python scripts/check_slider_gate.py models/energy-lm-v7            # exit rules
  python scripts/check_slider_gate.py --json models/energy-lm-v7     # machine output
  python scripts/check_slider_gate.py models/joy-lm-v7 \\
      --leakage_prompts conceptmod/textsliders/data/prompts-joy-v7.yaml --device 1
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _last_json(save_dir: Path) -> Path | None:
    lm = save_dir / f"{save_dir.name}_last.json"
    if lm.exists():
        return lm
    cands = [
        p for p in sorted(glob.glob(str(save_dir / "*_last.json")))
        if "comfyui" not in p and "unit_last" not in p
    ]
    return Path(cands[-1]) if cands else None


def check_leakage(save_dir: Path, prompts_file: Path, row: int, device: int) -> tuple[bool, dict]:
    """Hidden-space +/- direction gate. Loads the LM (~16 GB) on `device`."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_pole_leakage as ppl  # noqa: E402

    sidecar = ppl.json_sidecar(_last_json(save_dir) or save_dir)
    edge = "edge" in str(sidecar.get("prompts_file", ""))
    faithful = str(sidecar.get("target_mode")) == "faithful" and not edge

    import torch

    torch_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(ppl.DEFAULT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(model_dir) / "language_model"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm.to(torch_device).eval().requires_grad_(False)

    weights = _last_json(save_dir)
    side = json.loads(weights.read_text(encoding="utf-8"))
    wpath = Path(side.get("weights") or side.get("checkpoint") or "")
    if not wpath.is_absolute():
        wpath = Path(__file__).resolve().parents[1] / wpath
    net = ppl._wrap(
        lm,
        list(sidecar.get("target_replace") or ["Qwen3Attention"]),
        str(sidecar.get("prefix") or "lora_te"),
        int(sidecar.get("rank", 8)),
        float(sidecar.get("alpha", 8.0)),
        torch_device,
        wpath,
        train_method=str(sidecar.get("train_method") or "full"),
    )

    rows, _meta = ppl._load_rows(prompts_file)
    m = ppl.measure(lm, tokenizer, torch_device, net, rows[min(row, len(rows) - 1)])

    if edge:
        # axis is pos - neu here; only the + displacement must track it
        ok = m["cos_dplus_dminus"] <= 0.0 and abs(m["cos_dplus_caption_axis"]) >= 0.50 and m["dplus_norm"] >= 0.03
    elif faithful:
        # KL-pole halves keep some shared motion by design; require on-axis fit
        # of both sides but not full antisymmetry.
        ok = m["cos_dplus_dminus"] <= 0.20 and m["cos_dplus_caption_axis"] >= 0.30 \
            and abs(m["cos_dminus_caption_axis"]) >= 0.30 and m["dplus_norm"] >= 0.03
    else:
        ok = (
            m["cos_dplus_dminus"] <= -0.80
            and m["cos_dplus_caption_axis"] >= 0.50
            and m["cos_dminus_caption_axis"] <= -0.50
            and m["dplus_norm"] >= 0.03
        )
    out = {"leakage": m, "ok": ok}
    return ok, out


def check(save_dir: Path) -> tuple[bool, dict]:
    jf = _last_json(save_dir)
    if jf is None:
        return False, {"error": f"no sidecar found in {save_dir}"}
    d = json.loads(jf.read_text(encoding="utf-8"))
    out: dict = {"checkpoint": str(jf)}

    if d.get("kind") == "language_model":
        last = d.get("last", {})
        cp, cn = float(last.get("cos_pos", 0)), float(last.get("cos_neg", 0))
        col = float(last.get("collapse", 0))
        ed = max(abs(float(last.get("edrift_p", 9))), abs(float(last.get("edrift_n", 9))))
        faithful = str(d.get("target_mode")) == "faithful"
        edge = "edge" in str(d.get("prompts_file", ""))
        if edge:
            # Single-direction control: tgt(-1)=neu by construction, so the
            # minus side is an intentional no-op. Judge the + side only.
            ok = cp >= 0.85 and col <= 0.0 and ed <= 0.10
            out.update(kind="lm", mode="faithful-edge", cos_pos=cp, cos_neg=cn,
                       collapse=col, max_edrift=ed, ok=ok)
        elif faithful:
            # Faithful/KL-pole halves legitimately sit near collapse 0 (the dust
            # ears-winner ships at −0.15); judge them on real-caption fit and
            # plan stability instead of antisymmetry.
            ok = cp >= 0.60 and cn >= 0.60 and col <= 0.0 and ed <= 0.10
            out.update(kind="lm", mode="faithful", cos_pos=cp, cos_neg=cn,
                       collapse=col, max_edrift=ed, ok=ok)
        else:
            ok = cp >= 0.85 and cn >= 0.85 and col <= -0.8 and ed <= 0.10
            out.update(kind="lm", mode="symmetric", cos_pos=cp, cos_neg=cn,
                       collapse=col, max_edrift=ed, ok=ok)
    else:
        ev = d.get("eval", {})
        cos = float(ev.get("cos", 0))
        col = float(ev.get("collapse", 0))
        low = bool(d.get("axis_tracking_low"))
        ok = cos >= 0.45 and col <= -0.5 and not low
        out.update(kind="tf", cos=cos, collapse=col, axis_tracking_low=low, ok=ok)
    return ok, out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--leakage_prompts", type=Path, default=None,
                   help="run the hidden-space +/- leakage gate against this prompt file")
    p.add_argument("--leakage_row", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args(argv)

    all_ok = True
    results = []
    for save in args.dirs:
        ok, info = check(save)
        if args.leakage_prompts is not None and info.get("kind") == "lm":
            try:
                lok, linfo = check_leakage(save, args.leakage_prompts, args.leakage_row, args.device)
            except Exception as exc:  # noqa: BLE001
                lok, linfo = False, {"leakage_error": str(exc)}
            info.update(linfo)
            ok = ok and lok
        all_ok &= ok
        results.append(info)
        if args.json:
            continue
        verdict = "PASS" if ok else "FAIL"
        print(f"[{verdict}] {save.name}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in info.items() if k != 'checkpoint'})}")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
