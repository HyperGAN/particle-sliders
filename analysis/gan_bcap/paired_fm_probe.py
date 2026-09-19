"""CPU comparison of batch-mean and row-paired feature matching.

Uses the production conditional span critic and cap. Four deterministic
teacher rows share a component and each carry their own offset. Span tokens
within one row repeat the same vector; the last token has a distinct target.
This makes exact fake==real attainable despite the critic's set invariance.
No base-model weights or GPU are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conceptmod.textsliders.lm_adv import (
    RowConditionalD, SpanTransformerD, cap_penalty, rp_d_loss, rp_g_loss,
)


def metrics(fake, real):
    f, r = fake.detach(), real.detach()
    row_cos = F.cosine_similarity(f[:, -1], r[:, -1], dim=-1)
    ratios = f[:, -1].norm(dim=-1) / r[:, -1].norm(dim=-1)
    return {
        "nrmse": float((f - r).square().mean().sqrt() / r.square().mean().sqrt()),
        "last_nrmse": float((f[:, -1] - r[:, -1]).norm() / r[:, -1].norm()),
        "last_cos": float(row_cos.mean()),
        "last_cos_min": float(row_cos.min()),
        "magnitude": float(ratios.mean()),
        "magnitude_max": float(ratios.max()),
    }


def run(seed, mode, fm_weight, args):
    torch.manual_seed(seed)
    n, seq, dim = 4, args.seq, args.dim
    shared = torch.randn(1, dim) * 0.8
    unique = torch.randn(n, dim) * 0.6
    span = shared + unique
    last = span + torch.randn(n, dim) * 0.3
    real = torch.cat([span[:, None].expand(n, seq - 1, dim), last[:, None]], dim=1)
    row_ids = torch.arange(n)
    base = SpanTransformerD(
        dim, width=args.width, n_layers=2, n_heads=4,
        in_mode="scaled", readout="mean_last",
    )
    d = RowConditionalD(base, n_conditions=n, feature_dim=2 * args.width)
    d.calibrate_input_scale(real)
    fake = torch.nn.Parameter(torch.zeros_like(real))
    opt_g = torch.optim.Adam([fake], lr=args.g_lr, betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.d_lr, betas=(0.0, 0.999))
    rows = []
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        d.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        pen, _ = cap_penalty(
            d, real, fake.detach(), coeff=1.0, kappa=1.0,
            condition_real=row_ids, condition_fake=row_ids,
        )
        dl = rp_d_loss(d(real, row_ids=row_ids), d(fake.detach(), row_ids=row_ids))
        (dl + pen).backward()
        opt_d.step()
        d.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with torch.no_grad():
            rf = d.features(real)
            rl = d(real, row_ids=row_ids)
        ff = d.features(fake)
        fm = F.mse_loss(ff.mean(0), rf.mean(0)) if mode == "batch" else F.mse_loss(ff, rf)
        adv = rp_g_loss(d(fake, row_ids=row_ids), rl)
        loss = args.adv_weight * adv + fm_weight * fm
        record = step == 1 or step % 50 == 0 or step == args.steps
        if record:
            adv_grad = torch.autograd.grad(args.adv_weight * adv, fake, retain_graph=True)[0]
            fm_grad = torch.autograd.grad(fm_weight * fm, fake, retain_graph=True)[0]
            error = real - fake.detach()
            def alignment(g):
                return float(F.cosine_similarity(-g.flatten(), error.flatten(), dim=0))
            grad_stats = {
                "adv_grad": float(adv_grad.norm()), "fm_grad": float(fm_grad.norm()),
                "adv_fit_alignment": alignment(adv_grad), "fm_fit_alignment": alignment(fm_grad),
            }
        loss.backward()
        grad_finite = bool(torch.isfinite(fake.grad).all())
        opt_g.step()
        if not grad_finite or not bool(torch.isfinite(fake).all()):
            raise RuntimeError(f"nonfinite gradient or fake: {seed}/{mode}/{fm_weight}/{step}")
        if record:
            rows.append({
                "step": step, **metrics(fake, real), **grad_stats,
                "d_loss": float(dl.detach()), "d_pen": float(pen.detach()),
                "g_adv": float(adv.detach()), "fm": float(fm.detach()),
            })
    result = {
        "seed": seed, "mode": mode, "fm_weight": fm_weight,
        "seconds": time.monotonic() - start, "history": rows,
        "final": rows[-1], "fake_equals_real": metrics(real, real),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 101])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--g_lr", type=float, default=0.01)
    p.add_argument("--d_lr", type=float, default=0.00075)
    p.add_argument("--adv_weight", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("paired_fm_results.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    results = []
    for seed in args.seeds:
        for mode, weight in [("batch", 1.0), ("paired", 1.0), ("paired", 10.0)]:
            results.append(run(seed, mode, weight, args))
            args.out.write_text(json.dumps({
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runs": results,
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
