#!/usr/bin/env python3
"""NON-DEFAULT explore: per-row / multi-residual scaffolds for lyric_span_entangle.

Fire #20: shared AdvResidual is a HARD BOUNDARY on heterogeneous row_amps
(multi-row coverage 0/5). Locked recipe unchanged (1200 / c1.5 / faithful_guard_e /
FM0 / n≤12 / b_cap=1). This script is exploratory — does NOT merge into trainer.

Ablation families (CPU):
  A) shared baseline (confirm)
  B) per-row residual heads (independent odd+even per row)
  C) soft row curriculum on SHARED residual (homo→hetero anneal)
  D) row-weighted cover on SHARED (no gate lowering)
  E) shared direction + per-row scale (cheap hybrid)
  F) leftover / close positive-control under per-row scaffold

Also: geometry notes (why one δ cannot equal five a_r).
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.adv import (  # noqa: E402
    AdvConfig,
    EMA,
    Fourier2MLP,
    ParticlePrior,
    cap_penalty,
    delayed_cosine,
    feature_match_loss,
    input_grad,
    rp_d_loss,
    rp_g_loss,
    sample_real_cloud,
    vicreg_loss,
)
from analysis.slider2d.field import cosine  # noqa: E402
from analysis.slider2d.field3d import (  # noqa: E402
    Field3D,
    close_field3d,
    field3d_teacher_points,
    leftover_field3d,
    lyric_span_entangle_field3d,
    score_adv_field3d_exam,
    unused_e_field3d,
)
from analysis.slider2d.gan import AdvResidual, default_cfg  # noqa: E402
from conceptmod.textsliders.slider_targets import leftover_bipolar, lm_hold_dir  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "per_row_residual_explore_20260909.json"
OUT_MD = NOTES / "per_row_residual_explore_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"
LABEL = "NON_DEFAULT_explore"

M1_AMPS = (
    (1.05, 0.55, 0.35),
    (0.60, 1.10, 0.45),
    (0.75, 0.50, 0.90),
    (1.10, 0.85, 0.55),
    (0.90, 0.70, 0.65),
)
M1_SCALES = (0.75, 0.95, 1.05, 1.2, 1.35)
HOMO_AMPS = tuple((1.0, 0.6, 0.45) for _ in range(5))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def cfg(seed: int, **kw) -> AdvConfig:
    base = dict(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        fm_weight=0.0,
        n_particles=12,
        particle_l2=0.02,
    )
    base.update(kw)
    return default_cfg(**base)


def _collect_teachers(field, teacher, leak_dir):
    plus, minus, neus = [], [], []
    for row in range(int(field.rows)):
        t_plus, t_minus = field3d_teacher_points(
            field, row, teacher=teacher, leak_dir=leak_dir
        )
        neu = field.poles(row)[2]
        plus.append(t_plus.flatten())
        minus.append(t_minus.flatten())
        neus.append(neu.flatten())
    return torch.stack(plus), torch.stack(minus), torch.stack(neus)


# ---------------------------------------------------------------------------
# Geometry: why shared residual fails
# ---------------------------------------------------------------------------

def geometry_report(field: Field3D) -> dict:
    """Math/geometry notes for shared-δ impossibility under hetero row_amps."""
    odds = [field.odd(r).detach() for r in range(int(field.rows))]
    # Restrict to R3 axes (û,ĉ,ê) — lyric dims are zero in odd by construction.
    odds3 = [o[:3] for o in odds]
    norms = [float(o.norm()) for o in odds3]
    mean = torch.stack(odds3).mean(0)
    dist_mean = [float((o - mean).norm()) for o in odds3]
    # Pairwise cosine / L2
    pairs = []
    for i in range(len(odds3)):
        for j in range(i + 1, len(odds3)):
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "cos": float(cosine(odds3[i], odds3[j])),
                    "l2": float((odds3[i] - odds3[j]).norm()),
                    "rel_l2": float(
                        (odds3[i] - odds3[j]).norm()
                        / max(odds3[i].norm().item(), odds3[j].norm().item(), 1e-8)
                    ),
                }
            )
    # Best single vector under MSE to all a_r is the mean; residual cover error floor.
    mse_floor = float(sum((o - mean).pow(2).sum().item() for o in odds3) / len(odds3))
    # Rel err if δ = mean: ||δ - a_r|| / ||a_r||
    rel_errs = [
        float((o - mean).norm() / o.norm().clamp_min(1e-8)) for o in odds3
    ]
    covered_if_mean = sum(1 for e in rel_errs if e <= 0.20)
    # Linear independence: Gram rank of stacked odds3
    G = torch.stack(odds3)  # (R,3)
    # SVD of G^T G
    svals = torch.linalg.svdvals(G).tolist()
    note = (
        "Shared AdvResidual learns one δ(+1)=w_odd (+ even). Cover wants "
        "δ ≈ a_r = odd(row) for EVERY row simultaneously. When row_amps differ, "
        "{a_r} are distinct vectors in span{û,ĉ,ê}; the MSE-optimal single δ is "
        "their mean, and per-row relative error often exceeds the 0.20 cover gate. "
        "Curriculum / cover reweight cannot create degrees of freedom that do not "
        "exist. Per-row heads give each row its own w_odd_r ≈ a_r."
    )
    return {
        "rows": int(field.rows),
        "amps": [list(a) for a in (field.row_amps or ())],
        "scales": list(field.row_scales[: field.rows]),
        "odd_norms_r3": norms,
        "pairwise": pairs,
        "min_pairwise_cos": min(p["cos"] for p in pairs) if pairs else 1.0,
        "max_pairwise_rel_l2": max(p["rel_l2"] for p in pairs) if pairs else 0.0,
        "mse_floor_to_mean": round(mse_floor, 6),
        "rel_err_if_delta_eq_mean": [round(e, 4) for e in rel_errs],
        "rows_covered_if_delta_eq_mean": covered_if_mean,
        "rows_total": int(field.rows),
        "singular_values_odds": [round(s, 4) for s in svals],
        "math_note": note,
    }


# ---------------------------------------------------------------------------
# Scaffold adapters
# ---------------------------------------------------------------------------

@dataclass
class MultiResidual:
    """Per-row odd+even heads. delta(scale, row=r) uses head r."""

    heads: list[AdvResidual]

    def delta(self, scale: float, row: int = 0) -> torch.Tensor:
        return self.heads[int(row)].delta(scale)

    def parameters(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for h in self.heads:
            out.extend(h.parameters())
        return out

    def mean_residual(self) -> AdvResidual:
        wo = torch.stack([h.w_odd.detach() for h in self.heads]).mean(0)
        we = torch.stack([h.w_even.detach() for h in self.heads]).mean(0)
        return AdvResidual(wo.clone(), we.clone())

    def snapshot(self) -> "MultiResidual":
        return MultiResidual([h.snapshot() for h in self.heads])


@dataclass
class ScaledSharedResidual:
    """Shared direction + per-row positive scale on odd (even shared)."""

    w_odd: torch.Tensor
    w_even: torch.Tensor
    row_scales: torch.Tensor  # (R,) unconstrained → softplus

    def delta(self, scale: float, row: int = 0) -> torch.Tensor:
        s = torch.nn.functional.softplus(self.row_scales[int(row)]) + 1e-3
        return float(scale) * s * self.w_odd + abs(float(scale)) * self.w_even

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_odd, self.w_even, self.row_scales]

    def snapshot(self) -> "ScaledSharedResidual":
        return ScaledSharedResidual(
            self.w_odd.detach().clone(),
            self.w_even.detach().clone(),
            self.row_scales.detach().clone(),
        )

    def as_shared_proxy(self) -> AdvResidual:
        # Use mean softplus scale * w_odd for leftover readout
        s = torch.nn.functional.softplus(self.row_scales).mean()
        return AdvResidual(
            (s * self.w_odd).detach().clone(), self.w_even.detach().clone()
        )


def fit_scaffold(
    field: Field3D,
    *,
    mode: str,
    teacher: str = TEACHER,
    cfg: AdvConfig | None = None,
    curriculum_end_amps=None,
    row_cover_weights: list[float] | None = None,
    coupling_weight: float = 0.0,
) -> tuple[object, dict]:
    """Fit residual under scaffold mode.

    Modes:
      shared | per_row | scaled_shared | curriculum_shared | weighted_cover_shared
    coupling_weight: for per_row, penalize (1-cos) between heads in R3 (shared identity).
    """
    cfg = cfg or AdvConfig()
    leak_dir = field.declared_e()
    torch.manual_seed(int(cfg.seed))
    dim = int(field.dim)
    n_rows = int(field.rows)

    # Optional curriculum: interpolate amps from HOMO → end over steps
    use_curriculum = mode == "curriculum_shared"
    end_amps = curriculum_end_amps or field.row_amps

    if mode in ("shared", "curriculum_shared", "weighted_cover_shared"):
        residual: object = AdvResidual(
            torch.zeros(dim, requires_grad=True),
            torch.zeros(dim, requires_grad=True),
        )
        g_params = residual.parameters()
    elif mode == "per_row":
        residual = MultiResidual(
            [
                AdvResidual(
                    torch.zeros(dim, requires_grad=True),
                    torch.zeros(dim, requires_grad=True),
                )
                for _ in range(n_rows)
            ]
        )
        g_params = residual.parameters()
    elif mode == "scaled_shared":
        residual = ScaledSharedResidual(
            torch.zeros(dim, requires_grad=True),
            torch.zeros(dim, requires_grad=True),
            torch.zeros(n_rows, requires_grad=True),
        )
        g_params = residual.parameters()
    else:
        raise ValueError(mode)

    prior_p = ParticlePrior(cfg.n_particles, dim)
    prior_m = ParticlePrior(cfg.n_particles, dim)
    critic = Fourier2MLP(
        dim, n_rand=cfg.critic_n_rand, hidden=cfg.critic_hidden, seed=cfg.seed
    )
    g_params = g_params + list(prior_p.parameters()) + list(prior_m.parameters())
    opt_g = torch.optim.Adam(g_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(
        residual.parameters()
        if not isinstance(residual, MultiResidual)
        else residual.parameters(),
        decay=cfg.ema,
    )

    # Teachers from (possibly curriculum) field snapshot at start; curriculum
    # rebuilds teachers each step from interpolated amps.
    def teachers_for_step(step: int):
        if use_curriculum and end_amps is not None:
            t = step / max(int(cfg.steps) - 1, 1)
            # smoothstep
            t = t * t * (3 - 2 * t)
            amps = []
            for r in range(n_rows):
                h = HOMO_AMPS[r]
                e = end_amps[r]
                amps.append(
                    tuple((1 - t) * h[k] + t * e[k] for k in range(3))
                )
            f = Field3D(
                kind=field.kind,
                rows=field.rows,
                row_scales=field.row_scales,
                row_amps=tuple(amps),
                slider=field.slider,
                content=field.content,
                leak=field.leak,
                e_on_u=field.e_on_u,
                e_on_content=field.e_on_content,
                e_unused=field.e_unused,
            )
            ld = f.declared_e()
            return _collect_teachers(f, teacher, ld), f, ld
        return _collect_teachers(field, teacher, leak_dir), field, leak_dir

    (poles_p, poles_m, neus), field_now, leak_now = teachers_for_step(0)
    half = max(1, int(cfg.batch) // 2)
    weights = row_cover_weights
    if weights is None:
        weights = [1.0] * n_rows
    assert len(weights) == n_rows

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step, total=cfg.steps, delay=cfg.delay, min_ratio=cfg.min_lr_ratio
        )
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group["lr"] = float(cfg.lr) * scale

    def delta_for(row: int, scale: float) -> torch.Tensor:
        if isinstance(residual, AdvResidual):
            return residual.delta(scale)
        return residual.delta(scale, row=row)  # type: ignore[arg-type]

    def fake_batch():
        idx_p = torch.randint(0, neus.shape[0], (half,))
        idx_m = torch.randint(0, neus.shape[0], (half,))
        # Per-row: each sample uses its row's residual head
        fake_p_list, fake_m_list = [], []
        for i in range(half):
            r_p = int(idx_p[i])
            r_m = int(idx_m[i])
            fake_p_list.append(
                neus[r_p]
                + delta_for(r_p, 1.0)
                + prior_p.sample(1, jitter=cfg.particle_jitter)[0]
            )
            fake_m_list.append(
                neus[r_m]
                + delta_for(r_m, -1.0)
                + prior_m.sample(1, jitter=cfg.particle_jitter)[0]
            )
        return torch.stack(fake_p_list), torch.stack(fake_m_list)

    logs = {"d": [], "g": [], "cap": []}
    for step in range(int(cfg.steps)):
        set_lr(step)
        if use_curriculum and (step % 50 == 0 or step + 1 == cfg.steps):
            (poles_p, poles_m, neus), field_now, leak_now = teachers_for_step(step)

        real_p = sample_real_cloud(
            poles_p, neus, n=half, cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac, end_margin=cfg.end_margin,
        )
        real_m = sample_real_cloud(
            poles_m, neus, n=half, cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac, end_margin=cfg.end_margin,
        )
        real = torch.cat([real_p, real_m], dim=0)

        for _ in range(int(cfg.d_steps)):
            fake_p, fake_m = fake_batch()
            fake = torch.cat([fake_p, fake_m], dim=0).detach()
            real_g = real.detach().requires_grad_(True)
            fake_g = fake.detach().requires_grad_(True)
            d_real = critic(real_g)
            d_fake = critic(fake_g)
            cap = cap_penalty(
                input_grad(critic, real_g),
                input_grad(critic, fake_g),
                coeff=cfg.b_cap,
            )
            d_loss = rp_d_loss(d_real, d_fake) + cap
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        fake_p, fake_m = fake_batch()
        fake = torch.cat([fake_p, fake_m], dim=0)
        d_real = critic(real.detach())
        d_fake = critic(fake)
        g_adv = rp_g_loss(d_real, d_fake)
        g_extra = fake.new_zeros(())
        if float(cfg.fm_weight) > 0.0:
            g_extra = g_extra + float(cfg.fm_weight) * feature_match_loss(
                critic.hidden(real.detach()),
                critic.hidden(fake),
                normalize=cfg.fm_normalize,
            )
        parts = torch.cat([prior_p.particles, prior_m.particles], dim=0)
        if float(cfg.vicreg_weight) > 0.0:
            g_extra = g_extra + float(cfg.vicreg_weight) * vicreg_loss(parts)
        if float(cfg.particle_l2) > 0.0:
            g_extra = g_extra + float(cfg.particle_l2) * parts.pow(2).mean()
        if float(cfg.cover_weight) > 0.0:
            cover = fake.new_zeros(())
            wsum = 0.0
            for i in range(neus.shape[0]):
                w = float(weights[i])
                cover = cover + w * (
                    neus[i] + delta_for(i, 1.0) - poles_p[i]
                ).pow(2).mean()
                cover = cover + w * (
                    neus[i] + delta_for(i, -1.0) - poles_m[i]
                ).pow(2).mean()
                wsum += w
            g_extra = g_extra + float(cfg.cover_weight) * cover / max(wsum, 1e-8)
        if float(coupling_weight) > 0.0 and isinstance(residual, MultiResidual):
            # Soft shared-identity: heads should stay acute in ûĉê subspace.
            couple = fake.new_zeros(())
            n_h = len(residual.heads)
            for i in range(n_h):
                for j in range(i + 1, n_h):
                    wi = residual.heads[i].w_odd[:3]
                    wj = residual.heads[j].w_odd[:3]
                    cos_ij = torch.nn.functional.cosine_similarity(
                        wi.unsqueeze(0), wj.unsqueeze(0)
                    ).squeeze()
                    couple = couple + (1.0 - cos_ij)
            n_pairs = n_h * (n_h - 1) / 2.0
            g_extra = g_extra + float(coupling_weight) * couple / max(n_pairs, 1.0)
        g_loss = g_adv + g_extra
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema.update(residual.parameters())

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == cfg.steps:
            logs["d"].append(float(d_loss.detach()))
            logs["g"].append(float(g_loss.detach()))
            logs["cap"].append(float(cap.detach()))

    ema.copy_to(residual.parameters())
    # Final teachers = end field (hetero) for curriculum
    if use_curriculum:
        (poles_p, poles_m, neus), field_now, leak_now = teachers_for_step(
            int(cfg.steps) - 1
        )
    else:
        field_now, leak_now = field, leak_dir

    snap = residual.snapshot() if hasattr(residual, "snapshot") else residual
    stats = {
        "mode": mode,
        "teacher": teacher,
        "b_cap": float(cfg.b_cap),
        "steps": int(cfg.steps),
        "cover_weight": float(cfg.cover_weight),
        "d_loss": logs["d"][-1] if logs["d"] else None,
        "g_loss": logs["g"][-1] if logs["g"] else None,
        "label": LABEL,
    }
    return snap, stats, field_now, leak_now  # type: ignore[return-value]


def _row_coverage_scaffold(residual, field, teacher, leak_dir) -> list[dict]:
    out = []
    for r in range(int(field.rows)):
        t_plus, t_minus = field3d_teacher_points(
            field, r, teacher=teacher, leak_dir=leak_dir
        )
        neu = field.poles(r)[2]
        if isinstance(residual, AdvResidual):
            pred_p = neu + residual.delta(1.0)
            pred_m = neu + residual.delta(-1.0)
        else:
            pred_p = neu + residual.delta(1.0, row=r)
            pred_m = neu + residual.delta(-1.0, row=r)
        err_p = float((pred_p - t_plus).norm() / t_plus.norm().clamp_min(1e-8))
        err_m = float((pred_m - t_minus).norm() / t_minus.norm().clamp_min(1e-8))
        out.append(
            {
                "row": int(r),
                "pole_rel_err_plus": err_p,
                "pole_rel_err_minus": err_m,
                "covered": bool(err_p <= 0.20 and err_m <= 0.20),
            }
        )
    return out


def _leftover_metrics_from_delta(field, d_plus, d_minus):
    a = field.odd(0)
    leftover = leftover_bipolar(d_plus, d_minus)
    on_u = float(d_plus @ field.short_u())
    on_c = float(d_plus @ field.content_dir())
    on_e = float(d_plus @ field.leak_e())
    a_u = float(a @ field.short_u())
    a_c = float(a @ field.content_dir())
    u_kept = on_u / (a_u + 1e-8)
    content_kept = on_c / (a_c + 1e-8) if abs(a_c) > 1e-8 else 0.0
    leak_ratio = abs(on_e) / (abs(on_u) + 1e-8)
    collapse = float(cosine(d_plus, d_minus))
    pair_odd = float(cosine(d_plus, a))
    swing_proxy = max(0.0, min(1.0, 0.5 * (1.0 - collapse) * max(0.0, pair_odd)))
    if abs(a_c) > 1e-8:
        cont_proxy = min(u_kept, content_kept)
    else:
        cont_proxy = u_kept
    return {
        "u_kept": float(u_kept),
        "content_kept": float(content_kept),
        "leak_ratio": float(leak_ratio),
        "pass_u": bool(u_kept >= 0.85),
        "pass_content": bool(content_kept >= 0.75 if abs(a_c) > 1e-8 else True),
        "pass_leak": bool(leak_ratio <= 0.20),
        "collapse": collapse,
        "pair_odd_cos": pair_odd,
        "exam_swing": float(swing_proxy),
        "exam_cont": float(cont_proxy),
        "leak_frac": leftover["leak_frac"],
        "on_u": on_u,
        "on_e": on_e,
    }


def score_scaffold(
    field: Field3D,
    *,
    mode: str,
    seed: int,
    name: str,
    row_cover_weights: list[float] | None = None,
    curriculum_end_amps=None,
    coupling_weight: float = 0.0,
    cfg_kw: dict | None = None,
) -> dict:
    t0 = time.time()
    c = cfg(seed, **(cfg_kw or {}))
    snap, stats, field_eval, leak_dir = fit_scaffold(
        field,
        mode=mode,
        teacher=TEACHER,
        cfg=c,
        curriculum_end_amps=curriculum_end_amps,
        row_cover_weights=row_cover_weights,
        coupling_weight=coupling_weight,
    )
    # Always evaluate on the target field (hetero M1), not mid-curriculum.
    field_eval = field
    leak_dir = field.declared_e()
    row_cov = _row_coverage_scaffold(snap, field_eval, TEACHER, leak_dir)
    rows_covered = sum(1 for rc in row_cov if rc["covered"])

    # Leftover / exam proxies: row0 delta for shared; per-row head0 for multi;
    # also report ALL-rows leftover AND mean-head proxy for multi.
    if isinstance(snap, AdvResidual):
        d_plus = snap.delta(1.0)
        d_minus = snap.delta(-1.0)
        proxy = "shared"
        per_row_leftover_ok = None
    elif isinstance(snap, MultiResidual):
        d_plus = snap.heads[0].delta(1.0)
        d_minus = snap.heads[0].delta(-1.0)
        proxy = "per_row_head0"
        # Each head vs that row's odd — leftover gate per row
        ok_rows = []
        for r, h in enumerate(snap.heads):
            m = _leftover_metrics_from_delta(
                # use a one-row view? use full field but odd(r) via temp
                field_eval, h.delta(1.0), h.delta(-1.0)
            )
            # Recompute u_kept vs THIS row's odd
            a = field_eval.odd(r)
            on_u = float(h.delta(1.0) @ field_eval.short_u())
            on_c = float(h.delta(1.0) @ field_eval.content_dir())
            on_e = float(h.delta(1.0) @ field_eval.leak_e())
            a_u = float(a @ field_eval.short_u())
            a_c = float(a @ field_eval.content_dir())
            u_kept = on_u / (a_u + 1e-8)
            content_kept = on_c / (a_c + 1e-8) if abs(a_c) > 1e-8 else 0.0
            leak_ratio = abs(on_e) / (abs(on_u) + 1e-8)
            ok_rows.append(
                bool(u_kept >= 0.85 and leak_ratio <= 0.20 and (
                    content_kept >= 0.75 if abs(a_c) > 1e-8 else True
                ))
            )
        per_row_leftover_ok = all(ok_rows)
        # Also mean residual proxy
        mean_r = snap.mean_residual()
        mean_m = _leftover_metrics_from_delta(
            field_eval, mean_r.delta(1.0), mean_r.delta(-1.0)
        )
    else:  # ScaledSharedResidual
        d_plus = snap.delta(1.0, row=0)
        d_minus = snap.delta(-1.0, row=0)
        proxy = "scaled_row0"
        per_row_leftover_ok = None
        mean_m = None

    m0 = _leftover_metrics_from_delta(field_eval, d_plus, d_minus)
    # Teacher-aligned continuation: compare delta to (teacher_plus - neu), not raw odd.
    # Per-row faithfully matches teacher; raw-odd content_kept can false-fail when
    # faithful_guard_e strips content/e (shared compromise can accidentally restore content).
    t_plus0, _t_minus0 = field3d_teacher_points(
        field_eval, 0, teacher=TEACHER, leak_dir=leak_dir
    )
    neu0 = field_eval.poles(0)[2]
    teach_a = t_plus0 - neu0
    a_raw = field_eval.odd(0)
    on_u = float(d_plus @ field_eval.short_u())
    on_c = float(d_plus @ field_eval.content_dir())
    ta_u = float(teach_a @ field_eval.short_u())
    ta_c = float(teach_a @ field_eval.content_dir())
    u_vs_teacher = on_u / (ta_u + 1e-8) if abs(ta_u) > 1e-8 else 0.0
    c_vs_teacher = on_c / (ta_c + 1e-8) if abs(ta_c) > 1e-8 else 1.0
    cont_teacher = min(u_vs_teacher, c_vs_teacher) if abs(ta_c) > 1e-8 else u_vs_teacher
    swing_teacher = max(
        0.0,
        min(1.0, 0.5 * (1.0 - float(cosine(d_plus, d_minus))) * max(0.0, float(cosine(d_plus, teach_a)))),
    )

    kind = field_eval.kind
    if kind in (
        "unused_e",
        "lyric_span_entangle",
        "cross_axis_rows",
        "cross_axis_span_sample",
        "axis_u_primary",
        "axis_leak_primary",
    ):
        leftover_ok = bool(m0["pass_leak"] and m0["pass_u"])
    else:
        leftover_ok = bool(m0["pass_u"] and m0["pass_content"])

    multi_ok = rows_covered == int(field_eval.rows)
    # Official-style (raw odd) — can false-fail per-row on content_kept
    pass_cont_raw = m0["exam_cont"] >= 0.85
    pass_swing = m0["exam_swing"] >= 0.60
    exam_pass_raw = bool(pass_cont_raw and pass_swing and leftover_ok and multi_ok)
    # Teacher-aligned scaffold exam (fair for per-row)
    pass_cont_t = cont_teacher >= 0.85
    pass_swing_t = swing_teacher >= 0.60
    exam_pass = bool(pass_cont_t and pass_swing_t and leftover_ok and multi_ok)
    # Fire #20 primary bite: multi-row + leftover û/leak (ignore content artifact)
    bite_cleared = bool(multi_ok and leftover_ok and m0["pass_u"])

    out = {
        "name": name,
        "mode": mode,
        "label": LABEL,
        "seed": seed,
        "cell": field_eval.kind,
        "proxy": proxy,
        "pass": exam_pass,
        "exam_pass": exam_pass,
        "exam_pass_raw_odd": exam_pass_raw,
        "bite_cleared": bite_cleared,
        "exam_score": float(min(cont_teacher, swing_teacher) if swing_teacher > 0 else cont_teacher),
        "exam_score_raw": float(min(m0["exam_cont"], m0["exam_swing"]) if m0["exam_swing"] > 0 else m0["exam_cont"]),
        "u_kept": m0["u_kept"],
        "content_kept": m0["content_kept"],
        "content_kept_vs_teacher": float(c_vs_teacher),
        "cont_teacher": float(cont_teacher),
        "leak_ratio": m0["leak_ratio"],
        "rows_covered": int(rows_covered),
        "rows_total": int(field_eval.rows),
        "pass_multi_row": bool(multi_ok),
        "pass_leftover_gate": bool(leftover_ok),
        "pass_cont": bool(pass_cont_t),
        "pass_cont_raw": bool(pass_cont_raw),
        "pass_swing": bool(pass_swing_t),
        "pass_u": m0["pass_u"],
        "pass_leak": m0["pass_leak"],
        "per_row_leftover_all_ok": per_row_leftover_ok,
        "coupling_weight": float(coupling_weight),
        "row_coverage": row_cov,
        "wall_s": round(time.time() - t0, 2),
        **{k: stats[k] for k in ("d_loss", "g_loss", "steps", "cover_weight")},
    }
    if isinstance(snap, MultiResidual):
        out["mean_proxy_u_kept"] = mean_m["u_kept"]
        out["mean_proxy_leak"] = mean_m["leak_ratio"]
        out["mean_proxy_pass_leak"] = mean_m["pass_leak"]
        # pairwise head cosine in R3
        heads3 = [h.w_odd.detach()[:3] for h in snap.heads]
        cos_pairs = []
        for i in range(len(heads3)):
            for j in range(i + 1, len(heads3)):
                cos_pairs.append(float(cosine(heads3[i], heads3[j])))
        out["head_min_cos"] = min(cos_pairs) if cos_pairs else 1.0
        out["head_mean_cos"] = float(sum(cos_pairs) / len(cos_pairs)) if cos_pairs else 1.0
    if isinstance(snap, ScaledSharedResidual):
        scales = (torch.nn.functional.softplus(snap.row_scales) + 1e-3).detach().tolist()
        out["learned_row_scales"] = [round(s, 4) for s in scales]
    return out


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    npass = sum(1 for r in runs if r["pass"])
    multi = sum(1 for r in runs if r["pass_multi_row"])
    bite = sum(1 for r in runs if r.get("bite_cleared"))
    raw = sum(1 for r in runs if r.get("exam_pass_raw_odd"))
    return {
        "n": n,
        "pass": f"{npass}/{n}",
        "n_pass": npass,
        "multi_row": f"{multi}/{n}",
        "n_multi": multi,
        "bite_cleared": f"{bite}/{n}",
        "n_bite": bite,
        "exam_raw": f"{raw}/{n}",
        "mean_exam": round(sum(r["exam_score"] for r in runs) / n, 4) if n else 0.0,
        "mean_u": round(sum(r["u_kept"] for r in runs) / n, 4) if n else 0.0,
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4) if n else 0.0,
        "mean_rows_cov": round(sum(r["rows_covered"] for r in runs) / n, 2) if n else 0.0,
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "multi_fail_seeds": [r["seed"] for r in runs if not r["pass_multi_row"]],
        "runs": runs,
    }


def grid(name: str, mode: str, field_fn, seeds, **score_kw) -> dict:
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s}", flush=True)
        runs.append(
            score_scaffold(
                field_fn(seed=s) if callable(field_fn) else field_fn,
                mode=mode,
                seed=s,
                name=f"{name}_s{s}",
                **score_kw,
            )
        )
    # field_fn may ignore seed for leftover
    out = summarize(runs)
    out["name"] = name
    out["mode"] = mode
    out["label"] = LABEL
    print(
        f"  {name}: pass={out['pass']} multi={out['multi_row']} bite={out['bite_cleared']} "
        f"exam={out['mean_exam']} leak_max={out['leak_max']} "
        f"rows_cov≈{out['mean_rows_cov']} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def main() -> None:
    t_wall = time.time()
    sha = git_sha()
    print(f"=== per_row_residual_explore {LABEL} @ {sha} ===", flush=True)

    geo_m1 = geometry_report(lyric_span_entangle_field3d())
    geo_homo = geometry_report(
        Field3D(
            kind="homo_scales",
            rows=5,
            row_scales=M1_SCALES,
            row_amps=HOMO_AMPS,
            slider=1.0,
            content=0.7,
            leak=0.55,
            e_on_u=0.05,
            e_on_content=0.0,
            e_unused=0.7,
        )
    )
    print(
        f"  geometry M1: rows_covered_if_mean={geo_m1['rows_covered_if_delta_eq_mean']}/"
        f"{geo_m1['rows_total']} max_rel_l2={geo_m1['max_pairwise_rel_l2']:.3f} "
        f"min_cos={geo_m1['min_pairwise_cos']:.3f}",
        flush=True,
    )
    print(
        f"  geometry HOMO+scales: rows_covered_if_mean="
        f"{geo_homo['rows_covered_if_delta_eq_mean']}/{geo_homo['rows_total']}",
        flush=True,
    )

    cells = []

    # --- A) Shared baseline ---
    print("\n[A] shared baseline", flush=True)
    cells.append(
        grid(
            "A_shared_lyric_m1",
            "shared",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
        )
    )
    cells.append(
        grid(
            "A_shared_leftover",
            "shared",
            lambda seed: leftover_field3d(),
            SEEDS_SMOKE,
        )
    )

    # --- B) Per-row heads ---
    print("\n[B] per-row residual heads", flush=True)
    cells.append(
        grid(
            "B_per_row_lyric_m1",
            "per_row",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
        )
    )
    # Expand to full seeds if smoke multi≥2/3
    b_smoke = cells[-1]
    if b_smoke["n_multi"] >= 2:
        print("  expanding B lyric to full seeds…", flush=True)
        cells.append(
            grid(
                "B_per_row_lyric_m1_full",
                "per_row",
                lambda seed: lyric_span_entangle_field3d(seed=seed),
                SEEDS_FULL,
            )
        )
    # B2: per-row + head coupling (soft shared slider identity)
    cells.append(
        grid(
            "B_per_row_coupled_w0.3_lyric",
            "per_row",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            coupling_weight=0.3,
        )
    )
    cells.append(
        grid(
            "B_per_row_coupled_w1.0_lyric",
            "per_row",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            coupling_weight=1.0,
        )
    )

    cells.append(
        grid(
            "B_per_row_leftover",
            "per_row",
            lambda seed: leftover_field3d(),
            SEEDS_SMOKE,
        )
    )
    cells.append(
        grid(
            "B_per_row_close",
            "per_row",
            lambda seed: close_field3d(),
            SEEDS_SMOKE,
        )
    )
    cells.append(
        grid(
            "B_per_row_unused_e",
            "per_row",
            lambda seed: unused_e_field3d(),
            SEEDS_SMOKE,
        )
    )

    # --- C) Soft curriculum (shared) ---
    print("\n[C] soft row curriculum (shared residual)", flush=True)
    cells.append(
        grid(
            "C_curriculum_shared_lyric",
            "curriculum_shared",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            curriculum_end_amps=M1_AMPS,
        )
    )

    # --- D) Row-weighted cover (shared), no gate change ---
    print("\n[D] row-weighted cover (shared)", flush=True)
    # Weight ∝ ||a_r - mean|| so hard rows get more pin (does not lower 0.20 gate)
    odds = [lyric_span_entangle_field3d().odd(r)[:3] for r in range(5)]
    mean = torch.stack(odds).mean(0)
    dists = [float((o - mean).norm()) + 0.05 for o in odds]
    cells.append(
        grid(
            "D_weighted_cover_shared_lyric",
            "weighted_cover_shared",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            row_cover_weights=dists,
        )
    )
    # Inverse weights (favor easy/center rows) — should NOT fake-pass via gates
    inv = [1.0 / d for d in dists]
    cells.append(
        grid(
            "D_inv_weighted_cover_shared_lyric",
            "weighted_cover_shared",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            row_cover_weights=inv,
        )
    )

    # --- E) Scaled shared (direction + per-row scale) ---
    print("\n[E] scaled shared residual", flush=True)
    cells.append(
        grid(
            "E_scaled_shared_lyric",
            "scaled_shared",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
        )
    )
    # Homo amps + staggered scales: scale head should help
    cells.append(
        grid(
            "E_scaled_shared_homo_amps",
            "scaled_shared",
            lambda seed: Field3D(
                kind="unused_e",  # leftover-style gate
                rows=5,
                row_scales=M1_SCALES,
                row_amps=HOMO_AMPS,
                slider=1.0,
                content=0.7,
                leak=0.55,
                e_on_u=0.05,
                e_on_content=0.0,
                e_unused=0.7,
            ),
            SEEDS_SMOKE,
        )
    )
    cells.append(
        grid(
            "E_shared_homo_amps_control",
            "shared",
            lambda seed: Field3D(
                kind="unused_e",
                rows=5,
                row_scales=M1_SCALES,
                row_amps=HOMO_AMPS,
                slider=1.0,
                content=0.7,
                leak=0.55,
                e_on_u=0.05,
                e_on_content=0.0,
                e_unused=0.7,
            ),
            SEEDS_SMOKE,
        )
    )

    wall = round(time.time() - t_wall, 1)

    # Verdict
    def cell(name):
        for c in cells:
            if c["name"] == name:
                return c
        return None

    a_lyric = cell("A_shared_lyric_m1")
    b_lyric = cell("B_per_row_lyric_m1_full") or cell("B_per_row_lyric_m1")
    b_left = cell("B_per_row_leftover")
    b_close = cell("B_per_row_close")
    b_unused = cell("B_per_row_unused_e")
    c_cur = cell("C_curriculum_shared_lyric")
    d_w = cell("D_weighted_cover_shared_lyric")
    e_sc = cell("E_scaled_shared_lyric")
    e_homo = cell("E_scaled_shared_homo_amps")
    e_ctrl = cell("E_shared_homo_amps_control")

    per_row_promising = bool(
        b_lyric and b_lyric["n_multi"] >= max(2, b_lyric["n"] // 2)
        and (b_left is None or b_left["n_pass"] == b_left["n"])
    )
    # Stronger: full multi + teacher-aligned exam + leftover control
    per_row_yes = bool(
        b_lyric
        and b_lyric["n_multi"] == b_lyric["n"]
        and b_lyric["n_pass"] == b_lyric["n"]
        and b_left
        and b_left["n_pass"] == b_left["n"]
    )
    per_row_bite_yes = bool(
        b_lyric and b_lyric.get("n_bite", 0) == b_lyric["n"]
        and b_left and b_left["n_pass"] == b_left["n"]
    )
    controls_ok = bool(
        (b_left is None or b_left["n_pass"] == b_left["n"])
        and (b_close is None or b_close["n_pass"] >= b_close["n"] - 0)  # require all
        and (cell("A_shared_leftover") and cell("A_shared_leftover")["n_pass"] == cell("A_shared_leftover")["n"])
    )
    # close with per_row — require all smoke pass
    if b_close:
        controls_ok = controls_ok and (b_close["n_pass"] == b_close["n"])
    if b_unused:
        controls_ok = controls_ok and (b_unused["n_pass"] == b_unused["n"])

    shared_still_dead = bool(a_lyric and a_lyric["n_multi"] == 0)
    curriculum_helps = bool(c_cur and c_cur["n_multi"] > (a_lyric["n_multi"] if a_lyric else 0))
    weighted_helps = bool(d_w and d_w["n_multi"] > (a_lyric["n_multi"] if a_lyric else 0))
    scaled_helps_m1 = bool(e_sc and e_sc["n_multi"] > (a_lyric["n_multi"] if a_lyric else 0))
    scaled_helps_homo = bool(
        e_homo and e_ctrl and e_homo["n_multi"] > e_ctrl["n_multi"]
    )

    if per_row_yes and controls_ok:
        verdict = (
            "YES — per-row residual recovers multi-span (teacher-aligned exam) "
            "without breaking leftover/close smoke controls"
        )
    elif per_row_bite_yes and controls_ok:
        verdict = (
            "YES (bite) — per-row clears Fire #20 multi-row+leftover bite; "
            "teacher-aligned exam may still be partial; controls hold"
        )
    elif per_row_promising and controls_ok:
        verdict = (
            "PROMISING — per-row multi-row coverage rises; controls hold; "
            "not yet full exam_pass lock"
        )
    elif per_row_promising and not controls_ok:
        verdict = "MIXED — per-row helps multi-row but control regression risk"
    else:
        verdict = "NO — scaffolds tested do not unlock multi-span under exam gates (or destroy controls)"

    payload = {
        "sha": sha,
        "label": LABEL,
        "locked_recipe": "1200/c1.5/faithful_guard_e/FM0/n<=12/b_cap=1",
        "wall_s": wall,
        "geometry_m1": geo_m1,
        "geometry_homo_scales": geo_homo,
        "cells": [
            {k: v for k, v in c.items() if k != "runs"} | {"runs": c["runs"]}
            for c in cells
        ],
        "verdict": {
            "text": verdict,
            "per_row_promising": per_row_promising,
            "per_row_yes": per_row_yes,
            "per_row_bite_yes": per_row_bite_yes,
            "controls_ok": controls_ok,
            "shared_still_dead": shared_still_dead,
            "curriculum_helps_multi": curriculum_helps,
            "weighted_cover_helps_multi": weighted_helps,
            "scaled_shared_helps_m1": scaled_helps_m1,
            "scaled_shared_helps_homo_scales": scaled_helps_homo,
            "recipe_change": False,
            "merge_to_trainer": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # Markdown
    lines = [
        f"# Per-row / multi-residual explore (NON-DEFAULT) — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. **Locked recipe unchanged.**",
        f"Label: `{LABEL}` — analysis-only; do **not** merge to trainer without multi-seed Music.",
        "",
        "## Why shared AdvResidual fails (geometry)",
        "",
        geo_m1["math_note"],
        "",
        f"- M1 pairwise min cos(a_i,a_j) = **{geo_m1['min_pairwise_cos']:.4f}**, "
        f"max rel L2 = **{geo_m1['max_pairwise_rel_l2']:.4f}**",
        f"- If δ = mean(a_r): rel_err = {geo_m1['rel_err_if_delta_eq_mean']} → "
        f"rows_covered **{geo_m1['rows_covered_if_delta_eq_mean']}/{geo_m1['rows_total']}** "
        f"(0.20 gate). MSE floor={geo_m1['mse_floor_to_mean']}",
        f"- Homo amps + staggered scales: rows_covered_if_mean "
        f"**{geo_homo['rows_covered_if_delta_eq_mean']}/{geo_homo['rows_total']}** "
        f"(scale hetero alone is already a partial bite)",
        "",
        "### Scoring note (content_kept artifact)",
        "",
        "`faithful_guard_e` teacher strips content/e relative to raw `odd(row)`. "
        "Shared δ compromise can *accidentally* restore content_kept vs raw odd "
        "(exam_cont≈1) while covering 0 rows. Per-row matches teacher → lower "
        "raw content_kept but high teacher-aligned cont. Primary Fire #20 metric "
        "here is **multi-row coverage + leftover û/leak** (`bite_cleared`); "
        "`exam_pass` uses teacher-aligned cont/swing.",
        "",
        "## Ablation grid",
        "",
        "| cell | mode | PASS | multi | bite | mean exam | leak max | rows_cov | fail |",
        "|---|---|:---:|:---:|:---:|---:|---:|---:|---|",
    ]
    for c in cells:
        lines.append(
            f"| `{c['name']}` | {c['mode']} | {c['pass']} | {c['multi_row']} | "
            f"{c.get('bite_cleared','?')} | {c['mean_exam']} | {c['leak_max']} | "
            f"{c['mean_rows_cov']} | {c['fail_seeds']} |"
        )
    lines += [
        "",
        "## Family takeaways",
        "",
        f"- **A shared lyric:** multi={a_lyric['multi_row'] if a_lyric else '?'} "
        f"(confirm Fire #20 dead)",
        f"- **B per-row lyric:** multi={b_lyric['multi_row'] if b_lyric else '?'} "
        f"pass={b_lyric['pass'] if b_lyric else '?'} "
        f"bite={b_lyric.get('bite_cleared') if b_lyric else '?'}",
        f"- **B controls:** leftover={b_left['pass'] if b_left else '?'}, "
        f"close={b_close['pass'] if b_close else '?'}, "
        f"unused_e={b_unused['pass'] if b_unused else '?'}",
        f"- **C curriculum shared:** multi={c_cur['multi_row'] if c_cur else '?'} "
        f"(helps? {curriculum_helps})",
        f"- **D weighted cover shared:** multi={d_w['multi_row'] if d_w else '?'} "
        f"(helps? {weighted_helps}) — gates unchanged",
        f"- **E scaled shared M1:** multi={e_sc['multi_row'] if e_sc else '?'} "
        f"(helps? {scaled_helps_m1}); homo+scales "
        f"{e_homo['multi_row'] if e_homo else '?'} vs shared ctrl "
        f"{e_ctrl['multi_row'] if e_ctrl else '?'} (helps? {scaled_helps_homo})",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        f"- per_row_promising={per_row_promising}; per_row_yes={per_row_yes}; "
        f"controls_ok={controls_ok}",
        "- recipe_change=NO; merge_to_trainer=NO",
        "- Next: if YES/PROMISING, multi-seed Music smoke with per-row student "
        "(GPU later); keep shared AdvResidual as locked default",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))

    # Append research log
    log_block = [
        "",
        "## Fire — per-row / multi-residual explore (NON-DEFAULT) (2026-09-09)",
        "",
        f"- Host: box-cpu @ SHA `{sha}` (laptop offline; no Cursor cloud; no Music GPU)",
        f"- Dig: `per_row_residual_explore_20260909.{{py,json,md}}` wall={wall}s",
        f"- Geometry: M1 mean-δ covers {geo_m1['rows_covered_if_delta_eq_mean']}/"
        f"{geo_m1['rows_total']} (floor); shared lyric multi="
        f"{a_lyric['multi_row'] if a_lyric else '?'}",
        f"- Per-row lyric: pass={b_lyric['pass'] if b_lyric else '?'} "
        f"multi={b_lyric['multi_row'] if b_lyric else '?'} "
        f"bite={b_lyric.get('bite_cleared') if b_lyric else '?'}",
        f"- Controls under per-row: leftover={b_left['pass'] if b_left else '?'}, "
        f"close={b_close['pass'] if b_close else '?'}, "
        f"unused_e={b_unused['pass'] if b_unused else '?'}",
        f"- Curriculum/weighted/scaled shared: multi "
        f"{c_cur['multi_row'] if c_cur else '?'}/"
        f"{d_w['multi_row'] if d_w else '?'}/"
        f"{e_sc['multi_row'] if e_sc else '?'} (expect still dead on M1 amp mix)",
        f"- Verdict: {verdict}",
        "- recipe_change=NO; merge_to_trainer=NO; locked defaults untouched",
        "",
    ]
    with LOG.open("a") as f:
        f.write("\n".join(log_block))

    print("\n=== VERDICT ===", flush=True)
    print(verdict, flush=True)
    print(f"wrote {OUT_JSON} {OUT_MD}; appended {LOG}; wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
