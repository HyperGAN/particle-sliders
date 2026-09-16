"""Locked Music Arm B baseline — single source of truth (CLONE ARM series).

`LOCKED` pins the ParticleGAN-faithful recipe plus the Music extras that the
2-D diagnostics need. Every train path, CLI default, and formulation arm in
this series derives from here; nothing else may redefine these knobs.

KEEP (do not change without a new lock note):
  b_cap coeff=1.0, kappa=1.0, arm=b_cap, norm=l2, lazy=1, anneal=none
  + cover_weight=1.5, teacher=faithful_guard_e, fm_weight=0.0 (FM off),
    n_particles=12, particle_l2=0.02, vicreg_weight=0.05.

Formulation arms are propose_only: `arm_cfg(name)` returns a locked config
with exactly one sanctioned delta, and `assert_only_delta` proves nothing
else moved. Production argv (`run_lm_adv.py` defaults) and the live Music
trainer default (`--lm_target v9`) are untouched by every arm.
"""

from __future__ import annotations

from dataclasses import fields

from analysis.slider2d.adv import AdvConfig
from analysis.slider2d.gan import DEFAULT_TEACHER

# Full AdvConfig mirror. `test_pg_big_particles` asserts this stays
# identical to `AdvConfig()` field-by-field, so drift fails loudly.
LOCKED: dict = {
    "steps": 1200,
    "lr": 5.0e-3,
    "beta1": 0.0,
    "beta2": 0.99,
    "b_cap": 1.0,
    "kappa": 1.0,
    "grad_arm": "b_cap",
    "grad_norm": "l2",
    "grad_lazy": 1,
    "target_anneal": "none",
    "n_particles": 12,
    "batch": 32,
    "cloud_std": 0.03,
    "particle_jitter": 0.01,
    "span_frac": 0.40,
    "end_margin": 0.60,
    "fm_weight": 0.0,
    "fm_normalize": True,
    "vicreg_weight": 0.05,
    "particle_l2": 0.02,
    "cover_weight": 1.5,
    "d_steps": 1,
    "ema": 0.995,
    "delay": 80,
    "min_lr_ratio": 0.05,
    "critic_hidden": 64,
    "critic_n_rand": 16,
    "seed": 0,
}
LOCKED_TEACHER = DEFAULT_TEACHER  # faithful_guard_e

# run_lm_adv.py CLI defaults. Mirrored here so the arm tests can prove no
# arm (including this series) flipped production argv.
PRODUCTION_ARGV_DEFAULTS: dict = {
    "steps": 1200,
    "exam_steps": 1200,
    "baseline_steps": 400,
    "seed": 0,
    "teacher": LOCKED_TEACHER,
    "b_cap": 1.0,
    "kappa": 1.0,
    "fm_weight": 0.0,
    "cover_weight": 1.5,
}

# Only these keys may be overridden on locked_cfg: compute budget, not
# formulation. Any formulation key raises instead of silently forking locked.
BUDGET_KEYS = frozenset({"steps", "seed"})

# propose_only formulation arms: name -> {delta, status, note}.
# ARM 1/5: scale ParticlePrior toward ParticleGAN (~20k particles). CPU toys
# use a large-but-feasible n; everything else stays locked.
FORMULATION_ARMS: dict = {
    "pg_big_particles": {
        "delta": {"n_particles": 64},
        "feasible_max": 256,
        "status": "propose_only",
        "note": "ParticleGAN-faithful particle cloud scale; production stays n=12.",
    },
}


def locked_cfg(**overrides) -> AdvConfig:
    """The locked baseline. Only `steps`/`seed` (budget) may be overridden."""
    bad = [k for k in overrides if k not in BUDGET_KEYS]
    if bad:
        raise ValueError(
            f"locked_cfg refuses formulation overrides {bad}: "
            "locked knobs are single-sourced; use arm_cfg(name) for arms"
        )
    cfg = AdvConfig(**{**LOCKED, **overrides})
    return cfg


def arm_cfg(name: str, **budget) -> AdvConfig:
    """A propose_only formulation arm: locked plus exactly its named delta."""
    if name not in FORMULATION_ARMS:
        raise ValueError(
            f"unknown formulation arm {name!r} (expected one of {sorted(FORMULATION_ARMS)})"
        )
    arm = FORMULATION_ARMS[name]
    if arm["status"] != "propose_only":
        raise ValueError(f"arm {name!r} is not propose_only")
    bad = [k for k in budget if k not in BUDGET_KEYS]
    if bad:
        raise ValueError(f"arm_cfg budget keys are {sorted(BUDGET_KEYS)}, got {bad}")
    merged = {**LOCKED, **arm["delta"], **budget}
    return AdvConfig(**merged)


def assert_only_delta(cfg: AdvConfig, name: str, *, budget: dict | None = None) -> None:
    """Prove `cfg` differs from locked only by the arm delta (+ budget)."""
    arm = FORMULATION_ARMS[name]
    allowed = set(arm["delta"]) | set((budget or {}))
    locked = AdvConfig(**LOCKED)
    for f in fields(AdvConfig):
        got, want = getattr(cfg, f.name), getattr(locked, f.name)
        if f.name in allowed:
            continue
        assert got == want, f"arm {name!r} moved locked knob {f.name}: {got!r} != {want!r}"
    for key, value in arm["delta"].items():
        assert getattr(cfg, key) == value, f"arm {name!r} delta {key} not applied"
