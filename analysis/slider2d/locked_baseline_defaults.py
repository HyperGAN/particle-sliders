"""Locked Music Arm B baseline — single source of truth (clone salvage).

`LOCKED` pins the ParticleGAN-faithful recipe plus the Music extras that the
2-D diagnostics need. Every train path, CLI default, and formulation arm in
the consolidated salvage derives from here; nothing else may redefine these
knobs.

KEEP (do not change without a new lock note):
  b_cap coeff=1.0, kappa=1.0, arm=b_cap, norm=l2, lazy=1, anneal=none
  + cover_weight=1.5, teacher=faithful_guard_e, FM0, n=12, particle_l2=0.02,
    vicreg_weight=0.05, shared LR 5e-3 (d/prior mults 1.0).

Formulation arms are propose_only and never change production argv, the
Music trainer row, or the live default (`--lm_target v9`).

Salvage note: the n-particles-only arm (`pg_big_particles`, n 12 -> 64)
tied locked on CPU toys (field2d/sheet/gaussian, wall time flat) and is
HOLD as a propose-only delta here — DROPPED as a living recipe (no trainer
row, no CLI flag, no default change). See
`analysis/slider2d/notes/PG_CLONE_SALVAGE_20260916.md`.
"""

from __future__ import annotations

from dataclasses import fields

from analysis.slider2d.adv import AdvConfig, make_grad_regularizer
from analysis.slider2d.gan import DEFAULT_TEACHER

__all__ = [
    "LOCKED",
    "LOCKED_TEACHER",
    "LOCKED_COVER",
    "LOCKED_FM",
    "LOCKED_BCAP",
    "LOCKED_KAPPA",
    "LOCKED_GRAD_ARM",
    "LOCKED_GRAD_NORM",
    "LOCKED_PARTICLE_L2",
    "LOCKED_N_PARTICLES",
    "LOCKED_N_PARTICLES_MAX",
    "LOCKED_STEPS",
    "PRODUCTION_ARGV_DEFAULTS",
    "BUDGET_KEYS",
    "FORMULATION_ARMS",
    "locked_cfg",
    "arm_cfg",
    "assert_only_delta",
    "check_locked_baseline_adv_config",
    "check_grad_regularizer_champion",
    "check_demo_teacher",
    "assert_advconfig_defaults_match_locked",
]

# Full AdvConfig mirror. `test_pg_clone_consolidated` asserts this stays
# identical to `AdvConfig()` field-by-field, so drift fails loudly.
LOCKED: dict = {
    "steps": 1200,
    "lr": 5.0e-3,
    "beta1": 0.0,
    "beta2": 0.99,
    "d_lr_mult": 1.0,
    "prior_lr_mult": 1.0,
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

# run_lm_adv.py CLI defaults. Mirrored here so the consolidated tests can
# prove the salvage never flipped production argv.
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
    "lr": 5.0e-3,
    "d_lr_mult": 1.0,
    "prior_lr_mult": 1.0,
    "grad_arm": "b_cap",
    "target_anneal": "none",
}

# Only these keys may be overridden on locked_cfg: compute budget, not
# formulation. Any formulation key raises instead of silently forking locked.
BUDGET_KEYS = frozenset({"steps", "seed"})

# Propose-only formulation arms: name -> {delta, status, note}.
# pg_big_particles scales the ParticlePrior toward ParticleGAN (~20k
# particles); CPU toys use n=64 as the feasible probe. Verdict HOLD (tie),
# dropped as a living recipe: production stays n=12.
FORMULATION_ARMS: dict = {
    "pg_big_particles": {
        "delta": {"n_particles": 64},
        "feasible_max": 256,
        "status": "propose_only",
        "note": "HOLD tie vs locked; dropped as living recipe, production stays n=12.",
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


# -- Fire #91 scalar pins + fail-closed gates ------------------------------
# Readable names for the locked #94 shape, derived from the single-source
# LOCKED mirror above (no redefined knobs). The fail-closed gate below
# pins bare AdvConfig() dataclass defaults against this shape.

LOCKED_COVER = LOCKED["cover_weight"]  # 1.5
LOCKED_FM = LOCKED["fm_weight"]  # 0.0
LOCKED_BCAP = LOCKED["b_cap"]  # 1.0
LOCKED_KAPPA = LOCKED["kappa"]  # 1.0
LOCKED_GRAD_ARM = LOCKED["grad_arm"]  # b_cap
LOCKED_GRAD_NORM = LOCKED["grad_norm"]  # l2
LOCKED_PARTICLE_L2 = LOCKED["particle_l2"]  # 0.02
LOCKED_N_PARTICLES = LOCKED["n_particles"]  # 12
LOCKED_N_PARTICLES_MAX = FORMULATION_ARMS["pg_big_particles"]["feasible_max"]  # 256
LOCKED_STEPS = LOCKED["steps"]  # 1200


def check_locked_baseline_adv_config(
    cfg: AdvConfig,
    *,
    require_steps: bool = False,
    require_exact_n: bool = False,
) -> list:
    """List mismatches between `cfg` and the locked #94 shape ([] if clean).

    Defaults are lenient (budget keys `steps`/`seed` skipped, `n_particles`
    allowed up to the propose-only feasible max) so arm configs can reuse
    this; the fail-closed gate below passes `require_steps=True,
    require_exact_n=True` for bare `AdvConfig()` defaults.
    """
    bad: list = []
    for key, want in LOCKED.items():
        if key in BUDGET_KEYS and not require_steps:
            continue
        got = getattr(cfg, key, "<missing>")
        if key == "n_particles" and not require_exact_n:
            if got == want or (
                isinstance(got, int) and got <= LOCKED_N_PARTICLES_MAX
            ):
                continue
            bad.append(
                f"n_particles: {got!r} != locked {want!r} "
                f"(feasible max {LOCKED_N_PARTICLES_MAX})"
            )
            continue
        if got != want:
            bad.append(f"{key}: {got!r} != locked {want!r}")
    return bad


def check_grad_regularizer_champion(reg) -> list:
    """List mismatches between `reg` and the locked b_cap champion ([] if clean)."""
    bad: list = []
    want = {
        "arm": LOCKED_GRAD_ARM,
        "coeff": LOCKED_BCAP,
        "kappa": LOCKED_KAPPA,
        "norm": LOCKED_GRAD_NORM,
        "lazy_k": LOCKED["grad_lazy"],
        "target_anneal": LOCKED["target_anneal"],
    }
    for key, value in want.items():
        if getattr(reg, key, "<missing>") != value:
            bad.append(f"reg.{key}: {getattr(reg, key, '<missing>')!r} != locked {value!r}")
    return bad


def check_demo_teacher(teacher: str | None = None) -> list:
    """List mismatches between the demo teacher and locked ([] if clean)."""
    bad: list = []
    got = DEFAULT_TEACHER if teacher is None else teacher
    if got != LOCKED_TEACHER:
        bad.append(f"teacher: {got!r} != locked {LOCKED_TEACHER!r}")
    if LOCKED_TEACHER != "faithful_guard_e":
        bad.append(f"LOCKED_TEACHER: {LOCKED_TEACHER!r} != 'faithful_guard_e'")
    return bad


def assert_advconfig_defaults_match_locked(
    cfg: AdvConfig | None = None,
    *,
    check_teacher: bool = True,
    check_reg: bool = True,
) -> None:
    """Fail-closed: bare ``AdvConfig()`` dataclass defaults match locked #94 shape.

    Analysis / selection drift detector (Fire #91). Does **not** mutate
    ``AdvConfig`` fields or Music trainer argv — propose_only. Prefer this
    named gate when asserting stock dataclass defaults specifically (vs
    ``check_locked_baseline_adv_config`` which also accepts a constructed cfg).
    """
    cfg = cfg if cfg is not None else AdvConfig()
    # Explicit field-by-field against LOCKED_* (readable failure strings).
    bad = check_locked_baseline_adv_config(
        cfg, require_steps=True, require_exact_n=True
    )
    if check_reg:
        bad.extend(check_grad_regularizer_champion(make_grad_regularizer(cfg)))
    if check_teacher:
        bad.extend(check_demo_teacher())
    if bad:
        raise AssertionError(
            "AdvConfig() defaults drifted from locked #94 shape: " + "; ".join(bad)
        )
