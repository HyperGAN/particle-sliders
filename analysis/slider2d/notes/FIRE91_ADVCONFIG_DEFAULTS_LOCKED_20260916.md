# Fire #91 — AdvConfig() defaults match locked #94 shape (landed on main)

Bare `AdvConfig()` dataclass defaults must match the locked #94 shape
(cover 1.5 · FM 0 · n=12 · particle_l2=0.02 · b_cap=1 · kappa=1 ·
grad_arm=b_cap · grad_norm=l2 · steps=1200 · teacher faithful_guard_e).
Analysis / selection drift detector — does **not** flip Music trainer
production argv, Music `ARM_B`, or the live `--lm_target v9` default.

## What landed

- `analysis/slider2d/locked_baseline_defaults.py`
  - `LOCKED_*` scalar pins derived from the single-source `LOCKED` mirror
    (no redefined knobs).
  - `check_locked_baseline_adv_config(cfg, *, require_steps,
    require_exact_n)` — list of mismatch strings (`[]` if clean).
  - `check_grad_regularizer_champion(reg)` — b_cap champion pin
    (arm/coeff/kappa/norm/lazy/anneal).
  - `check_demo_teacher()` — `DEFAULT_TEACHER == LOCKED_TEACHER ==
    faithful_guard_e` pin.
  - `assert_advconfig_defaults_match_locked()` — named fail-closed gate
    (all exported in `__all__`).
  - Untouched: `LOCKED`, `locked_cfg`, `arm_cfg`, `assert_only_delta`,
    `PRODUCTION_ARGV_DEFAULTS`, `BUDGET_KEYS`, `FORMULATION_ARMS`.
- `tests/test_lm_2d_adv.py`
  - `test_advconfig_defaults_match_locked_baseline_shape` — field-by-field
    bare-`AdvConfig()` vs `LOCKED_*` asserts plus the named gate.

`AdvConfig` dataclass defaults were already correct — the test is the
product (regression lock). No field edits, CPU-only.
