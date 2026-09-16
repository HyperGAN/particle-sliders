"""CLONE 5/5 — pg_full_clone: one coherent closest-ParticleGAN recipe (CPU toys).

Propose-only arm. Closes the five intentional drifts vs ParticleGAN
(particle count direction, VICReg weight/form, beta2 0.999, 60% LR hold,
EMA on G+particles) in a single preset while holding the Music adaptations
(leftover-gated teacher, span/end cloud, cover pin) and the live defaults.

CPU only. No Music GPU train. locked_shared (ARM_B) is untouched — these
tests pin that the two rows differ by construction and can never coincide.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.adv import (
    AdvConfig,
    PG_FULL_CLONE,
    PG_FULL_CLONE_LEDGER,
    effective_delay,
    ema_param_groups,
    pg_full_clone_cfg,
    vicreg_loss,
    vicreg_loss_for_cfg,
)
from analysis.slider2d.gan import default_cfg, fit_adv
from analysis.slider2d.sheet import leaky_field
from conceptmod.textsliders.train_lm_slider_music3 import (
    ADV_PRESETS,
    ARM_B,
    PG_FULL_CLONE as TRAINER_PG_ROW,
    PG_FULL_CLONE_PROPOSE_ONLY,
    parse_args,
)


def _clone_argv(*extra: str) -> list[str]:
    return [
        "--prompts_file", "x.yaml",
        "--lm_target", "faithful_guard_e",
        "--adv_preset", "pg_full_clone",
        "--parts", "8",
        "--vicreg_weight", "1.0",
        *extra,
    ]


# -- recipe values -------------------------------------------------------


def test_full_clone_row_matches_intended_deltas():
    cfg = pg_full_clone_cfg()
    for key, want in PG_FULL_CLONE.items():
        assert getattr(cfg, key) == want, key
    assert cfg.beta2 == pytest.approx(0.999)
    assert cfg.delay_frac == pytest.approx(0.6)
    assert cfg.ema_scope == "g_all"
    assert cfg.n_particles == 32
    assert cfg.vicreg_weight == pytest.approx(1.0)
    assert cfg.vicreg_sim_weight == pytest.approx(0.0)
    assert cfg.particle_l2 == pytest.approx(0.0)
    assert cfg.recipe == "pg_full_clone"


def test_ledger_covers_all_five_drifts():
    text = " ".join(" ".join(row) for row in PG_FULL_CLONE_LEDGER)
    for knob in ("particles", "VICReg", "beta2", "LR hold", "EMA"):
        assert knob in text, knob
    assert len(PG_FULL_CLONE_LEDGER) >= 10


def test_locked_toy_defaults_did_not_move():
    cfg = AdvConfig()
    assert cfg.beta2 == pytest.approx(0.99)
    assert cfg.delay_frac is None
    assert cfg.delay == 80
    assert cfg.ema_scope == "residual"
    assert cfg.n_particles == 12
    assert cfg.vicreg_weight == pytest.approx(0.05)
    assert cfg.vicreg_sim_weight == pytest.approx(10.0)
    assert cfg.particle_l2 == pytest.approx(0.02)
    assert cfg.recipe == "toy"


def test_overrides_still_work_for_sweeps():
    cfg = pg_full_clone_cfg(steps=400, seed=1)
    assert cfg.steps == 400
    assert cfg.seed == 1
    assert cfg.beta2 == pytest.approx(0.999)
    assert effective_delay(cfg) == 240


# -- schedule / EMA / VICReg wiring --------------------------------------


def test_effective_delay_matches_sixty_percent_hold():
    assert effective_delay(pg_full_clone_cfg(steps=1200)) == 720
    assert effective_delay(pg_full_clone_cfg(steps=400)) == 240
    assert effective_delay(AdvConfig()) == 80
    with pytest.raises(ValueError):
        effective_delay(pg_full_clone_cfg(delay_frac=0.0))
    with pytest.raises(ValueError):
        effective_delay(pg_full_clone_cfg(delay_frac=1.5))


def test_ema_scope_covers_priors_only_for_full_clone():
    res = [torch.zeros(2, requires_grad=True)]
    priors = [torch.zeros(3, requires_grad=True), torch.zeros(3, requires_grad=True)]
    toy = ema_param_groups(res, priors, AdvConfig())
    assert len(toy) == 1
    clone = ema_param_groups(res, priors, pg_full_clone_cfg())
    assert len(clone) == 3
    with pytest.raises(ValueError):
        ema_param_groups(res, priors, pg_full_clone_cfg(ema_scope="prior_only"))


def test_var_cov_only_vicreg_still_pressures_spread():
    torch.manual_seed(0)
    # Wide + decorrelated: zero sample covariance, per-dim std >> target,
    # so var and cov terms are both free (sim term is off in the clone).
    base = torch.tensor([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0], [0.0, -3.0]])
    spread = base.repeat(16, 1)
    collapsed = torch.zeros(64, 2)
    cfg = pg_full_clone_cfg()
    assert cfg.vicreg_sim_weight == pytest.approx(0.0)
    assert float(vicreg_loss_for_cfg(spread, cfg)) == pytest.approx(0.0, abs=1e-6)
    # Collapse (zero variance) pays the var hinge; spread does not.
    assert float(vicreg_loss_for_cfg(collapsed, cfg)) > float(
        vicreg_loss_for_cfg(spread, cfg)
    )
    # The locked toy keeps its sim term; the clone must not inherit it.
    rand_spread = torch.randn(64, 2) * 2.0
    assert float(vicreg_loss(rand_spread)) != pytest.approx(
        float(vicreg_loss_for_cfg(rand_spread, cfg)), rel=1e-3
    )


def test_fit_adv_runs_both_recipes_and_labels_stats():
    toy_cfg = default_cfg(steps=10, seed=0)
    _, toy_stats = fit_adv(leaky_field(), cfg=toy_cfg)
    assert toy_stats["recipe"] == "toy"
    assert toy_stats["ema_scope"] == "residual"
    assert toy_stats["beta2"] == pytest.approx(0.99)
    clone_cfg = pg_full_clone_cfg(steps=10, seed=0)
    _, clone_stats = fit_adv(leaky_field(), cfg=clone_cfg)
    assert clone_stats["recipe"] == "pg_full_clone"
    assert clone_stats["ema_scope"] == "g_all"
    assert clone_stats["beta2"] == pytest.approx(0.999)
    for stats in (toy_stats, clone_stats):
        assert stats["steps"] == 10
        assert stats["teacher"] == "faithful_guard_e"


# -- trainer: propose-only, never locked_shared ---------------------------


def test_trainer_preset_is_registered_and_flagged_propose_only():
    assert "pg_full_clone" in ADV_PRESETS
    assert PG_FULL_CLONE_PROPOSE_ONLY is True


def test_trainer_full_clone_row_shares_scaffolding_but_not_parts():
    for key in (
        "lm_target", "adv_arch", "adv_norm", "fm_weight",
        "pole_weight", "cover_weight", "adv_reg_kappa", "adv_b_cap",
    ):
        assert TRAINER_PG_ROW[key] == ARM_B[key], key
    assert TRAINER_PG_ROW["parts"] == 8
    assert TRAINER_PG_ROW["vicreg_weight"] == pytest.approx(1.0)
    assert ARM_B["parts"] == 0
    assert ARM_B["vicreg_weight"] == pytest.approx(0.0)


def test_trainer_live_defaults_unchanged():
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.adv_preset == "none"
    assert args.require_arm_b is False


def test_trainer_full_clone_argv_parses_with_exact_row():
    args = parse_args(_clone_argv())
    for key, want in TRAINER_PG_ROW.items():
        got = getattr(args, key)
        if isinstance(want, float):
            assert float(got) == pytest.approx(want), key
        else:
            assert got == want, key


def test_trainer_full_clone_drift_is_refused():
    with pytest.raises(SystemExit):
        parse_args(_clone_argv("--parts", "4"))
    with pytest.raises(SystemExit):
        parse_args(_clone_argv("--vicreg_weight", "0.05"))
    with pytest.raises(SystemExit):
        parse_args(_clone_argv("--cover_weight", "1.5"))
    with pytest.raises(SystemExit):
        parse_args(["--prompts_file", "x.yaml", "--lm_target", "faithful_guard_e",
                    "--adv_preset", "pg_full_clone"])


def test_trainer_full_clone_never_satisfies_require_arm_b():
    with pytest.raises(SystemExit):
        parse_args(_clone_argv("--require_arm_b"))
    # And the clone values fail the Arm B row even without the preset.
    with pytest.raises(SystemExit):
        parse_args(["--prompts_file", "x.yaml", "--lm_target", "faithful_guard_e",
                    "--require_arm_b", "--parts", "8", "--vicreg_weight", "1.0"])


def test_trainer_arm_b_still_passes_unchanged():
    args = parse_args(["--prompts_file", "x.yaml", "--lm_target", "faithful_guard_e",
                       "--adv_preset", "arm_b"])
    for key, want in ARM_B.items():
        got = getattr(args, key)
        if isinstance(want, float):
            assert float(got) == pytest.approx(want), key
        else:
            assert got == want, key
