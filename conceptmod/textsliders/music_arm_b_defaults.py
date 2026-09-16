"""Music Arm B single source of truth (Strategy C, architecture-first).

Mirrors the Field3D / Slider-CPU winning config from
``analysis/slider2d`` (KEEP / DR hold_ablation, ``#94`` ParticleGAN-faithful
RpGAN + ``b_cap``) and translates it to Music-LM trainer argv. The live Music
trainer (``conceptmod.textsliders.train_lm_slider_music3``) must READ its Arm B
defaults from this module only — never from hardcoded literals — and must call
:func:`assert_music_arm_b_argv` fail-closed before any Arm B spend.

Field3D -> Music translation (deliberate, not drift):

==================  ==================  ==================  ====================
Knob                Field3D demo        Music Arm B         Why
==================  ==================  ==================  ====================
grad regularizer    ``b_cap`` coeff 1,  same: ``b_cap`` 1,  ParticleGAN-faithful
                    kappa 1, norm l2    kappa 1, norm l2    ``(coeff/2)(E_r+E_f)
                                                relu(||grad D||-k)^2``;
                                                ``analysis/slider2d/adv.py``
                                                ``cap_penalty`` hardcodes
                                                kappa=1, so Music argv pins
                                                ``--adv_reg_kappa 1`` to that.
adv loss            RpGAN logistic      same (analysis       Relativistic pair;
                    (relativistic pair) path, not a Music   no new loss.
                                        reimplementation)
feature matching    OFF (``fm=0``)      OFF (``fm=0``)      Raw FM is uncapped
                                                            by ``b_cap``.
teacher / leftover  ``faithful_guard_e`` same               Leftover-gated
                                                            caption poles.
critic arch         Fourier-2 MLP       ``mlp``             Arm B. Never combine
                    (analysis ``gan``)  (``--adv_arch``)    ``faithful_guard_e``
                                                            with ``tx``
                                                            (dual-arm
                                                            incompatible).
cover weight        1.5 (sheet/exam     1.0                 Field demo needs the
                    mode pin)                               stronger pin; Music
                                                            transfer prefers
                                                            1.0 per handoff.
pole weight         (same cover pin)    1.0                 Music pole/cover
                                                            pair, both 1.0.
particles           ``n<=12``,          ``--parts 0``       Music: no particle
                    ``particle_l2=0.02``(``vicreg=0``)      prior; toy posture
                                                            keeps ``n<=12`` /
                                                            ``l2=0.02`` as
                                                            reference only.
VICReg              small (0.05 demo)   0 when ``parts=0``  No particles to
                                                            regularize.
==================  ==================  ==================  ====================

Do not invent new recipe knobs here. Steps / seeds / LR schedules are run
budgets, not recipe shape, and are intentionally NOT part of this gate.
"""

from __future__ import annotations

from typing import Any, Mapping

# --- Music Arm B locked recipe ------------------------------------------------
MUSIC_ARM_B_SHA_HINT = "435e873"  # Port ParticleGAN RpGAN + b_cap (#94)
MUSIC_ARM_B_GRAD_ARM = "b_cap"
MUSIC_ARM_B_B_CAP = 1.0
MUSIC_ARM_B_KAPPA = 1.0
MUSIC_ARM_B_GRAD_NORM = "l2"
MUSIC_ARM_B_FM_WEIGHT = 0.0
MUSIC_ARM_B_TEACHER = "faithful_guard_e"  # --lm_target value
MUSIC_ARM_B_ADV_ARCH = "mlp"  # never "tx" with faithful_guard_e
MUSIC_ARM_B_PARTS = 0
MUSIC_ARM_B_COVER_WEIGHT = 1.0
MUSIC_ARM_B_POLE_WEIGHT = 1.0
MUSIC_ARM_B_VICREG_WEIGHT = 0.0  # required 0 when parts == 0
# Toy-posture reference only (Field3D demo; moot when --parts 0).
MUSIC_ARM_B_PARTICLE_L2 = 0.02
MUSIC_ARM_B_PARTICLE_L2_REF = MUSIC_ARM_B_PARTICLE_L2

# Field3D demo values kept here so the translation is reviewable, not silent.
FIELD_COVER_WEIGHT = 1.5
FIELD_N_PARTICLES = 12
FIELD_N_PARTICLES_MAX = 12
FIELD_STEPS = 1200


def music_arm_b_dict() -> dict[str, Any]:
    """Flat snapshot of the Music Arm B locked knobs."""
    return {
        "grad_arm": MUSIC_ARM_B_GRAD_ARM,
        "b_cap": MUSIC_ARM_B_B_CAP,
        "kappa": MUSIC_ARM_B_KAPPA,
        "grad_norm": MUSIC_ARM_B_GRAD_NORM,
        "fm_weight": MUSIC_ARM_B_FM_WEIGHT,
        "lm_target": MUSIC_ARM_B_TEACHER,
        "adv_arch": MUSIC_ARM_B_ADV_ARCH,
        "parts": MUSIC_ARM_B_PARTS,
        "cover_weight": MUSIC_ARM_B_COVER_WEIGHT,
        "pole_weight": MUSIC_ARM_B_POLE_WEIGHT,
        "vicreg_weight": MUSIC_ARM_B_VICREG_WEIGHT,
        "particle_l2_ref": MUSIC_ARM_B_PARTICLE_L2_REF,
    }


def _approx_eq(a: Any, b: Any, *, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _lookup(mapping: Mapping[str, Any], *names: str) -> tuple[bool, Any]:
    """Return (found, value) for the first present key in ``names``."""
    for name in names:
        if name in mapping:
            return True, mapping[name]
    return False, None


def _as_mapping(args: Any) -> Mapping[str, Any]:
    if isinstance(args, Mapping):
        return args
    get = getattr(args, "__dict__", None)
    if isinstance(get, dict):
        return get
    out: dict[str, Any] = {}
    for key in (
        "lm_target",
        "adv_arch",
        "b_cap",
        "adv_reg_kappa",
        "kappa",
        "adv_reg_norm",
        "grad_norm",
        "fm_weight",
        "cover_weight",
        "pole_weight",
        "parts",
        "n_particles",
        "vicreg_weight",
    ):
        if hasattr(args, key):
            out[key] = getattr(args, key)
    return out


def check_music_arm_b_argv(args: Any) -> list[str]:
    """Return human-readable mismatch strings (empty = Arm B shape holds).

    Accepts an ``argparse.Namespace`` (e.g. ``train_lm_slider_music3.parse_args``)
    or a plain mapping. Missing keys are reported as drift (fail-closed):
    an Arm B run must explicitly carry the full shape.
    """
    m = _as_mapping(args)
    bad: list[str] = []

    found, lm_target = _lookup(m, "lm_target", "teacher")
    if not found:
        bad.append("missing lm_target (want 'faithful_guard_e')")
    elif str(lm_target) != MUSIC_ARM_B_TEACHER:
        bad.append(f"lm_target={lm_target!r} want {MUSIC_ARM_B_TEACHER!r}")

    found, adv_arch = _lookup(m, "adv_arch")
    if not found:
        bad.append(f"missing adv_arch (want {MUSIC_ARM_B_ADV_ARCH!r})")
    elif str(adv_arch) != MUSIC_ARM_B_ADV_ARCH:
        if str(adv_arch) == "tx":
            bad.append(
                f"adv_arch='tx' with lm_target='faithful_guard_e' is dual-arm "
                f"incompatible (want {MUSIC_ARM_B_ADV_ARCH!r})"
            )
        else:
            bad.append(f"adv_arch={adv_arch!r} want {MUSIC_ARM_B_ADV_ARCH!r}")

    found, b_cap = _lookup(m, "b_cap")
    if not found:
        bad.append(f"missing b_cap (want {MUSIC_ARM_B_B_CAP})")
    elif not _approx_eq(b_cap, MUSIC_ARM_B_B_CAP):
        bad.append(f"b_cap={b_cap} want {MUSIC_ARM_B_B_CAP}")

    found, kappa = _lookup(m, "adv_reg_kappa", "kappa", "adv_reg_kappa_")
    if not found:
        bad.append(f"missing adv_reg_kappa (want {MUSIC_ARM_B_KAPPA})")
    elif not _approx_eq(kappa, MUSIC_ARM_B_KAPPA):
        bad.append(f"adv_reg_kappa={kappa} want {MUSIC_ARM_B_KAPPA}")

    found, norm = _lookup(m, "adv_reg_norm", "grad_norm", "grad_arm_norm")
    if not found:
        bad.append(f"missing adv_reg_norm (want {MUSIC_ARM_B_GRAD_NORM!r})")
    elif str(norm) != MUSIC_ARM_B_GRAD_NORM:
        bad.append(f"adv_reg_norm={norm!r} want {MUSIC_ARM_B_GRAD_NORM!r}")

    found, arm = _lookup(m, "grad_arm", "adv_reg_arm")
    if found and str(arm) != MUSIC_ARM_B_GRAD_ARM:
        bad.append(f"grad_arm={arm!r} want {MUSIC_ARM_B_GRAD_ARM!r}")

    found, fm = _lookup(m, "fm_weight")
    if not found:
        bad.append(f"missing fm_weight (want {MUSIC_ARM_B_FM_WEIGHT}; FM off)")
    elif not _approx_eq(fm, MUSIC_ARM_B_FM_WEIGHT):
        bad.append(f"fm_weight={fm} want {MUSIC_ARM_B_FM_WEIGHT} (FM off)")

    found, cover = _lookup(m, "cover_weight")
    if not found:
        bad.append(f"missing cover_weight (want {MUSIC_ARM_B_COVER_WEIGHT})")
    elif not _approx_eq(cover, MUSIC_ARM_B_COVER_WEIGHT):
        bad.append(
            f"cover_weight={cover} want {MUSIC_ARM_B_COVER_WEIGHT} "
            f"(Field3D demo uses {FIELD_COVER_WEIGHT}; Music transfer is 1.0)"
        )

    found, pole = _lookup(m, "pole_weight")
    if not found:
        bad.append(f"missing pole_weight (want {MUSIC_ARM_B_POLE_WEIGHT})")
    elif not _approx_eq(pole, MUSIC_ARM_B_POLE_WEIGHT):
        bad.append(f"pole_weight={pole} want {MUSIC_ARM_B_POLE_WEIGHT}")

    found, parts = _lookup(m, "parts", "n_particles")
    parts_key = "parts" if "parts" in m else "n_particles"
    if not found:
        bad.append(f"missing parts (want {MUSIC_ARM_B_PARTS})")
    else:
        try:
            n = int(parts)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            bad.append(f"{parts_key}={parts!r} want {MUSIC_ARM_B_PARTS}")
            n = None
        if n is not None and n != int(MUSIC_ARM_B_PARTS):
            bad.append(f"{parts_key}={n} want {MUSIC_ARM_B_PARTS} (Music: no particles)")

    found, vicreg = _lookup(m, "vicreg_weight")
    if not found:
        bad.append(f"missing vicreg_weight (want {MUSIC_ARM_B_VICREG_WEIGHT} when parts=0)")
    elif not _approx_eq(vicreg, MUSIC_ARM_B_VICREG_WEIGHT):
        bad.append(
            f"vicreg_weight={vicreg} want {MUSIC_ARM_B_VICREG_WEIGHT} when parts=0"
        )

    return bad


def assert_music_arm_b_argv(args: Any) -> None:
    """Fail closed if ``args`` drifts from the Music Arm B shape.

    Raises ``AssertionError`` listing every drifted knob. Call this at the top
    of ``train()`` for Arm B runs (before any GPU/model spend).
    """
    bad = check_music_arm_b_argv(args)
    if bad:
        raise AssertionError(
            "Music Arm B argv drifted from locked shape "
            f"(SHA hint {MUSIC_ARM_B_SHA_HINT}): " + "; ".join(bad)
        )


def music_b_cap_penalty(grad_real: Any, grad_fake: Any, *, coeff: float | None = None) -> Any:
    """ParticleGAN-faithful one-sided ``b_cap`` penalty for Music.

    Thin wrapper around ``analysis.slider2d.adv.cap_penalty`` — the tested
    ``0.5*coeff*(mean(relu(||g_r||-1)^2)+mean(relu(||g_f||-1)^2))`` path with
    ``kappa=1`` hardcoded. Do NOT reimplement the formula here (no thinned
    ``kappa``-hardcoded copy without the shared ``GradRegularizer`` path);
    wire to this function instead.
    """
    from analysis.slider2d.adv import cap_penalty

    return cap_penalty(
        grad_real,
        grad_fake,
        coeff=MUSIC_ARM_B_B_CAP if coeff is None else float(coeff),
    )


def music_arm_b_adv_kwargs() -> dict[str, Any]:
    """``AdvConfig``-compatible kwargs for the Music translation.

    ``cover_weight`` is 1.0 (not the Field3D 1.5), ``fm_weight`` 0,
    ``b_cap`` 1. ``n_particles`` is 0 for Music (no particle prior);
    analysis harnesses that actually run ``fit_adv`` should keep the demo
    ``n<=12`` posture instead of passing 0 to ``ParticlePrior``.
    """
    return {
        "b_cap": MUSIC_ARM_B_B_CAP,
        "fm_weight": MUSIC_ARM_B_FM_WEIGHT,
        "cover_weight": MUSIC_ARM_B_COVER_WEIGHT,
        "n_particles": MUSIC_ARM_B_PARTS,
    }


def check_adv_config_arm_b_shape(cfg: Any) -> list[str]:
    """Check an ``AdvConfig`` against the Music Arm B translatable subset.

    Only ``b_cap`` / ``fm_weight`` / ``cover_weight`` transfer 1:1
    (with the documented 1.5 -> 1.0 cover translation). ``kappa``/``l2``
    are properties of the shared ``cap_penalty`` path, and ``parts`` has
    no ``AdvConfig`` analogue for Music (0 = disabled).
    """
    bad: list[str] = []
    if not _approx_eq(getattr(cfg, "b_cap", None), MUSIC_ARM_B_B_CAP):
        bad.append(f"b_cap={getattr(cfg, 'b_cap', None)} want {MUSIC_ARM_B_B_CAP}")
    if not _approx_eq(getattr(cfg, "fm_weight", None), MUSIC_ARM_B_FM_WEIGHT):
        bad.append(
            f"fm_weight={getattr(cfg, 'fm_weight', None)} "
            f"want {MUSIC_ARM_B_FM_WEIGHT} (FM off)"
        )
    if not _approx_eq(getattr(cfg, "cover_weight", None), MUSIC_ARM_B_COVER_WEIGHT):
        bad.append(
            f"cover_weight={getattr(cfg, 'cover_weight', None)} "
            f"want {MUSIC_ARM_B_COVER_WEIGHT} (Music; demo {FIELD_COVER_WEIGHT})"
        )
    return bad


__all__ = [
    "MUSIC_ARM_B_SHA_HINT",
    "MUSIC_ARM_B_GRAD_ARM",
    "MUSIC_ARM_B_B_CAP",
    "MUSIC_ARM_B_KAPPA",
    "MUSIC_ARM_B_GRAD_NORM",
    "MUSIC_ARM_B_FM_WEIGHT",
    "MUSIC_ARM_B_TEACHER",
    "MUSIC_ARM_B_ADV_ARCH",
    "MUSIC_ARM_B_PARTS",
    "MUSIC_ARM_B_COVER_WEIGHT",
    "MUSIC_ARM_B_POLE_WEIGHT",
    "MUSIC_ARM_B_VICREG_WEIGHT",
    "MUSIC_ARM_B_PARTICLE_L2",
    "MUSIC_ARM_B_PARTICLE_L2_REF",
    "FIELD_COVER_WEIGHT",
    "FIELD_N_PARTICLES",
    "FIELD_N_PARTICLES_MAX",
    "FIELD_STEPS",
    "music_arm_b_dict",
    "check_music_arm_b_argv",
    "assert_music_arm_b_argv",
    "music_b_cap_penalty",
    "music_arm_b_adv_kwargs",
    "check_adv_config_arm_b_shape",
]
