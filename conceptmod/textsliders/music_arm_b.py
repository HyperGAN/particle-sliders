"""Music Arm B (RpGAN + b_cap) recipe adapter — interface first.

Winning config from Slider's CPU / Field3D selection work (KEEP / DRv):
``locked_shared`` / hold_ablation, ``recommend=locked_shared``,
``posture=hold_ablation``, ``bottleneck=close``. Lowrank / exam-only arms
are DROP_EX — do not adopt.

This module is the single place operators and the Music trainer read the
Arm B shape from. It is deliberately dependency-free (no torch, no Music
weights) so ``scripts/smoke_music_arm_b_argv.py`` and CI can validate the
argv without loading anything heavy.

Row-by-row contract (handoff table):

==================  =====================================================
Knob                Value
==================  =====================================================
Grad regularizer    ParticleGAN ``b_cap`` — ``coeff=1``, ``kappa=1``,
                    ``norm=l2`` — ``(coeff/2)(E_r+E_f) relu(||grad D||-k)^2``
Adversarial loss    RpGAN logistic (relativistic pair)
Feature matching    OFF — ``fm_weight=0``
Teacher / leftover  ``faithful_guard_e``
Critic arch         ``mlp`` (Arm B). Never combine ``faithful_guard_e``
                    with ``--adv_arch tx`` (dual-arm incompatible)
``b_cap`` /         ``1`` / ``1``
``adv_reg_kappa``
Cover / pole        ``cover_weight=1.0``, ``pole_weight=1.0`` (the Field3D
(Music transfer)    demo uses cover 1.5 — Music trains prefer 1.0)
Particles           Music: ``--parts 0``. Toy posture: ``n_particles<=12``,
                    ``particle_l2=0.02``
==================  =====================================================

Music-posture ADOPTs (when the trainer exposes them):

- ``vicreg_weight=0`` when ``n_particles<=1`` / ``--parts 0``
- for close-family cells: prefer ``n>=2`` or ``vic0@n1``
- always sample / eval scales ``-1, 0, 0.5, 1``
- multi-seed before claiming a win

Propose-only YAML hygiene (not proven trainer defaults): ``e_on_content=0``;
FLP: no lyric/caption row with unused-e primary (``se < max(su,sc)``).

Explicitly do NOT use: FM-on / raw feature matching under ``b_cap``;
``tx`` + ``faithful_guard_e`` in one argv; lowrank_k3 / FLP student arms
as the Music train recipe; thinned "b_cap with kappa hardcoded and no
GradRegularizer".
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Single-source provenance for the demo side of the contract. The 2-D
# harness encodes the same game in ``analysis/slider2d/adv.py``
# (``AdvConfig`` / ``cap_penalty`` / ``rp_d_loss`` / ``rp_g_loss``).
DEMO_SINGLE_SOURCE = "analysis/slider2d/adv.py"
SELECTION_STAMP = (
    "recommend=locked_shared; posture=hold_ablation; bottleneck=close; DRv"
)

# The Field3D demo pins the shared residual with cover 1.5; Music transfer
# prefers 1.0. The recipe carries the Music value; the demo value is kept
# here so drift between the two is explicit, not silent.
DEMO_COVER_WEIGHT = 1.5

# Knobs the current supervised Music trainer (``train_lm_slider_music3.py``)
# honors today. Everything else in the recipe is propose-only until an
# adversarial Music trainer lands — the smoke and ``--arm_b`` wiring treat
# those keys as record-only, never as silent training behavior.
TRAINER_HONORED_KEYS = ("lm_target",)

# Keys that must never appear together with the Arm B teacher in one argv.
INCOMPATIBLE_CRITIC_ARCHES = ("tx", "transformer")

# DROP_EX student arms — must never be adopted as the Music train recipe.
DROP_EX_STUDENT_ARMS = ("lowrank_k3", "flp_student")


@dataclass(frozen=True)
class MusicArmBRecipe:
    """Winning Music Arm B argv, one field per handoff-table row."""

    adv_loss: str = "rpgan_logistic"
    grad_reg: str = "b_cap"
    grad_coeff: float = 1.0
    adv_reg_kappa: float = 1.0
    grad_norm: str = "l2"
    b_cap: float = 1.0
    fm_weight: float = 0.0
    lm_target: str = "faithful_guard_e"
    adv_arch: str = "mlp"
    cover_weight: float = 1.0
    pole_weight: float = 1.0
    parts: int = 0
    vicreg_weight: float = 0.0
    n_particles: int = 0
    particle_l2: float = 0.02
    eval_scales: tuple = (-1.0, 0.0, 0.5, 1.0)
    # Propose-only YAML hygiene (not proven trainer defaults).
    e_on_content: int = 0
    student_arm: str = "arm_b"
    extra: tuple = field(default_factory=tuple)


def get_music_arm_b_recipe() -> MusicArmBRecipe:
    """Return the canonical winning Arm B recipe."""
    return MusicArmBRecipe()


def to_dict(recipe: MusicArmBRecipe | None = None) -> dict:
    """Recipe as a plain dict (trainer-style keys)."""
    r = recipe or get_music_arm_b_recipe()
    return {
        "adv_loss": r.adv_loss,
        "grad_reg": r.grad_reg,
        "grad_coeff": r.grad_coeff,
        "adv_reg_kappa": r.adv_reg_kappa,
        "grad_norm": r.grad_norm,
        "b_cap": r.b_cap,
        "fm_weight": r.fm_weight,
        "lm_target": r.lm_target,
        "adv_arch": r.adv_arch,
        "cover_weight": r.cover_weight,
        "pole_weight": r.pole_weight,
        "parts": r.parts,
        "vicreg_weight": r.vicreg_weight,
        "n_particles": r.n_particles,
        "particle_l2": r.particle_l2,
        "eval_scales": list(r.eval_scales),
        "e_on_content": r.e_on_content,
        "student_arm": r.student_arm,
    }


def to_argv(recipe: MusicArmBRecipe | None = None) -> list[str]:
    """Recipe as a trainer-style argv list for operators to copy/paste."""
    d = to_dict(recipe)
    argv = [
        f"--adv_loss {d['adv_loss']}",
        f"--grad_reg {d['grad_reg']}",
        f"--grad_coeff {d['grad_coeff']}",
        f"--adv_reg_kappa {d['adv_reg_kappa']}",
        f"--grad_norm {d['grad_norm']}",
        f"--b_cap {d['b_cap']}",
        f"--fm_weight {d['fm_weight']}",
        f"--lm_target {d['lm_target']}",
        f"--adv_arch {d['adv_arch']}",
        f"--cover_weight {d['cover_weight']}",
        f"--pole_weight {d['pole_weight']}",
        f"--parts {d['parts']}",
        f"--vicreg_weight {d['vicreg_weight']}",
        f"--eval_scales {','.join(str(s) for s in d['eval_scales'])}",
    ]
    return argv


def validate_music_arm_b_argv(mapping: dict) -> list[str]:
    """Check an argv mapping against the handoff table.

    Returns a list of human-readable diffs; empty means the shape matches.
    Anything non-empty must stop a train — never silently train drifted.
    """
    errors: list[str] = []
    want = to_dict()

    def _num(key: str) -> float | None:
        try:
            return float(mapping[key])
        except (KeyError, TypeError, ValueError):
            return None

    # Grad regularizer: ParticleGAN b_cap, coeff=1, kappa=1, norm=l2.
    if mapping.get("grad_reg") != "b_cap":
        errors.append(
            f"grad_reg={mapping.get('grad_reg')!r} != 'b_cap' "
            "(thinned kappa-hardcoded variants without GradRegularizer are banned)"
        )
    if _num("grad_coeff") != want["grad_coeff"]:
        errors.append(f"grad_coeff={mapping.get('grad_coeff')!r} != 1")
    if _num("adv_reg_kappa") != want["adv_reg_kappa"]:
        errors.append(f"adv_reg_kappa={mapping.get('adv_reg_kappa')!r} != 1")
    if _num("b_cap") != want["b_cap"]:
        errors.append(f"b_cap={mapping.get('b_cap')!r} != 1")
    if mapping.get("grad_norm") != "l2":
        errors.append(f"grad_norm={mapping.get('grad_norm')!r} != 'l2'")

    # Adversarial loss: relativistic-pair logistic only.
    if mapping.get("adv_loss") != "rpgan_logistic":
        errors.append(
            f"adv_loss={mapping.get('adv_loss')!r} != 'rpgan_logistic' (RpGAN)"
        )

    # Feature matching OFF — raw FM is uncapped by b_cap.
    if _num("fm_weight") != 0.0:
        errors.append(
            f"fm_weight={mapping.get('fm_weight')!r} != 0 "
            "(FM-on / raw feature matching under b_cap is banned)"
        )

    # Teacher / leftover gate.
    if mapping.get("lm_target") != "faithful_guard_e":
        errors.append(
            f"lm_target={mapping.get('lm_target')!r} != 'faithful_guard_e'"
        )

    # Critic arch mlp; tx + faithful_guard_e is dual-arm incompatible.
    arch = mapping.get("adv_arch")
    if arch != "mlp":
        errors.append(f"adv_arch={arch!r} != 'mlp' (Arm B)")
    if arch in INCOMPATIBLE_CRITIC_ARCHES and mapping.get("lm_target") == (
        "faithful_guard_e"
    ):
        errors.append(
            f"adv_arch={arch!r} + faithful_guard_e is dual-arm incompatible"
        )

    # Music transfer: cover/pole 1.0, not the Field3D demo 1.5.
    if _num("cover_weight") != want["cover_weight"]:
        errors.append(
            f"cover_weight={mapping.get('cover_weight')!r} != 1.0 "
            f"(Music transfer; the Field3D demo uses {DEMO_COVER_WEIGHT})"
        )
    if _num("pole_weight") != want["pole_weight"]:
        errors.append(f"pole_weight={mapping.get('pole_weight')!r} != 1.0")

    # Particles: --parts 0 on Music; vicreg 0 in that posture.
    if mapping.get("parts") != 0 and _num("parts") != 0.0:
        errors.append(
            f"parts={mapping.get('parts')!r} != 0 (Music posture is --parts 0)"
        )
    if _num("vicreg_weight") != 0.0:
        errors.append(
            f"vicreg_weight={mapping.get('vicreg_weight')!r} != 0 "
            "(vicreg_weight=0 when n_particles<=1 / --parts 0)"
        )

    # Eval scales always -1, 0, 0.5, 1.
    scales = mapping.get("eval_scales")
    try:
        got_scales = tuple(float(s) for s in scales)
    except (TypeError, ValueError):
        got_scales = ()
    if got_scales != tuple(want["eval_scales"]):
        errors.append(
            f"eval_scales={scales!r} != {want['eval_scales']} "
            "(always sample/eval scales -1, 0, 0.5, 1)"
        )

    # DROP_EX student arms must never be the Music recipe.
    if mapping.get("student_arm") in DROP_EX_STUDENT_ARMS:
        errors.append(
            f"student_arm={mapping.get('student_arm')!r} is DROP_EX "
            "(lowrank / exam-only arms are not the Music train recipe)"
        )

    return errors


def honored_trainer_overrides(
    recipe: MusicArmBRecipe | None = None,
) -> dict:
    """Subset of the recipe the current supervised trainer honors today.

    Only ``lm_target=faithful_guard_e`` exists in
    ``train_lm_slider_music3.py``; every adversarial knob (critic, RpGAN,
    b_cap, parts, cover/pole, FM, vicreg) is propose-only until an
    adversarial Music trainer lands.
    """
    r = recipe or get_music_arm_b_recipe()
    return {"lm_target": r.lm_target}


def apply_arm_b_defaults(args, *, recipe: MusicArmBRecipe | None = None,
                         explicit_lm_target: bool = True):
    """Apply the honored Arm B subset onto a trainer argparse namespace.

    Fail-closed: if the operator explicitly passed a conflicting
    ``--lm_target``, raise instead of silently training a drifted recipe.
    A non-explicit trainer default (``explicit_lm_target=False``) is
    overridden with a printed notice. The full (propose-only) recipe is
    printed so before/after verification from the handoff can be done by
    eye. Returns the namespace.
    """
    r = recipe or get_music_arm_b_recipe()
    current = getattr(args, "lm_target", None)
    if current is not None and current != r.lm_target and explicit_lm_target:
        raise ValueError(
            f"--arm_b requires --lm_target {r.lm_target!r}, "
            f"got {current!r}: stop and report the diff, do not silently "
            "train a drifted recipe"
        )
    args.lm_target = r.lm_target
    print("[arm_b] applied honored trainer subset:", honored_trainer_overrides(r))
    print("[arm_b] full recipe argv (adv keys are propose-only on this trainer):")
    for item in to_argv(r):
        print(f"  {item}")
    print(
        "[arm_b] verify before/after train: critic mlp + faithful_guard_e, "
        "FM=0, b_cap=1, kappa=1, parts=0, cover/pole=1.0"
    )
    return args
