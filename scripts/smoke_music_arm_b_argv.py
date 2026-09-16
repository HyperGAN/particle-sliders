#!/usr/bin/env python3
"""Smoke the Music Arm B (RpGAN + b_cap) argv without loading Music weights.

Prints the winning argv from ``conceptmod.textsliders.music_arm_b``,
validates it against the handoff table, and proves the validator catches
single-knob drift (FM-on, demo cover 1.5, parts!=0, wrong teacher,
tx critic, wrong kappa). Exits 0 only if the Arm B shape matches;
any diff exits 1 so a drifted recipe can never silently train.

Usage:
    python scripts/smoke_music_arm_b_argv.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.music_arm_b import (
    DEMO_SINGLE_SOURCE,
    SELECTION_STAMP,
    get_music_arm_b_recipe,
    to_argv,
    to_dict,
    validate_music_arm_b_argv,
)


# Each drift must be rejected: (label, mutated key, mutated value).
_DRIFTS = [
    ("fm_on", "fm_weight", 0.1),
    ("demo_cover", "cover_weight", 1.5),
    ("pole_drift", "pole_weight", 1.5),
    ("parts_nonzero", "parts", 4),
    ("vicreg_on_at_parts0", "vicreg_weight", 0.05),
    ("wrong_teacher", "lm_target", "faithful_raw"),
    ("tx_critic", "adv_arch", "tx"),
    ("kappa_drift", "adv_reg_kappa", 0.5),
    ("b_cap_drift", "b_cap", 2.0),
    ("wrong_adv_loss", "adv_loss", "hinge"),
    ("wrong_grad_reg", "grad_reg", "r1"),
    ("dropped_student", "student_arm", "lowrank_k3"),
    ("bad_scales", "eval_scales", [-1.0, 1.0]),
]


def main(argv: list[str] | None = None) -> int:
    del argv
    recipe = get_music_arm_b_recipe()
    mapping = to_dict(recipe)

    print("=== Music Arm B recipe ===")
    print(f"selection: {SELECTION_STAMP}")
    print(f"demo single-source: {DEMO_SINGLE_SOURCE}")
    print()
    print("=== Winning argv ===")
    for item in to_argv(recipe):
        print(item)
    print()

    failures: list[str] = []

    errors = validate_music_arm_b_argv(mapping)
    if errors:
        failures.append("canonical recipe failed validation:")
        failures.extend(f"  - {e}" for e in errors)
    else:
        print("canonical recipe: SHAPE MATCHES handoff table")

    print()
    print("=== Drift rejection checks ===")
    for label, key, value in _DRIFTS:
        drifted = copy.deepcopy(mapping)
        drifted[key] = value
        drift_errors = validate_music_arm_b_argv(drifted)
        if not drift_errors:
            failures.append(f"drift {label!r} ({key}={value!r}) was NOT rejected")
            print(f"  [FAIL] {label}: not rejected")
        else:
            print(f"  [ok] {label}: rejected ({drift_errors[0]})")

    # Optional cross-check against the live trainer's honored subset. The
    # trainer needs torch, so a missing import only skips this probe — the
    # exit code stays tied to the Arm B shape above.
    print()
    print("=== Trainer honored-subset probe (optional, needs torch) ===")
    try:
        from conceptmod.textsliders.train_lm_slider_music3 import parse_args

        honored = {"lm_target": mapping["lm_target"]}
        probed = parse_args(
            ["--prompts_file", "x.yaml", "--lm_target", honored["lm_target"]]
        )
        if probed.lm_target != honored["lm_target"]:
            failures.append("trainer does not honor lm_target=faithful_guard_e")
        else:
            print("  [ok] trainer accepts --lm_target faithful_guard_e")
        defaulted = parse_args(["--prompts_file", "x.yaml"])
        if defaulted.lm_target != "v9":
            failures.append("trainer live default moved away from v9")
        else:
            print("  [ok] trainer live default is still v9 (untouched)")
    except ImportError as exc:
        print(f"  [skip] torch/trainer unavailable: {exc}")

    print()
    if failures:
        print("SMOKE FAILED:")
        for line in failures:
            print(line)
        return 1
    print("SMOKE PASSED: Arm B shape matches handoff table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
