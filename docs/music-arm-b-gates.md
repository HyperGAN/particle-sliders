# Music Arm B gates — Field3D honesty mapping (CPU-only)

Arm B = `#94` ParticleGAN-faithful RpGAN + `b_cap`, leftover-gated:
`fm_weight=0`, `adv_b_cap=1`, `adv_reg_kappa=1`, `faithful_guard_e`, critic
`mlp` (`--adv_norm l2`), `--parts 0`, `pole_weight=1`, `cover_weight=1`,
`vicreg_weight=0`. Fail-closed gates: `tests/test_music_arm_b_gates.py`
(shape) + `tests/test_music_arm_b.py` (formulation); single source is the
trainer `ARM_B` row in `conceptmod/textsliders/train_lm_slider_music3.py`.
No Music GPU train in this port.

This note reconciles Strategy E (#100, tests-first gates) onto Strategy B
(#101, ParticleGAN `GradRegularizer` + `--adv_preset arm_b`), which merged
first. Formulation ownership is #101's; what this layer adds is E's
fail-closed shape coverage: the `--vicreg_weight` 0-at-parts-0 gate,
`--require_arm_b` parse-time refusal, and the Field3D-honesty mapping below.

## Field3D honesty gates → what Music can assert without GPU

| Field3D gate | Meaning | Music CPU assertion | Cannot claim without GPU |
|---|---|---|---|
| leak ≤ 0.20 | hidden-geometry leftover lock (`COMPILED_LEAK_LOCK`) | analysis sheet/exam cells already assert it (`test_lm_2d_adv`); trainer precondition is the leftover-gated teacher, enforced by `--adv_preset arm_b` / `--require_arm_b` | rendered-audio leak on a Music run |
| DR✓ | double-render verified | preconditions only: Arm B argv shape, `mlp` critic, FM off, parts 0 | DR✓ itself — double render is GPU by definition |
| locked_shared | selection stamp: one shared recipe, hold_ablation | trainer `ARM_B` single source + parse-time refusal before spend | any win claim (needs multi-seed GPU + eval scales `-1, 0, 0.5, 1`) |

Live-default posture: the trainer keeps `--lm_target v9` / `--pole_mode
hidden` / `--adv_preset none`; Arm B is explicit (`--lm_target
faithful_guard_e` + `--adv_preset arm_b`, or `--require_arm_b`) and
checked, never silent. Flipping the live teacher default is a GPU-train
decision, out of scope here.

## Residual intentional diffs vs ParticleGAN 100-Gaussians (ranked)

1. **parts** — ParticleGAN: 20k-particle prior (`z_dim=4`); Music Arm B:
   `--parts 0` (no cloud at all). Biggest structural diff.
2. **VICReg** — ParticleGAN: `ParticleRegularizer` weight 1 (var+cov);
   Music Arm B: `vicreg_weight=0` at parts 0 (regularizer absent by
   construction; toy uses sim+var+cov at 0.05). Gated: any nonzero
   `--vicreg_weight` fails `--adv_preset arm_b` / `--require_arm_b`.
3. **LR / schedule** — ParticleGAN: 6e-4 G, ×1.5 D, ×10 prior, 60%
   anneal hold; Music trainer: `lr=5e-4` + early-stop (toy: shared
   5e-3, delay 80). Different optimization regime.
4. **Thinned `b_cap` — RESOLVED by #101.** E's checkout pinned a local
   `cap_penalty` with κ hardcoded; main now vendors ParticleGAN's real
   `GradRegularizer` (`analysis/slider2d/grad_regularizers.py`, SOURCE-pinned
   to `particlegan 0.2.0`), `cap_penalty` delegates its phi to
   `GradRegularizer._phi`, and κ rides `--adv_reg_kappa` (explicit, tested
   at non-default values). The gate in `test_music_arm_b_gates.py` pins the
   faithful shape through that path so it cannot silently thin again.
5. **Minor** — Adam β2 0.99 vs 0.999; EMA on residual only (not a
   prior cloud).

Matches kept: RpGAN logistic pair loss, `b_cap` coeff 1, FM off,
`faithful_guard_e` teacher on both demo (`DEFAULT_TEACHER`) and Arm B
argv. Music extras layered on top: cover/pole weights, span/end cloud,
`particle_l2`, sheet/exam scaffolding.

## Propose-only (not gates)

Reoc0 (`e_on_content=0` lexical proxy) and FLP (`se < max(su,sc)` over
caller-supplied vectors) live in `conceptmod/textsliders/arm_b_yaml_lint.py`
with `PROPOSE_ONLY = True`. Encoder wiring, thresholds, eval scales, and
multi-seed wins stay unproven until a GPU run.
