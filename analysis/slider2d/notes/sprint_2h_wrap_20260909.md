# 2h adversarial-slider sprint wrap — 2026-09-09

Host: pop-os `/ml2/music/sliders-conceptmod` @ `435e873` (+ adv CLI restore).
CPU Field3D/2D only. **No Music GPU train.** nano-work-server / ComfyUI /
`run_server.py` left alone.

**Definitive Music transfer card:**
[`MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md`](MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md)

```text
Music #94 = faithful_guard_e + mlp + pole_weight=1 + FM0 + b_cap(c=κ=1) + parts0
            [VERIFY_ONLY; Arm T lyric+tx = listen diagnostic only]
```

---

## What transfers

| finding | evidence | Music implication |
|---|---|---|
| Leftover gate `faithful_guard_e` | Fire #1–#2; raw `faithful` leak≈0.48 | `--lm_target faithful_guard_e` + YAML `leak_*` |
| Gate **∧** cover/pole together | Fire #15 `A_n1_cover0` 0/6; Rec B ablation 3/3 vs 0/3 single-arm | modest `--pole_weight` required with gate |
| Music-transfer cover/pole **= 1.0** | parallel Arm B sweep; sheet+most Field3D @ n=1 | `--pole_weight 1` (not invent `--cover_weight`) |
| `b_cap` c=κ=1, FM off | Fire #6–#9; CLI restore defaults | `--fm_weight 0`, `--adv_reg_coeff/kappa 1` |
| No ParticlePrior port | Fire #15 `n=1` ≈ `n=12` on Arm A (sheet kept 0.935 vs 0.930) | **`--parts 0` always** |
| PairField leftover cells | Fire #12/#13/#17: divergent/unused_e/leftover 6/6 | mlp leftover arm exam rows |
| Sheet leftover @ n=1 cover=1.0 | Fire #17: 6/6 kept mean **0.9334** | Arm B posture solid on sheet |
| Extreme mismatch solid-FAIL | Fire #13/#17: cross_declare 0/6 | fix YAML `leak_*`, don’t retune recipe |
| Dual-arm constraint | gaps note: tx∩guard SystemExit | sequential Arm B vs Arm T — never one argv |
| False lock killed | Fire #4: 800×cover3.0 1/6 | never short×high-pin |

---

## What does **not** transfer / caveats

| item | why |
|---|---|
| `n_particles` / `particle_l2` → `--parts` | different objects (latent prior vs row miner) |
| `tx ∩ faithful_guard_e` | hard SystemExit; Arm T is diagnostic only |
| Toy `cover_weight=1.5` as Music start | Music-locked pole=**1.0**; 1.5 unnecessary for Arm B |
| Arm T alone as #94 | listen/span smoke; cover_only without gate leaks |
| `lyrichold_weight ≡ cover` | lyrichold is UNI/tx; leftover arm uses `pole_weight` |
| Pathological Field3D mismatch | recipe won’t rescue wrong declared ê |
| **`f3d_close` seed0 flake @ n=1** | Fire #17/#18: 5/6 both cover 1.0 and 1.5; seeds≠0 PASS — multi-seed close check on Music |

---

## Portable recipe status

| surface | status |
|---|---|
| Toy AdvConfig historical | steps=1200, cover=1.5, guard_e, FM0, n≤12, l2=0.02, b_cap=1 — **unchanged** for toy defaults; do not adopt 800×c3 |
| **Music #94 Arm B** | **LOCKED** per `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md` (pole=1, parts=0, mlp, FM0, b_cap1) |
| Smoke wrapper | `music_arm_b_locked_smoke_20260909.sh` (VERIFY_ONLY) |

---

## Sprint fires (this agent track)

| fire | verdict | path |
|---|---|---|
| Pytest | 48 passed (2d_adv+highd); `test_field3d` 8 passed | `tests/test_field3d.py` |
| #13 hard multipair | `hard_multipair_partial` — PairField OK; mismatch 0/6 | `field3d_hard_multipair_20260909.*` |
| #15 dual-arm (partial) | n=1 gate+cover solid; cover0 fails | `/tmp/fire15_dual_arm.log` (interrupted for cover=1.0 priority) |
| #17 n=1 cover=1.0 multipair | `n1_cover1_partial` — only f3d_close knife | `hard_multipair_n1_cover1_20260909.*` |
| #18 close cover knife | `close_fragile_both` — seed0 only | `f3d_close_cover_knife_20260909.*` |

Also: `music_lm_transfer_map_20260909.md`, `strategy_recB_locked_20260909.md`,
`mismatch_fail_music_yaml_20260909.md`, `field3d_README_20260909.md`.

---

## Recommended next Music experiment (**hypotheses only** — no train this sprint)

1. **VERIFY_ONLY** Arm B: `bash analysis/slider2d/notes/music_arm_b_locked_smoke_20260909.sh`
2. When Mikkel greenlights: idle GPU≠0, `RUN_TRAIN=1`, energy-v4 YAML with `leak_*`,
   locked argv from `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md`
3. Seed-check close/delivery pairs (≥6 seeds); if seed0-like flake, demand margin —
   do **not** jump pole_weight to 1.5 without evidence (Fire #18: 1.5 doesn’t fix seed0)
4. Arm T listen smoke separately — do not merge LoRA with Arm B

---

## Stop

Sprint wall ~2h from 14:55 MT. In-flight Fire #17/#18 complete. No GPU Music train.
Servers untouched.
