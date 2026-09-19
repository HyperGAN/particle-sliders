# Strategy locked: Recommendation B (2026-09-09) — cover **1.0**

Music #94 **true transfer target** (Arm B / leftover mlp):

```text
faithful_guard_e + mlp + pole_weight=1.0 + FM0 + b_cap1 + parts0
```

| item | role |
|---|---|
| leftover gate `faithful_guard_e` | required |
| **pole_weight / toy cover_weight = 1.0** | locked Music-transferable (was 1.5 toy default) |
| cover 0.5 | knife-edge — do not adopt |
| cover 2.0+ | unnecessary |
| `fm_weight=0`, `b_cap` c=κ=1, `--parts 0` | required |
| Arm T `faithful_plus_neu_lyric` + tx | diagnostic only |
| cover_only / gate-only (`cover=0`) | fails / leaks |

Toy note: historical 2D AdvConfig default `cover_weight=1.5` still works; Music-transfer
posture prefers **1.0** (sheet+Field3D 3/3 in parallel sweep). Stress Field3D under
`n_particles=1`, leftover∧cover=1.0.

Smoke: `music_arm_b_locked_smoke_20260909.sh` (`POLE_WEIGHT=1`).
