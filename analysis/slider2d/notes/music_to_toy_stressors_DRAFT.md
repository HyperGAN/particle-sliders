# Music → toy Field2D/Field3D stressors — 2026-09-09

Frame: catalog Music LM (#94 / Arm B+T) failure modes, then mirror each as a
toy stress cell under locked recipe (1200 / cover=1.5 / faithful_guard_e / FM0 /
n_particles≤12 / particle_l2=0.02 / b_cap=1). Prefer parts0 proxy = n_particles=1.
No Music GPU train.

## Music-only issues → toy cells

| # | Music failure / risk | Evidence (notes) | Toy stressor cell | Status |
|---|---|---|---|---|
| M1 | Span-entangled leftover (lyric spans share residual with content) | Fire #13 `cross_axis_span_sample` 0/6 u≈0.83; PairField entangle notes | `cross_axis_span_sample` / new `lyric_span_entangle` | FAIL solid — hard boundary candidate |
| M2 | Multi-span lyrics = heterogeneous per-row axis mix | Fire #13 `cross_axis_rows` 0/6 u≈0.58 | `cross_axis_rows` | FAIL solid — dig/harden or hard-bound |
| M3 | Close-pair seed knife (delivery/close axes flake) | Fire #17/#18 f3d_close @ n=1: 5/6 both cover 1.0 & 1.5; **seed0 only** | `f3d_close` @ n=1 | knife — dig seed0 |
| M4 | Dual-arm leftover vs listen cannot share argv | `tx ∩ faithful_guard_e` SystemExit; dual_arm strategy | separate exam cells: `arm_leftover_gate_cover` vs `arm_listen_cover_only` | already ablated (gate∧cover required) |
| M5 | `--parts 0` regime (no ParticlePrior on Music) | Fire #15/#17 n=1 ≈ n=12 on Arm A; Music parts0 | force `n_particles=1` on all new stressors | locked Music proxy |
| M6 | Bad declared leak YAML / amplitude lie | Fire #13 `axis_leak_primary` leak≈2.15; `cross_axis_mismatch_declare` | keep as **negative controls** | expected FAIL |
| M7 | Cover-only listen arm leaks ê | dual_arm cover_only sheet leak≈0.23 | `cover_only_no_guard` cell | documented FAIL |
| M8 | Gate-only (pole=0) undershoots | leftover_only sheet kept≈0.48 | `guard_only_cover0` | documented FAIL |
| M9 | False lock short×high-pin | Fire #4 800×cover3.0 1/6 | reject cell (do not adopt) | reject |
| M10 | lyrichold ≠ cover / pole | transfer gaps | do **not** equate in toy | doc only |

## Open digs (this job)

1. **M3 / f3d_close seed0 @ n=1** — diagnose why seed0 fails when {1,2,3,7,42} PASS; try multi-seed harden (init/jitter) OR document flake margin.
2. **M2 / cross_axis_rows** — can Field3D scaffold/recipe be made robust under true cross-axis R³, or hard-bound with repro numbers?
3. **New biting stressor** — implement ≥1 Music→toy cell that locked recipe fails (prefer lyric-span-entangle / dual-arm listen-as-cell / close@parts0).

## Locked recipe (unchanged)

Toy AdvConfig: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, fm=0,
n_particles≤12 (Music proxy n=1), particle_l2=0.02, b_cap=1.
Reject 800×cover3.0.
