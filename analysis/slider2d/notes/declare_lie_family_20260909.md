# Declare-lie family — unified theory + negative-control suite (2026-09-09)

Host: box-cpu @ `435e87363bf1`. Wall 1243.4s. CPU only. No Music GPU.
Locked shared AdvResidual **unchanged**. per-row = analysis-only.

## Theory

**Declare-lie family:** the YAML / declared-ê vector is *not* the exam leak axis
the leftover gate scores. Under `faithful_guard_e`, the teacher subtracts along
the *declared* direction. When that direction restates **û** (M14), points at
**content** (M20), or splits ambiguously (M26), the gate deletes or fails content
while û may still look healthy.

| member | lie geometry | locked nature | exam≈ | content | leak |
|---|---|---|---:|---:|---:|
| **M14** `e_on_u_declare_lie` | declared ê ∥ û (`e_on_u` hot) | content_deleted_under_declare_lie | 0.1817 | 0.1817 | 0.1007 |
| **M20** `amp_lie_leftover_declare` | declared ê ∥ content on leftover geom | content_deleted_under_declare_lie | -0.0826 | -0.0826 | 0.3841 |
| **M26** `declare_split_three` | ê split û/content/unused | content_deleted_under_declare_lie | 0.0914 | 0.0914 | 0.0499 |
| **M6** `cross_axis_mismatch_declare` | hetero mismatch declare (legacy neg) | declare_lie_raw_leak_blowup | 0.6496 | 1.3286 | 0.7064 |

**Not in family (DoF contrast):** M16 `scale_stagger_homo` — shared BITES multi_row,
per-row **clears**. Fix = more heads / per-row residual (analysis-only), not YAML.

**Not in family (ê-floor):** M21/M30/M31 — teacher declared_e tilt vs exam pure ê;
analytic floor; per-row cannot clear (see `m21_eoc_leak_irreducible`).

**Not in family (guard refuse):** M28 — blend guard refuses → raw-pole leak≈0.66.

### Why n / vic / per-row cannot clear

Declare-lie is a **target/YAML bug**, not a particle / VICReg / DoF bug.
More particles or per-row heads still train toward the *wrong declared axis*;
the leftover content gate still fails (or content stays deleted).

### Cliff (M14)

`e_on_u=0` → PASS; **`e_on_u≥0.15` → BITES** (sharper than prior ≥0.3 note; any û-restatement on).


## Taxonomy (fail modes)

| family | members | clears with per-row? | clears with n/vic? | fix path |
|---|---|---|---|---|
| **declare-lie content delete** | M14, M20, M26 | NO | NO | YAML / declared ê |
| **declare-lie raw leak blowup** | M6 mismatch | NO | NO | YAML (legacy neg) |
| **ê-floor / eoc** | M21, M30, M31 | NO | NO | don't strip eoc; document floor |
| **guard refuse** | M28 | NO | NO | don't hot-YAML past guard |
| **DoF multi-row** | M1, M16, M17, M2, M24, M27, M29 | YES (w≤0.3 analysis) | NO | per-row NON_DEFAULT only |

## Verdict

family M14/M20 natures=content_deleted_under_declare_lie/content_deleted_under_declare_lie same_content_delete=True; M26=content_deleted_under_declare_lie; M6=declare_lie_raw_leak_blowup; clearance n/vic/per_row any_declare=False ({'M14': {'n2': False, 'vic0': False, 'per_row': False}, 'M20': {'n2': False, 'vic0': False, 'per_row': False}, 'M26': {'n2': False, 'vic0': False, 'per_row': False}}); neg_ok={'leftover': True, 'm14_eou0': True, 'm14_faithful': True, 'm20_faithful': True, 'm26_faithful': True, 'm16_shared_bites': True, 'm16_per_row_clears': True}; suite_pass=True; keep HARD_BITEs; recipe_change=NO

## Negative-control suite

| control | expected | result | PASS |
|---|---|---|:---:|
| leftover CTRL | PASS | 3/3 | YES |
| M14 eou=0 | PASS | 3/3 | YES |
| M14 faithful (no guard) | PASS | 3/3 | YES |
| M20 faithful | PASS | 3/3 | YES |
| M26 faithful | PASS | 3/3 | YES |
| M16 shared | BITES | 0/3 | YES |
| M16 per_row | CLEARS | 3/3 | YES |

**suite_pass = True** (declare-lies stay biting; negs + DoF contrast hold).

## Clearance (expect all NO)

| mid | n2 | vic0@n1 | per_row |
|---|:---:|:---:|:---:|
| M14 | 0/3 | 0/3 | 0/3 |
| M20 | 0/3 | 0/3 | 0/3 |
| M26 | 0/6 | 0/6 | 0/6 |

## eou ladder

| eou | PASS | exam | content |
|---|:---:|---:|---:|
| 0.0 | 3/3 | 0.9563 | 0.9955 |
| 0.15 | 0/3 | 0.1799 | 0.1799 |
| 0.3 | 0/3 | 0.1799 | 0.1799 |
| 1.5 | 0/3 | 0.1799 | 0.1799 |

## Full cells

| cell | mode | PASS | exam | u | content | leak | fail |
|---|---|:---:|---:|---:|---:|---:|---|
| `CTRL_leftover` | shared | 3/3 | 0.9304 | 0.9918 | 0.9949 | 0.0003 | [] |
| `M14_locked` | shared | 0/6 | 0.1817 | 0.9878 | 0.1817 | 0.1007 | [0, 1, 2, 3, 7, 42] |
| `M20_locked` | shared | 0/6 | -0.0826 | 0.9854 | -0.0826 | 0.3841 | [0, 1, 2, 3, 7, 42] |
| `M26_locked` | shared | 0/6 | 0.0914 | 0.9882 | 0.0914 | 0.0499 | [0, 1, 2, 3, 7, 42] |
| `M6_mismatch_declare` | shared | 0/6 | 0.6496 | 0.6496 | 1.3286 | 0.7064 | [0, 1, 2, 3, 7, 42] |
| `NEG_M14_eou0` | shared | 3/3 | 0.9563 | 0.9883 | 0.9955 | 0.0009 | [] |
| `NEG_M14_faithful` | shared | 3/3 | 0.9885 | 0.9885 | 0.9929 | 0.3548 | [] |
| `NEG_M20_faithful` | shared | 3/3 | 0.9874 | 0.9874 | 0.9938 | 0.4547 | [] |
| `NEG_M26_faithful` | shared | 3/3 | 0.9895 | 0.9895 | 0.994 | 0.4531 | [] |
| `CONTRAST_M16_shared` | shared | 0/3 | 0.9434 | 1.4481 | 1.5093 | 0.0006 | [0, 1, 2] |
| `M14_n2` | shared | 0/3 | 0.1851 | 0.988 | 0.1851 | 0.1021 | [0, 1, 2] |
| `M14_n1_vic0` | shared | 0/3 | 0.1826 | 0.9831 | 0.1826 | 0.1021 | [0, 1, 2] |
| `M14_per_row` | per_row | 0/3 | 0.9868 | 0.9906 | 0.1802 | 0.1019 | [0, 1, 2] |
| `M20_n2` | shared | 0/3 | -0.0828 | 0.9866 | -0.0828 | 0.3838 | [0, 1, 2] |
| `M20_n1_vic0` | shared | 0/3 | -0.0829 | 0.9882 | -0.0829 | 0.3811 | [0, 1, 2] |
| `M20_per_row` | per_row | 0/3 | 0.9914 | 0.9914 | -0.083 | 0.3803 | [0, 1, 2] |
| `M26_n2` | shared | 0/6 | 0.0916 | 0.9899 | 0.0916 | 0.0517 | [0, 1, 2, 3, 7, 42] |
| `M26_n1_vic0` | shared | 0/6 | 0.0905 | 0.9786 | 0.0905 | 0.0515 | [0, 1, 2, 3, 7, 42] |
| `M26_per_row` | per_row | 0/6 | 0.9858 | 0.993 | 0.0901 | 0.0507 | [0, 1, 2, 3, 7, 42] |
| `CONTRAST_M16_per_row` | per_row | 3/3 | 0.9998 | 1.0103 | 1.0181 | 0.0047 | [] |
| `M14_eou0.0` | shared | 3/3 | 0.9563 | 0.9883 | 0.9955 | 0.0009 | [] |
| `M14_eou0.15` | shared | 0/3 | 0.1799 | 0.9881 | 0.1799 | 0.1005 | [0, 1, 2] |
| `M14_eou0.3` | shared | 0/3 | 0.1799 | 0.9881 | 0.1799 | 0.1005 | [0, 1, 2] |
| `M14_eou1.5` | shared | 0/3 | 0.1799 | 0.9881 | 0.1799 | 0.1005 | [0, 1, 2] |

## Recipe / ADOPT

- Recipe change: **NO**
- merge_to_trainer: **NO**
- Keep M14 / M20 / M26 as HARD_BITEs (overpowered-head / YAML-lie falsifiers)
- Music-posture n≥2 / vic0@n1 do **not** apply
- Per-row w≤0.3 ADOPT does **not** clear this family (clears DoF only)
- Fix path: correct declared ê / YAML targets — not recipe knobs

JSON: `declare_lie_family_20260909.json`
