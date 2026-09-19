# Declare-lie / eoc-floor family — unified theory (2026-09-09)

Host: box-cpu @ `435e87363bf1`. Wall 827.8s. CPU only. **No Music GPU.**
Locked shared AdvResidual **unchanged**. per-row = analysis-only. merge=**NO**.

## One-page family map

```
faithful_guard_e  ×  declared_ê  (YAML / e_on_* tilt)
        │
        ├─► A  content_deleted_under_declare_lie
        │      M14 e_on_u_declare_lie   (ê ∥ û)
        │      M20 amp_lie_leftover     (ê ∥ content)
        │      M26 declare_split_three  (ê split)
        │      cliff: e_on_u=0 PASS → ≥0.15 BITES; faithful-no-guard PASS
        │      fix: YAML / declared target — NOT n/vic/per-row
        │
        ├─► B  ê-floor (admit mode)
        │      M21 hold_e_lyric_mix     eoc=0.35  lr_floor=0.20503
        │      M30 eoc_threshold_edge   eoc=0.33  (analytic edge)
        │      M31 leftover_hot_eoc     eoc=1.15 leftover amps (admits)
        │      cliff: analytic eoc 0.32Y/0.33N; emp@0.32 already BITES (residual); valley 0.55–0.65 ana<gate then refuse≥0.68
        │      exam scores pure leak_e; teacher subtracts tilted declared_e
        │      fix: declared_e / eoc geometry — NOT recipe / per-row
        │
        └─► C  guard-refuse / raw-pole
               M28 guard_refuse_hot_eoc eoc=0.85  lr≈0.66
               cliff: refuse onset eoc≥0.68 (to_pole > to_mid)
               teacher falls back to raw poles → full geometry leak
```

| ID | cell | branch | admit? | locked | leak≈ | per-row | cliff |
|---|---|---|:---:|:---:|---:|:---:|---|
| **M14** | `e_on_u_declare_lie` | A content_deleted | Y | 0/6 | 0.1007 | NO | eou 0→≥0.15 |
| **M20** | `amp_lie_leftover_declare` | A content_deleted | Y | 0/6 | 0.3841 | NO | guard×YAML |
| **M26** | `declare_split_three` | A content_deleted | Y | 0/2 | 0.0503 | NO | sibling A |
| **M21** | `hold_e_lyric_mix` | B ê-floor admit | Y | 0/6 | 0.211 | NO | eoc≥0.33 |
| **M30** | `eoc_threshold_edge` | B ê-floor admit | Y | 0/6 | 0.207 | knife 1/6 | ana 0.32Y/0.33N; emp@0.32 bites |
| **M31** | `leftover_hot_eoc_declare` | B ê-floor admit | Y | 0/3 | 0.2067 | NO | leftover×hot |
| **M28** | `guard_refuse_hot_eoc` | C refuse raw | N | 0/6 | 0.66 | NO | eoc≥0.68 |

**Not in family:** M16/M17/M2/M24/M27/M29 = **DoF** (per-row clears). M6 mismatch_declare = legacy hetero neg (leak blowup, not content-delete).

## Cliffs (this dig + prior)

### 1. M14 e_on_u ladder (branch A)

| eou | emp PASS | exam | content |
|---|:---:|---:|---:|
| 0.0 | 3/3 | 0.9563 | 0.9955 |
| 0.15 | 0/3 | 0.1799 | 0.1799 |
| 0.30 | 0/3 | 0.1799 | 0.1799 |

**Cliff:** any û-restatement on (`e_on_u≥0.15`) → content_deleted under guard (finer than prior ≥0.3 report; eou=0 still PASS).

### 2. M21-pool eoc ladder (branch B → C)

| eoc | admit | ana_lr | teach_lr | mode | anchor |
|---:|:---:|---:|---:|---|---|
| 0.00 | Y | 0.0000 | 0.0000 | pass_clean |  |
| 0.20 | Y | 0.1554 | 0.1554 | pass_clean |  |
| 0.30 | Y | 0.1948 | 0.1948 | admit_near_floor |  |
| 0.32 | Y | 0.1996 | 0.1996 | admit_near_floor |  |
| 0.33 | Y | 0.2016 | 0.2016 | admit_e_floor | M30 |
| 0.35 | Y | 0.2050 | 0.2050 | admit_e_floor | M21 |
| 0.40 | Y | 0.2096 | 0.2096 | admit_e_floor |  |
| 0.50 | Y | 0.2044 | 0.2044 | admit_e_floor |  |
| 0.55 | Y | 0.1959 | 0.1959 | admit_near_floor |  |
| 0.58 | Y | 0.1892 | 0.1892 | admit_near_floor |  |
| 0.60 | Y | 0.1843 | 0.1843 | admit_near_floor |  |
| 0.62 | Y | 0.1790 | 0.1790 | admit_near_floor |  |
| 0.65 | Y | 0.1703 | 0.1703 | admit_near_floor |  |
| 0.68 | N | 0.1610 | 0.6500 | refuse_raw_pole |  |
| 0.70 | N | 0.1544 | 0.6500 | refuse_raw_pole |  |
| 0.75 | N | 0.1372 | 0.6500 | refuse_raw_pole |  |
| 0.80 | N | 0.1189 | 0.6500 | refuse_raw_pole |  |
| 0.85 | N | 0.1000 | 0.6500 | refuse_raw_pole | M28 |
| 1.00 | N | 0.0421 | 0.6500 | refuse_raw_pole |  |

**Cliffs:**
- **ê-floor (analytic):** eoc **0.32 pass / 0.33 fail** (M30 edge; ana_lr 0.1996→0.2016)
- **ê-floor (emp shared):** eoc **0.32 already 0/3** (mean_lr≈0.205) — residual overshoots analytic
- **analytic valley:** eoc∈[0.55, 0.65] admit + ana_lr&lt;0.20 (under gate) before refuse
- **refuse onset:** eoc ≥ **0.68** → teach_lr jumps to **0.65** (raw poles); M28@0.85 in-band
- **M21 default 0.35** sits in admit ê-floor (ana_lr=0.20503)

### 3. Emp smoke at cliff anchors

| cell | PASS | exam | content | leak |
|---|:---:|---:|---:|---:|
| `CTRL_leftover` | 3/3 | 0.9304 | 0.9949 | 0.0003 |
| `M14_locked` | 0/3 | 0.1799 | 0.1799 | 0.1005 |
| `M20_locked` | 0/3 | -0.082 | -0.082 | 0.3838 |
| `M21pool_eoc0.32_pre_floor` | 0/3 | 0.7247 | 0.7247 | 0.205 |
| `M21pool_eoc0.33_M30_edge` | 0/3 | 0.7101 | 0.7101 | 0.2079 |
| `M21pool_eoc0.35_M21_default` | 0/3 | 0.6812 | 0.6812 | 0.2105 |
| `M21pool_eoc0.68_refuse_onset` | 0/3 | 1.0 | 1.1544 | 0.6568 |
| `M21pool_eoc0.85_M28_hot` | 0/3 | 1.0 | 1.1544 | 0.6568 |
| `M31_locked` | 0/3 | -0.1534 | -0.1534 | 0.2067 |
| `M14_eou0.0` | 3/3 | 0.9563 | 0.9955 | 0.0009 |
| `M14_eou0.15` | 0/3 | 0.1799 | 0.1799 | 0.1005 |
| `M14_eou0.3` | 0/3 | 0.1799 | 0.1799 | 0.1005 |
| `M26_locked` | 0/2 | 0.0904 | 0.0904 | 0.0503 |

### 4. M31 leftover-amp eoc (branch B vs A contrast)

| eoc | e_unused | admit | teach_lr | ana_lr | anchor |
|---:|---:|:---:|---:|---:|---|
| 0.0 | 1.0 | Y | 0.0000 | 0.0000 |  |
| 0.35 | 0.85 | Y | 0.1284 | 0.1284 |  |
| 0.55 | 0.7 | Y | 0.0954 | 0.0954 |  |
| 0.85 | 0.55 | Y | 0.0663 | 0.0663 |  |
| 1.15 | 0.45 | Y | 0.2036 | 0.2036 | M31 |
| 1.25 | 0.15 | Y | 0.3785 | 0.3785 | M20ish |

## Mechanism (shared spine)

1. Teacher `faithful_guard_e` subtracts along **declared_ê** (from YAML `e_on_*`).
2. Exam leak gate scores **pure `leak_e()`**.
3. When declared ≠ exam axis:
   - **A:** declared restates û/content → guard deletes content (leftover gate fails; û looks fine).
   - **B:** declared tilts off ê but guard **admits** → teacher target retains ê → **analytic floor** lr≈0.205.
   - **C:** declared so damaged guard **refuses** → raw poles → lr≈0.65.
4. n / cover / vic / per-row / couple **cannot** clear A/B/C (target/YAML geometry, not DoF).

## Clearance (prior digs — all NO for this family)

| mid | n2 | vic0@n1 | per_row | source |
|---|:---:|:---:|:---:|---|
| M14 | NO | NO | NO | m14_failmode |
| M20 | NO | NO | NO | m14_failmode / falsify |
| M21 | NO | NO | NO | m21_eoc_irreducible |
| M28 | — | NO | NO | batch4 / falsify_b4 |
| M30 | — | NO | knife 1/6 | m30_knife / falsify_b4 |
| M31 | — | — | NO | falsify_b4 |

## Verdict

family map: A=content_deleted(M14/M20/M26) B=ê-floor-admit(M21/M30/M31) C=refuse-raw(M28); cliffs eou 0→≥0.15; emp floor@0.32 / ana@0.33; refuse_on=0.68; suite_pass=True; n/vic/per_row clear? NO; recipe_change=NO; merge=NO

- **suite_pass = True**
- recipe_change=**NO**; merge_to_trainer=**NO**; No Music train
- Keep HARD_BITEs: M14, M20, M21, M26, M28, M30, M31
- Fix path: correct declared ê / eoc YAML — not recipe knobs

## Sources

- `m14_failmode_20260909`, `m21_eoc_leak_irreducible_20260909`, `music_to_toy_batch4_20260909`
- `m30_per_row_knife_20260909`, `per_row_falsify_batch4_20260909`
- this dig cliffs: admit→refuse + M31 leftover + eou 0.15 + emp anchors

JSON: `declare_lie_eoc_family_theory_20260909.json`
