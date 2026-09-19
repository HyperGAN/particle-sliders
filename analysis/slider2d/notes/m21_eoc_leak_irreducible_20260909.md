# M21 e_on_content leak irreducible — mechanistic dig — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 737.5s. CPU only. **No Music train.**
Label: `MECH_M21_eoc_irreducible` — analysis-only; **locked recipe unchanged; merge=NO.**

## Mechanism (closed form)

M21 `hold_e_lyric_mix`: homogeneous leftover + `e_on_content=0.35` so
`declared_e = 0.35·ĉ + 0.85·ê` (tilted off pure ê).

`faithful_guard_e` / `faithful_sub_e` subtracts `(a·ê̂_⊥)ê̂_⊥` from odd `a`.
With tilted ê̂, the **teacher target itself** retains an ê component:

- Analytic â ≈ `(1.0 û, 0.497929 ĉ, -0.20503 ê)`
- **leak_ratio floor = 0.20503** (gate ≤0.20 → FAIL)
- Teacher actual: on_e=-0.184527, lr=0.20503 (abs err lr=0.0; match=True)

Exam `leak_ratio` uses **pure `leak_e()`**, while training subtracts along
**tilted `declared_e`**. Perfect residual cover reproduces the teacher floor.
Per-row cannot help: rows are homogeneous and share the same tilted teacher.

## Analytic eoc sweep (content=0.85, leak=0.65, e_unused=0.85)

| eoc | â_e | leak_ratio | pass_leak | content_kept |
|---:|---:|---:|:---:|---:|
| 0.00 | 0.0000 | 0.0000 | Y | 1.0000 |
| 0.05 | -0.0476 | 0.0476 | Y | 0.9517 |
| 0.10 | -0.0898 | 0.0898 | Y | 0.8976 |
| 0.15 | -0.1258 | 0.1258 | Y | 0.8389 |
| 0.20 | -0.1554 | 0.1554 | Y | 0.7770 |
| 0.25 | -0.1783 | 0.1783 | Y | 0.7134 |
| 0.30 | -0.1948 | 0.1948 | Y | 0.6492 |
| 0.35 | -0.2050 | 0.2050 | N | 0.5858 |
| 0.40 | -0.2096 | 0.2096 | N | 0.5241 |
| 0.50 | -0.2044 | 0.2044 | N | 0.4087 |
| 0.70 | -0.1544 | 0.1544 | Y | 0.2206 |
| 1.00 | -0.0421 | 0.0421 | Y | 0.0421 |

**Fail band (analytic):** eoc ∈ [0.35, 0.4, 0.5] (M21 default 0.35 sits in-band).

## Analytic amp sweep @ eoc=0.35

| content | leak | â_e | leak_ratio | pass_leak |
|---:|---:|---:|---:|:---:|
| 0.55 | 0.45 | -0.1284 | 0.1284 | Y |
| 0.70 | 0.50 | -0.1740 | 0.1740 | Y |
| 0.85 | 0.65 | -0.2050 | 0.2050 | N |
| 0.85 | 0.45 | -0.2340 | 0.2340 | N |
| 0.55 | 0.65 | -0.0994 | 0.0994 | Y |
| 1.00 | 0.80 | -0.2361 | 0.2361 | N |
| 0.40 | 0.30 | -0.0973 | 0.0973 | Y |

## Empirical shared (locked declared_e)

| metric | value |
|---|---|
| pass_leak | **0/6** |
| mean leak_ratio | 0.210619 |
| mean on_e | -0.215691 |
| mean pole_rel_err | 0.102873 |
| residual≈analytic floor | **True** |

## Empirical eoc smoke (shared)

| eoc | emp mean_lr | analytic_lr | pass_leak |
|---:|---:|---:|---|
| 0.0 | 0.001185 | 0.0 | 3/3 |
| 0.2 | 0.159441 | 0.15541 | 3/3 |
| 0.35 | 0.21052 | 0.20503 | 0/3 |
| 0.7 | 0.656844 | 0.154433 | 0/3 |

## Per-row (NON_DEFAULT scaffold)

| mode | pass | multi | leak_max | exam | fail |
|---|:---:|:---:|---:|---:|---|
| per_row w=0 | 0/6 | 6/6 | 0.207 | 0.9921 | [0, 1, 2, 3, 7, 42] |
| per_row w=0.3 | 0/3 | 3/3 | 0.208 | 0.989 | [0, 1, 2] |

## Diagnostic: train on pure ê (ignore eoc) — geom ablation

- fit pass_leak: **6/6** mean_lr=0.001053
- score_adv pass: **3/3**
- Proves: failure is **teacher leak_dir tilt**, not odd amps / cover / particles.
- Do **not** delete eoc from the cell (that soft-deletes the Music symptom).

## Controls

- leftover CTRL: PASS=True ([{'seed': 0, 'pass': True, 'leak_ratio': 0.0002964590950968146, 'u_kept': 0.9919547339426048}, {'seed': 1, 'pass': True, 'leak_ratio': 0.0003990660953011316, 'u_kept': 0.991327335457973}, {'seed': 2, 'pass': True, 'leak_ratio': 0.00014633921923211518, 'u_kept': 0.9922678967431231}])
- M20 amp_lie still bites: True ([{'seed': 0, 'pass': False, 'pass_leak': False, 'leak_ratio': 0.38444662761070914}, {'seed': 1, 'pass': False, 'pass_leak': False, 'leak_ratio': 0.38426772098969547}, {'seed': 2, 'pass': False, 'pass_leak': False, 'leak_ratio': 0.38276560418888067}])


## Nuance: eoc=0.7 empirical ≠ analytic subtract floor

At eoc=0.7 emp_lr≈0.66 while analytic subtract predicts ≈0.15. Cause: **`lm_blend_guard`
refuses** the subtract (axis too damaged) → teacher falls back to **raw poles** →
full geometry leak (0.65/1.0≈0.65). Two fail modes on the eoc axis:

| eoc band | teacher | fail mode |
|---|---|---|
| ~0.35–0.5 (M21 default) | guard **admits** tilted subtract | **ê floor** lr≈0.205 (this dig) |
| high (e.g. 0.7+) | guard **refuses** | **raw-pole leak** lr≈leak/slider |

Also: teacher `leak_ratio` vs **declared_êhat** = **0.0** (δ ⊥ declared by construction)
while vs **pure ê** = 0.205 — smoking-gun axis mismatch. Scoring on declared axis would
false-pass; exam correctly uses pure `leak_e()`.

## Verdict

**YES — M21 leak is teacher-axis geometry floor (declared_e tilt), not shared-DoF / optimization; per-row cannot clear; train-pure-ê clears**

- recipe_change=**NO**; merge_to_trainer=**NO**
- M21 remains HARD BOUNDARY under shared **and** per-row
- Driver confirmed: `e_on_content>0` tilts declared_e → teacher ê floor > 0.20
- Distinct from M1/M24/M27 DoF bites (per-row clears those, not M21)

JSON: `m21_eoc_leak_irreducible_20260909.json`
