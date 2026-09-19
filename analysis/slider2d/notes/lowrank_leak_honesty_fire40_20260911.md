# lowrank leak-max honesty — Fire #40 — 2026-09-11

Host: box-cpu reanalysis of `lowrank_soft_continuum_20260910.json` @ `435e87363bf1`.
CPU only. **No Music GPU.** No new CELLS. Locked shared unchanged. merge=**NO**.

## Question

Does prior **ADOPT_analysis_lowrank_soft_continuum** (“full continuum soft under lowrank_k3”)
survive when soft-band / clear claims also require **`leak_max_rows ≤ EXAM_LEAK_LOCK (0.20)`**
instead of row0-only `pass_leak`?

## Root cause

`score_adv_field3d` computes `leak_ratio` / `pass_leak` from **row0 delta only**.
Portable `score_portable` sets `leftover_ok = pass_leak ∧ pass_u` for DoF kinds, so
`full_pass` can be True while per-row `leak_max_rows` is ≫ 0.20.
`lowrank_k3` on **cross_axis / M2** is the smoking gun: multi-row coverage ✓, row0 leak tiny,
non-row0 leak explodes.

## Stock anchors (from continuum dig)

| cell | mode | full_pass | honest (leak_max≤0.20) | mean leak_row0 | mean leak_max |
|---|---|:---:|:---:|---:|---:|
| M2_stock | shared | 0/3 | 0/3 | 0.3747 | 0.3747 |
| M2_stock | lowrank_k3 | 3/3 | 0/3 | 0.0093 | 2.5753 |
| M16_stock | shared | 0/3 | 0/3 | 0.0007 | 0.0007 |
| M16_stock | lowrank_k3 | 3/3 | 3/3 | 0.0027 | 0.0038 |
| CTRL leftover | lowrank_k3 | 3/3 | 3/3 | 0.0008 | 0.0008 |
| CTRL close | lowrank_k3 | 3/3 | 3/3 | 0.0097 | 0.0126 |

**M2_stock lowrank “3/3 clear” is a false clear** under leak_max honesty (honest **0/3**, leak_max≈2.57).
**M16_stock** stays honestly green under lowrank (scale/DoF path).

## Continuum cliffs under honest gate

| path | shared honest last_pass→first_bite | lowrank honest last_pass→first_bite | full continuum soft? | soft expands? |
|---|---|---|:---:|:---:|
| M22→M16 | 0.00→0.10 | **1.00 / —** | YES | YES |
| M23→M2 | 0.15→0.25 | **0.40→0.50 (leak_max)** | NO | YES |
| M22→M2 | 0.25→0.40 | **0.50→0.60 (leak_max)** | NO | YES |

Prior claim “all three paths full-continuum soft under lowrank” = **FALSE** once leak_max gates.
Soft bands still expand on all three; →M2 paths bite later via **leak_max**, not exam_dof.

## Verdict

**ADOPT_analysis_lowrank_leakmax_honesty** (refine prior soft-continuum ADOPT):

1. Analysis soft-band / clear claims **must** gate `leak_max_rows ≤ 0.20` (not row0-only `pass_leak`).
2. **lowrank_k3** still expands soft bands and honestly clears **scale (M16)** continuum end-to-end.
3. **lowrank_k3 does NOT honestly clear M2 / cross_axis** — DoF coverage without multi-row leak control.
4. Fire #39 “clears M2” under portable harness = **DoF-only**; revise leader language to **DoF✓ ∧ leak_max✗ on M2**.
5. recipe_change=**NO**; merge=**NO**; No Music GPU; No new CELLS.

JSON: `lowrank_leak_honesty_fire40_20260911.json`
