# Per-row / multi-residual explore (NON-DEFAULT) — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 875.3s. CPU only. **Locked recipe unchanged.**
Label: `NON_DEFAULT_explore` — analysis-only; do **not** merge to trainer without multi-seed Music.

## Why shared AdvResidual fails (geometry)

Shared AdvResidual learns one δ(+1)=w_odd (+ even). Cover wants δ ≈ a_r = odd(row) for EVERY row simultaneously. When row_amps differ, {a_r} are distinct vectors in span{û,ĉ,ê}; the MSE-optimal single δ is their mean, and per-row relative error often exceeds the 0.20 cover gate. Curriculum / cover reweight cannot create degrees of freedom that do not exist. Per-row heads give each row its own w_odd_r ≈ a_r.

- M1 pairwise min cos(a_i,a_j) = **0.8285**, max rel L2 = **0.5721**
- If δ = mean(a_r): rel_err = [0.5934, 0.389, 0.3246, 0.2501, 0.2265] → rows_covered **0/5** (0.20 gate). MSE floor=0.219076
- Homo amps + staggered scales: rows_covered_if_mean **3/5** (scale hetero alone is already a partial bite)

### Scoring note (content_kept artifact)

`faithful_guard_e` teacher strips content/e relative to raw `odd(row)`. Shared δ compromise can *accidentally* restore content_kept vs raw odd (exam_cont≈1) while covering 0 rows. Per-row matches teacher → lower raw content_kept but high teacher-aligned cont. Primary Fire #20 metric here is **multi-row coverage + leftover û/leak** (`bite_cleared`); `exam_pass` uses teacher-aligned cont/swing.

## Ablation grid

| cell | mode | PASS | multi | bite | mean exam | leak max | rows_cov | fail |
|---|---|:---:|:---:|:---:|---:|---:|---:|---|
| `A_shared_lyric_m1` | shared | 0/3 | 0/3 | 0/3 | 0.9226 | 0.1829 | 0.0 | [0, 1, 2] |
| `A_shared_leftover` | shared | 3/3 | 3/3 | 3/3 | 0.9926 | 0.0002 | 1.0 | [] |
| `B_per_row_lyric_m1` | per_row | 3/3 | 3/3 | 3/3 | 0.9893 | 0.1198 | 5.0 | [] |
| `B_per_row_lyric_m1_full` | per_row | 6/6 | 6/6 | 6/6 | 0.9894 | 0.132 | 5.0 | [] |
| `B_per_row_coupled_w0.3_lyric` | per_row | 3/3 | 3/3 | 3/3 | 0.9913 | 0.077 | 5.0 | [] |
| `B_per_row_coupled_w1.0_lyric` | per_row | 0/3 | 0/3 | 0/3 | 0.7878 | 0.2786 | 0.0 | [0, 1, 2] |
| `B_per_row_leftover` | per_row | 3/3 | 3/3 | 3/3 | 0.9926 | 0.0002 | 1.0 | [] |
| `B_per_row_close` | per_row | 3/3 | 3/3 | 3/3 | 0.988 | 0.0152 | 3.0 | [] |
| `B_per_row_unused_e` | per_row | 3/3 | 3/3 | 3/3 | 0.9906 | 0.0013 | 3.0 | [] |
| `C_curriculum_shared_lyric` | curriculum_shared | 0/3 | 0/3 | 0/3 | 0.9533 | 0.1138 | 0.0 | [0, 1, 2] |
| `D_weighted_cover_shared_lyric` | weighted_cover_shared | 0/3 | 0/3 | 0/3 | 0.9255 | 0.1597 | 0.0 | [0, 1, 2] |
| `D_inv_weighted_cover_shared_lyric` | weighted_cover_shared | 0/3 | 0/3 | 0/3 | 0.9266 | 0.1696 | 0.0 | [0, 1, 2] |
| `E_scaled_shared_lyric` | scaled_shared | 0/3 | 0/3 | 0/3 | 0.9449 | 0.0698 | 2.33 | [0, 1, 2] |
| `E_scaled_shared_homo_amps` | scaled_shared | 3/3 | 3/3 | 3/3 | 1.0 | 0.0013 | 5.0 | [] |
| `E_shared_homo_amps_control` | shared | 0/3 | 0/3 | 0/3 | 0.9999 | 0.0012 | 3.0 | [0, 1, 2] |

## Family takeaways

- **A shared lyric:** multi=0/3 (confirm Fire #20 dead)
- **B per-row lyric:** multi=6/6 pass=6/6 bite=6/6
- **B controls:** leftover=3/3, close=3/3, unused_e=3/3
- **C curriculum shared:** multi=0/3 (helps? False)
- **D weighted cover shared:** multi=0/3 (helps? False) — gates unchanged
- **E scaled shared M1:** multi=0/3 (helps? False); homo+scales 3/3 vs shared ctrl 0/3 (helps? True)

## Verdict

**YES — per-row residual recovers multi-span (teacher-aligned exam) without breaking leftover/close smoke controls**

- per_row_promising=True; per_row_yes=True; controls_ok=True
- recipe_change=NO; merge_to_trainer=NO
- Next: if YES/PROMISING, multi-seed Music smoke with per-row student (GPU later); keep shared AdvResidual as locked default


## Caveats (depth)

- **Head fragmentation:** unconstrained per-row heads on M1 have mean `head_min_cos`≈**0.56** (acute but not one slider). Soft coupling `w=0.3` raises min cos≈**0.62** and still **3/3** multi; `w=1.0` collapses to shared-like **0/3** multi — coupling is a knife, not free.
- **Scaled shared** unlocks staggered **homogeneous** amps (3/3 vs shared 0/3 @ rows_cov 5 vs 3) but **cannot** fix M1 amp-*mix* (still 0/3, rows_cov≈2.3).
- **Curriculum / row-weighted cover** on shared: still **0/3** multi — confirms DoF / geometry limit, not optimization.
- Raw-odd `exam_pass` false-fails per-row (`content_kept`≈0.45) while teacher-aligned and `bite_cleared` are green — document, do not chase content_kept vs raw odd.
- **NON-DEFAULT:** do not merge to trainer; locked shared AdvResidual stays. Next = Music multi-seed with per-row student (GPU later).

JSON: `per_row_residual_explore_20260909.json`
