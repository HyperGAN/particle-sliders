# Per-row residual deepen (NON-DEFAULT) — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 2138.4s. CPU only. **Locked recipe unchanged.**
Label: `NON_DEFAULT_deepen` — analysis-only; do **not** merge to trainer.

## 1) Couple weight band vs head_min_cos

| w | seeds | PASS | multi | bite | mean exam | leak max | head_min_cos | fail |
|---|---|:---:|:---:|:---:|---:|---:|---:|---|
| 0.0 | 3 | 3/3 | 3/3 | 3/3 | 0.9893 | 0.1198 | 0.5585 | [] |
| 0.1 | 3 | 3/3 | 3/3 | 3/3 | 0.9986 | 0.1116 | 0.5815 | [] |
| 0.3 | 3 | 3/3 | 3/3 | 3/3 | 0.9913 | 0.077 | 0.6216 | [] |
| 0.5 | 3 | 2/3 | 3/3 | 2/3 | 0.933 | 0.1002 | 0.6689 | [1] |
| 0.7 | 3 | 1/3 | 1/3 | 1/3 | 0.8532 | 0.0628 | 0.7686 | [0, 1] |
| 0.0 | 6 | 6/6 | 6/6 | 6/6 | 0.9894 | 0.132 | 0.5571 | [] |
| 0.1 | 6 | 6/6 | 6/6 | 6/6 | 0.9985 | 0.1116 | 0.581 | [] |
| 0.3 | 6 | 6/6 | 6/6 | 6/6 | 0.9928 | 0.1757 | 0.6186 | [] |
| 0.7 | 6 | 3/6 | 3/6 | 3/6 | 0.89 | 0.0628 | 0.7576 | [0, 1, 7] |

Soft edge: **w=0.5** smoke knife (pass/bite 2/3, multi still 3/3, head_min≈0.67); **w=0.7** collapses (full 3/6, head_min≈0.76). Prefer **w≤0.3**.

**Robust band (full seeds, multi+bite+exam, head_min_cos<0.85):** w ∈ [0.0, 0.3] → [{'w': 0.0, 'pass': '6/6', 'multi': '6/6', 'bite': '6/6', 'mean_head_min_cos': 0.5571, 'min_head_min_cos': 0.5547, 'mean_exam': 0.9894, 'leak_max': 0.132}, {'w': 0.1, 'pass': '6/6', 'multi': '6/6', 'bite': '6/6', 'mean_head_min_cos': 0.581, 'min_head_min_cos': 0.5771, 'mean_exam': 0.9985, 'leak_max': 0.1116}, {'w': 0.3, 'pass': '6/6', 'multi': '6/6', 'bite': '6/6', 'mean_head_min_cos': 0.6186, 'min_head_min_cos': 0.613, 'mean_exam': 0.9928, 'leak_max': 0.1757}]

## 2) Other Music→toy cells — does per-row save?

| cell | mid | shared multi | per_row multi | per_row pass | saves? | notes |
|---|---|:---:|:---:|:---:|:---:|---|
| `hold_e_lyric_mix` | M21 | 3/3 | 6/6 | 0/6 | NO | leak gate (~0.21); not DoF |
| `cross_axis_rows` | cross | 0/3 | 6/6 | 6/6 | YES | DoF / amp-mix |
| `amp_lie_leftover_declare` | M20 | 3/3 | 6/6 | 0/6 | NO | YAML lie — expect fail |

Saved by per-row: **['cross_axis_rows']**. Still fail: **['hold_e_lyric_mix', 'amp_lie_leftover_declare']**. amp_lie still fail (expected YAML): **True**.

## 3) Music-posture n=1 / cover=1.0 + per-row lyric

- shared: pass=0/3 multi=0/3 bite=0/3 fail=[0, 1, 2]
- per_row: pass=6/6 multi=6/6 bite=6/6 head_min_cos≈0.5497 fail=[]
- **clears?** True

## 4) Leftover / close smoke (per-row, locked)

- leftover: 3/3
- close: 3/3
- controls_ok: **True**

## 5) Music mapping sketch (propose-only, no Music train)

Per-row AdvResidual on the toy = one odd/even head per lyric row (or multipair). In lm_adv / Music this maps to multi-residual / per-span students: each lyric span (or caption-pair row) owns a residual head rather than one shared δ forced to cover hetero row_amps. Soft couple keeps a shared slider identity without collapsing to the shared DoF floor.

### Toy → Music

- **Toy:** row r / Field3D.odd(r) with hetero row_amps
  - **Music:** lyric span / multipair caption row with distinct ûĉê mix
- **Toy:** shared AdvResidual δ ≈ mean(a_r) — cover gate floor
  - **Music:** single student residual across all spans — Fire #20 limit
- **Toy:** MultiResidual heads[r].delta(scale)
  - **Music:** per-span / per-row residual bank; apply head matching active span
- **Toy:** coupling_weight on (1-cos) of heads in R3
  - **Music:** soft shared-identity loss across span heads (one slider personality)
- **Toy:** amp_lie YAML declared_e lie — per-row does NOT save
  - **Music:** bad leak / hold YAML still requires target fix, not more DoF
- **Toy:** hold_e_lyric_mix / cross_axis — DoF vs gate/mix bites
  - **Music:** test whether Music bite is amp-mix DoF or teacher/YAML

### lm_adv scaffold sketch

Scaffold only: AdvResidual → list[AdvResidual] keyed by lyric-span id (or row index in multipair batch). Cover loss sums per-span (neu_s + δ_s - teacher_s). Optional couple: mean pairwise (1 - cos(δ_s, δ_t)) in the held leftover subspace. Infer: pick head by active span or mean-pool heads. Do NOT change locked shared defaults until multi-seed Music GPU confirms.

### Risks

- Head fragmentation → multiple personalities (monitor head_min_cos)
- Over-coupling → shared floor returns (toy w≥1.0)
- YAML/teacher lies unchanged by DoF (amp_lie)
- VRAM × n_spans for per-row heads

## Verdict

**YES — couple robust band w∈[0.0, 0.3] clears lyric multi; per_row saves ['cross_axis_rows']; amp_lie expected_fail=True; music_posture_clears=True; controls_ok**

- recipe_change=NO; merge_to_trainer=NO; locked shared AdvResidual stays default
- Next: multi-seed Music GPU smoke with per-row student (later); keep couple in band

JSON: `per_row_residual_deepen_20260909.json`
