# Music→toy new stressors batch2 — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 659.4s. CPU only. No Music train.

## New cells (avoided lyric_span / cross_axis_rows / dual_arm / close_live_noise)

| ID | Cell | Music symptom |
|---|---|---|
| M20 | `amp_lie_leftover_declare` | declared YAML content-axis amplitude lie on leftover geom |
| M21 | `hold_e_lyric_mix` | content↔leftover mix ≈ hold-ê lyric pool |
| M22 | `stagger_mild_cross` | staggered scales + mild cross-axis (soft multipair) |
| M23 | `multipair_corr_seed` | multi-pair R³ correlated seeds |

Recipe probes: `close_fm_tempt` (FM0 vs 0.5), `particle_l2_extreme_n1` (l2∈{0,0.02,0.2} @ n=1).

## Bite / no-bite table

| cell | mid | posture | pass | exam_mean | leak_max | fail_seeds | verdict |
|---|---|---|:---:|---:|---:|---|---|
| `amp_lie_leftover_declare` | M20 | locked | **0/6** | -0.0826 | 0.3859 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `amp_lie_leftover_declare` | M20 | music | **0/6** | -0.0808 | 0.3971 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `hold_e_lyric_mix` | M21 | locked | **0/6** | 0.6807 | 0.2111 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `hold_e_lyric_mix` | M21 | music | **0/6** | 0.6844 | 0.2205 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `stagger_mild_cross` | M22 | locked | **6/6** | 0.9751 | 0.0019 | [] | **NO_BITE** |
| `stagger_mild_cross` | M22 | music | **4/6** | 0.9742 | 0.0118 | [2, 7] | **KNIFE** |
| `multipair_corr_seed` | M23 | locked | **6/6** | 0.9456 | 0.0299 | [] | **NO_BITE** |
| `multipair_corr_seed` | M23 | music | **6/6** | 0.9455 | 0.0345 | [] | **NO_BITE** |
| `close_fm0.0_locked` | FM_tempt | locked | **3/3** | 0.9799 | 0.0129 | [] | **NO_BITE** |
| `close_fm0.0_music` | FM_tempt | music | **2/3** | 0.7314 | 0.2024 | [0] | **KNIFE** |
| `close_fm0.5_locked` | FM_tempt | locked | **3/3** | 0.9808 | 0.0103 | [] | **NO_BITE** |
| `close_fm0.5_music` | FM_tempt | music | **2/3** | 0.7277 | 0.1654 | [0] | **KNIFE** |
| `leftover_n1_l2_0.0` | l2_extreme | music | **3/3** | 0.9311 | 0.0083 | [] | **NO_BITE** |
| `close_n1_l2_0.0` | l2_extreme | music | **2/3** | 0.7277 | 0.2225 | [0] | **KNIFE** |
| `leftover_n1_l2_0.02` | l2_extreme | music | **3/3** | 0.9321 | 0.0097 | [] | **NO_BITE** |
| `close_n1_l2_0.02` | l2_extreme | music | **2/3** | 0.7314 | 0.2024 | [0] | **KNIFE** |
| `leftover_n1_l2_0.2` | l2_extreme | music | **3/3** | 0.9286 | 0.0083 | [] | **NO_BITE** |
| `close_n1_l2_0.2` | l2_extreme | music | **2/3** | 0.7353 | 0.089 | [0] | **KNIFE** |
| `leftover_regression` | CTRL | locked | **3/3** | 0.9304 | 0.0004 | [] | **NO_BITE** |

## Registry

Registered in `CELLS_3D` (clean + tested this fire):

- `amp_lie_leftover_declare` (M20)
- `hold_e_lyric_mix` (M21)
- `stagger_mild_cross` (M22)
- `multipair_corr_seed` (M23)

## Verdict

- Recipe change: **NO** (keep FM0, l2=0.02, locked 1200+c1.5).
- Leftover regression must stay NO_BITE.
- FM temptation: document flat/tempt; do not enable FM.
- particle_l2 extremes @ n=1: document; keep default 0.02.

JSON: `music_to_toy_new_stressors_batch2_20260909.json`

## Interpretation

| Cell | Bite? | Why it matters |
|---|---|---|
| **M20 amp_lie_leftover_declare** | **HARD** 0/6 both | Content-axis YAML lie on leftover geom — expected negative control (swing/leak); distinct from `e_on_u_declare_lie` |
| **M21 hold_e_lyric_mix** | **HARD** 0/6 both | hold-ê lyric-pool mix (content↔ê) breaks leftover gate (leak≈0.21) despite multi-row covered — Music pool mix bites without hetero `row_amps` |
| **M22 stagger_mild_cross** | locked **NO**; music **KNIFE** 4/6 | Soft multipair survives locked; Music-posture n=1 flakes seeds {2,7} via multi_row — between homo pass and cross_axis hard fail |
| **M23 multipair_corr_seed** | **NO** 6/6 both | Correlated multi-pair seeds are seed-stable under locked+music — no new knife beyond close family |
| **close FM tempt** | locked NO; music KNIFE seed0 | FM0.5 ≈ FM0 under close — temptation flat; seed0 û-undershoot unchanged (do not enable FM) |
| **particle_l2 @ n=1** | leftover NO; close KNIFE seed0 | l2∈{0,0.02,0.2} flat on leftover; close seed0 knife persists — keep default 0.02 |
| **leftover regression** | **NO** 3/3 | No recipe damage |

**Wins (Music bug now fails in-toy):** M20 amplitude-lie declare; M21 hold-ê lyric-pool mix.

**No recipe change.** Keep FM0, particle_l2=0.02, locked 1200+c1.5 n≤12.
