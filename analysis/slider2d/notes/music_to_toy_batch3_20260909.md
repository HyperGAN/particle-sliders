# Music→toy batch3 — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 951.4s. CPU only. No Music train.

## New cells (M24–M27; avoided lyric_span / cross_axis / dual_arm / close_live / M20–M23)

| ID | Cell | Music symptom |
|---|---|---|
| M24 | `content_leak_flip_rows` | û-primary + content↔leak dominance flip mid-caption |
| M25 | `lyric_neu_heavy_gate` | heavy lyric neu vs leftover unused-ê gate |
| M26 | `declare_split_three` | ambiguous declared ê split û/content/unused |
| M27 | `scale_descent_homo` | homo amps + descending scales (traj/outro reverse) |

## Bite / no-bite table (new cells + leftover CTRL)

| cell | mid | posture | pass | exam_mean | leak_max | fail_seeds | verdict |
|---|---|---|:---:|---:|---:|---|---|
| `content_leak_flip_rows` | M24 | locked | **0/6** | 0.6382 | 0.0016 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `content_leak_flip_rows` | M24 | music | **0/6** | 0.6582 | 0.0093 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `lyric_neu_heavy_gate` | M25 | locked | **6/6** | 0.7516 | 0.0016 | [] | **NO_BITE** |
| `lyric_neu_heavy_gate` | M25 | music | **6/6** | 0.757 | 0.0261 | [] | **NO_BITE** |
| `declare_split_three` | M26 | locked | **0/6** | 0.0914 | 0.0511 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `declare_split_three` | M26 | music | **0/6** | 0.0916 | 0.0601 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `scale_descent_homo` | M27 | locked | **0/6** | 0.7044 | 0.0027 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `scale_descent_homo` | M27 | music | **0/6** | 0.7251 | 0.0113 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `leftover_regression` | CTRL | locked | **3/3** | 0.9304 | 0.0004 | [] | **NO_BITE** |

## M21 deepen (`hold_e_lyric_mix`) — portable knobs

| probe | pass | exam_mean | leak_max | fail_seeds | verdict |
|---|:---:|---:|---:|---|---|
| `baseline_locked` | **0/3** | 0.6812 | 0.211 | [0, 1, 2] | **HARD_BITE** |
| `baseline_music` | **0/3** | 0.6847 | 0.2205 | [0, 1, 2] | **HARD_BITE** |
| `n1_c1.5` | **0/3** | 0.6832 | 0.2129 | [0, 1, 2] | **HARD_BITE** |
| `n2_c1.5` | **0/3** | 0.6775 | 0.2101 | [0, 1, 2] | **HARD_BITE** |
| `n4_c1.5` | **0/3** | 0.6798 | 0.2087 | [0, 1, 2] | **HARD_BITE** |
| `n12_c1.5` | **0/3** | 0.6812 | 0.211 | [0, 1, 2] | **HARD_BITE** |
| `n12_c1.0` | **0/3** | 0.6805 | 0.2159 | [0, 1, 2] | **HARD_BITE** |
| `n12_c2.0` | **0/3** | 0.6807 | 0.2094 | [0, 1, 2] | **HARD_BITE** |
| `steps1600_c1.5_n12` | **0/3** | 0.6826 | 0.2082 | [0, 1, 2] | **HARD_BITE** |
| `vic0_c1.5_n12` | **0/3** | 0.6776 | 0.2106 | [0, 1, 2] | **HARD_BITE** |
| `vic0_music_n1` | **0/3** | 0.6806 | 0.2111 | [0, 1, 2] | **HARD_BITE** |
| `eoc0_locked` | **3/3** | 0.8962 | 0.0012 | [] | **NO_BITE** |
| `eoc0_music` | **3/3** | 0.8946 | 0.0085 | [] | **NO_BITE** |
| `soft_mix_eoc035` | **0/3** | 0.6618 | 0.1342 | [0, 1, 2] | **HARD_BITE** |
| `hot_mix_eoc0` | **3/3** | 0.8962 | 0.0012 | [] | **NO_BITE** |
| `false_lock_800_c3` | **0/3** | 0.6644 | 0.2188 | [0, 1, 2] | **HARD_BITE** |

**Recipe-portable recoveries (n / cover / steps / vic / false-lock 800×c3):** **NONE** — all stay HARD_BITE (leak≈0.21).

**Geom isolation (not a recipe knob):** `e_on_content=0` recovers 3/3 (locked+music); `hot_mix` (content=0.85,leak=0.65) + eoc0 also 3/3; `soft_mix` + eoc=0.35 still HARD_BITE (leak≈0.13).

**Verdict:** **recipe hard boundary** (like lyric_span). Driver = `e_on_content>0` on lyric-pool mix — do **not** delete eoc from the cell (that soft-deletes the Music symptom). No locked-recipe chase.

## Registry

Registered in `CELLS_3D` this fire:

- `content_leak_flip_rows` (M24)
- `lyric_neu_heavy_gate` (M25)
- `declare_split_three` (M26)
- `scale_descent_homo` (M27)

## Verdict

- Recipe change: **NO** (keep FM0, l2=0.02, locked 1200+c1.5).
- Leftover regression: **3/3** (must stay NO_BITE).
- M21: **recipe hard boundary** (n/cover/steps/vic/false-lock never recover); driver=`e_on_content>0`.
- False-lock 800×c3.0: still HARD_BITE here; never adopt.

JSON: `music_to_toy_batch3_20260909.json`

## Interpretation

- **M24 `content_leak_flip_rows`**: locked HARD_BITE 0/6; music HARD_BITE 0/6 (exam_locked=0.6382, leak_max_music=0.0093)
- **M25 `lyric_neu_heavy_gate`**: locked NO_BITE 6/6; music NO_BITE 6/6 (exam_locked=0.7516, leak_max_music=0.0261)
- **M26 `declare_split_three`**: locked HARD_BITE 0/6; music HARD_BITE 0/6 (exam_locked=0.0914, leak_max_music=0.0601)
- **M27 `scale_descent_homo`**: locked HARD_BITE 0/6; music HARD_BITE 0/6 (exam_locked=0.7044, leak_max_music=0.0113)
- **M21 deepen**: recipe_hard_boundary=True (n/cover/steps/vic/false-lock all bite); driver=e_on_content>0 (eoc0 / hot_mix_eoc0 recover — geom ablation only, keep cell eoc>0)
- **leftover CTRL**: 3/3 exam=0.9304

### Wins (Music bug now fails in-toy)

| Cell | Bite? | Why it matters |
|---|---|---|
| **M24 content_leak_flip_rows** | **HARD** 0/6 both | û-primary content↔leak flip — swing/content exam fail (leak tiny); mid-caption attr swap bites without full cross_axis |
| **M25 lyric_neu_heavy_gate** | **NO** 6/6 both | Heavy lyric neu alone does **not** break leftover gate — encoding: neu magnitude ≠ hold-ê mix |
| **M26 declare_split_three** | **HARD** 0/6 both | Ambiguous YAML ê split — content survival / exam collapse (exam≈0.09); milder than amp_lie but still negative control |
| **M27 scale_descent_homo** | **HARD** 0/6 both | Descending scales → multi_row 3/5 (u≈0.70); traj/outro reverse bites like ascending stagger soft-fail |
| **M21 deepen** | recipe **HARD BOUNDARY** | No portable recipe knob recovers; only e_on_content=0 clears — document driver, keep stressor |
| **leftover CTRL** | **NO** 3/3 | No recipe damage |

**No recipe change.** Keep FM0, particle_l2=0.02, locked 1200+c1.5 n≤12.
