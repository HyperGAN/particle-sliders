# Dual-arm transfer strategy — Music ← locked #94 (2026-09-09)

Host: pop-os `/ml2/music/sliders-conceptmod` @ SHA `435e873` (+ local adv CLI restore).
CPU transfer research only. **No Music LM GPU train.** nano-work-server /
ComfyUI :18888 / `/ml2/music/run_server.py` left alone.

Locked portable recipe (2D / Field3D): 1200 steps, `cover_weight=1.5`,
`teacher=faithful_guard_e`, FM off, `n_particles≤12`, `particle_l2=0.02`,
`b_cap=1`.

## Hard reject cite (tx ∩ guard)

Live trainer couples `--adv_arch tx` to the lyric UNI recipe only:

```text
# conceptmod/textsliders/train_lm_slider_music3.py
PLUS_NEU_LYRIC_RECIPES = frozenset({"faithful_plus_neu_lyric"})   # ~L206

# ~L2170–2176 (train setup, after argparse)
tx_on = adv_enabled and adv_arch == "tx"
if adv_arch == "tx" and recipe not in PLUS_NEU_LYRIC_RECIPES:
    raise SystemExit(
        "--adv_arch tx needs the lyric span (the only multi-position "
        "signal aligned across captions): use a lyric recipe "
        f"({sorted(PLUS_NEU_LYRIC_RECIPES)}), got {recipe}."
    )
```

Same gate for the optional second span head on MLP:

```text
# ~L2182–2186
if txfm_on and recipe not in PLUS_NEU_LYRIC_RECIPES:
    raise SystemExit(
        "--txfm_weight needs the lyric span: use a lyric recipe "
        f"({sorted(PLUS_NEU_LYRIC_RECIPES)}), got {recipe}."
    )
```

Therefore **one argv cannot** be both:

| arm | `--lm_target` | `--adv_arch` | exercises |
|---|---|---|---|
| **Leftover / #94** | `faithful_guard_e` | `mlp` | leftover ê gate + pole pin |
| **Listen / smoke** | `faithful_plus_neu_lyric` | `tx` | lyric span critic + lyrichold |

`faithful_guard_e` + `--adv_arch tx` → `SystemExit` at train start.

## Ablation evidence (CPU, Music-transferable subset)

`dual_arm_leftover_vs_cover_20260909.{py,json,md}` — FM=0, b_cap=1, steps=1200,
`n_particles=1` (proxies Music `--parts 0`; toy `n=0` breaks `ParticlePrior.sample`),
seeds {0,1,2}, sheet + Field3D leftover:

| cell | sheet pass | sheet kept / leak | f3d pass | f3d u_kept / leak |
|---|---|---|---|---|
| leftover_only (guard, cover=0) | 0/3 | 0.48 / ≤0.06 | 0/3 | 0.57 / ≤0.12 |
| cover_only (faithful, cover=1.5) | 0/3 | 0.99 / **0.23** | 0/3 | 1.00 / **0.46** |
| **locked** (guard+cover1.5) | **3/3** | **0.935 / ~0** | **3/3** | **0.997 / ~0** |
| neither | 0/3 | 0.71 / 0.52 | 0/3 | 0.56 / 0.65 |

**Read:** #94 needs **both** leftover gate and cover pin. Gate alone undershoots;
cover alone copies ê. Music listen arm (lyric+tx+lyrichold, no guard) is
structurally closer to **cover_only**. Guard+mlp with `pole_weight=0` is
**leftover_only**.

## Options

### Option A — two sequential Music runs

1. **Arm L (leftover):** `--lm_target faithful_guard_e --adv_arch mlp`
   + YAML `leak_*` + modest `--pole_weight` (cover analogue) + `FM=0`
   `adv_reg_coeff=κ=1` `--parts 0`.
2. **Arm T (listen):** `--lm_target faithful_plus_neu_lyric --adv_arch tx`
   + `--lyrichold_weight 1` + same FM/b_cap/parts knobs
   (`music_locked_recipe_smoke_20260909.sh`).

**What carries between arms (knobs / diagnostics only — not LoRA weights):**

| carries | does not carry |
|---|---|
| FM=0, b_cap c=κ=1, `--parts 0`, `gan_beta1=0`, `adv_weight=1` | LoRA / optimizer state (different teachers) |
| step-budget philosophy (prefer longer over short×high-pin) | leftover ê gate → lyric arm |
| reject list (no parts↔particles, no FM-on, no 800×cover3) | cover name: pole_weight (L) ≠ lyrichold (T) |
| listen / leak metrics as go/no-go for the *other* arm pin scale | claim that either arm alone is full #94 |

Use A as **operational dual smoke**: Arm L validates leftover geometry; Arm T
validates span listen. Do **not** merge checkpoints.

### Option B — one arm is the true #94 transfer

Map 2D leftover honestly:

| 2D / Field3D | Music Arm L |
|---|---|
| `faithful_guard_e` | `--lm_target faithful_guard_e` + declared `leak_*` |
| MLP critic (no span-TX in toy) | `--adv_arch mlp` |
| `cover_weight=1.5` | modest `--pole_weight` (start ~1; **unvalidated** listen) |
| `fm_weight=0` | `--fm_weight 0` |
| `b_cap=1` | `--adv_reg_coeff 1 --adv_reg_kappa 1` |
| ParticlePrior ≤12 / l2=0.02 | **`--parts 0` always** (no Music latent ParticlePrior) |
| steps=1200 | re-tune with listen; do not copy 1200 by name |

Arm T (lyric+tx) is a **separate listen product**, not the leftover recipe.
Calling Arm T "#94 transfer" would be a false lock (cover_only risk).

### Option C — code change to allow both (propose only)

**Do not implement this sprint** (Mikkel: transfer research; Ada merges PRs).
Possible designs for a later PR:

1. **Decouple arch from recipe:** allow `--adv_arch tx` whenever a verified
   lyric span exists on the batch, even if teacher is `faithful_guard_e`
   (guard poles + lyric tokens). Today the gate is recipe-set membership,
   not "has lyric span".
2. **Dual-teacher row:** poles from `faithful_guard_e`, span reals from lyric
   UNI; critic = `SpanTransformerD` on lyric + optional MLP on last-hidden.
3. **Docstring / note only (tiny OK):** comment next to the `SystemExit` that
   leftover #94 transfer uses mlp; tx is lyric-listen only — already partly
   in smoke script comments.

Reject shipping (1)/(2) without tests + Ada review. Tiny docstring optional.

## Recommendation

**Primary: B** — treat **guard + mlp + modest pole_weight + FM0 + b_cap1 + parts0**
as the true #94 → Music transfer target.

**Operational companion: A** — still run Arm T listen smoke for span/lyrichold
metrics, but label it listen-only; do not merge weights or claim #94 from Arm T
alone.

**C:** propose-only; no code change this track (except optional future docstring
beside the `SystemExit`).

Rationale: ablation shows gate∧cover both required; toy has no span-TX, so
mlp matches #94 critic; Music has no ParticlePrior → parts=0; cover analogue
on the leftover arm is **pole_weight**, not lyrichold.

## Reject list

1. **Single-argv `faithful_guard_e` + `--adv_arch tx`** — hard `SystemExit` (~L2171).
2. **Mapping `--parts` ↔ `n_particles` / `particle_l2`** — different objects; keep `--parts 0`.
3. **Inventing Music `--cover_weight`** without a design pass.
4. **Claiming `lyrichold_weight=1` ≡ `cover_weight=1.5`** without listen/metrics.
5. **Arm T alone as #94** — cover_only leak pattern (sheet leak≈0.23).
6. **Arm L with `pole_weight=0` as done** — leftover_only kept fail (sheet≈0.48).
7. **800×cover3.0 / short×high-pin** — Fire #4 false lock.
8. **FM on / raw FM** — locked off; Music FM lacks normalize.
9. **Music LM GPU train this track**; do not touch nano-work-server, ComfyUI
   :18888, `/ml2/music/run_server.py`.
10. **Implementing Option C** without Ada PR (research propose only).

## Suggested future GPU smokes (not run here)

```bash
# Arm L — true #94 transfer (needs leak_* YAML)
# CUDA_VISIBLE_DEVICES=<idle≠0> ... train_lm_slider_music3.py \
#   --lm_target faithful_guard_e --adv_arch mlp \
#   --adv_weight 1 --fm_weight 0 --adv_reg_coeff 1 --adv_reg_kappa 1 \
#   --pole_weight 1 --lyrichold_weight 0 --parts 0 --gan_beta1 0 ...

# Arm T — listen only (existing smoke)
# FM_WEIGHT=0 ADV_WEIGHT=1 MINERS=0 \
#   bash analysis/slider2d/notes/music_locked_recipe_smoke_20260909.sh <gpu> ...
```

## Paths

- This note: `analysis/slider2d/notes/dual_arm_transfer_strategy_20260909.md`
- Ablation: `analysis/slider2d/notes/dual_arm_leftover_vs_cover_20260909.{py,json,md}`
- Gaps: `analysis/slider2d/notes/transfer_gaps_post_cli_20260909.md`
- CLI restore: `analysis/slider2d/notes/music_adv_cli_restore_20260909.md`
- Smoke: `analysis/slider2d/notes/music_locked_recipe_smoke_20260909.sh`
- Cite: `conceptmod/textsliders/train_lm_slider_music3.py` L206, L2170–2176, L2182–2186
