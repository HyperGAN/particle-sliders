# Fire #23 — Wire music_close_posture_warn into dig runners (2026-09-09)

- Host: box-cpu @ `435e873` (pop-os pytest in parallel; no Music GPU; no CloudAgent)
- Thread: Fire #22 next → surface posture warn in Music→toy dig runners
- Wall: 112.97s (steps=400 short smoke)
- recipe_change: NO

## What landed

1. `summarize_music_close_posture(rows)` in `field3d.py`
2. `analysis/slider2d/notes/music_posture_dig_util.py` — `annotate_dig_rows` / `posture_block_for_md`
3. `tests/test_music_to_toy_stressors.py` asserts warn @ close_live_noise n=1, clear @ n=2 / lyric / dual

Score path already emitted the key (Fire #22). This fire makes dig JSON/md aggregation + pytest smoke mandatory.

### Music close posture (Fire #22/#23)
- rows=36 warn=12 clear=24 missing_key=0
- **any_warn=YES** — Music parts0 close knife posture present
  - `close_n1_s0` cell=close n=1 seed=0
  - `close_n1_s1` cell=close n=1 seed=1
  - `close_n1_s2` cell=close n=1 seed=2
  - `close_live_noise_n1_s0` cell=close_live_noise n=1 seed=0
  - `close_live_noise_n1_s1` cell=close_live_noise n=1 seed=1
  - `close_live_noise_n1_s2` cell=close_live_noise n=1 seed=2
  - `close_with_leak_n1_s0` cell=close_with_leak n=1 seed=0
  - `close_with_leak_n1_s1` cell=close_with_leak n=1 seed=1
  - `close_with_leak_n1_s2` cell=close_with_leak n=1 seed=2
  - `tiny_slider_dom_n1_s0` cell=tiny_slider_dom n=1 seed=0
  - `tiny_slider_dom_n1_s1` cell=tiny_slider_dom n=1 seed=1
  - `tiny_slider_dom_n1_s2` cell=tiny_slider_dom n=1 seed=2

- expect_warn_all_ok=True (12/12)
- expect_clear_all_ok=True
- verdict: PASS — dig util + score key wire-up holds for close-family@n=1; clears @n=2 and non-close cells

## Music rule (unchanged)

- Close-family under Music parts0 (`n_particles=1`): **mandatory multi-seed gate** + prefer `n_particles>=2`
- Optional candidate: `vicreg_weight=0` @ n=1 (propose only; no silent default flip)
- Hard boundaries stand: M1 lyric_span_entangle; cross_axis_rows

## Next

- Retrofit older dig scripts to call `annotate_dig_rows` when touching close cells
- Sync `field3d.py` + music stressor tests to pop-os when convenient (`SYNC_TO_POPOS.md`)
- Keep locked recipe; no Music LM train
