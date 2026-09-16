# Clone arms 4/5 (propose-only): anneal + g_interp_cap vs locked

CPU toy ablation only — leftover-gated caption poles plus a span/end
cloud (hidden-state deltas, **not** rendered audio). This page does not
claim Music 3 listen quality. The locked demo stays sample-point
`b_cap` / anneal `none`; the Music trainer argv (`ARM_B`) and the YuE2
locked recipe are untouched.

## Cards

- `locked_b_cap_none`: locked reference (not a proposal).
- `a4_bcap_delayed`: arm 4 — sample-point cap + ParticleGAN `delayed`
  anneal (center held 60%, then linear to 0).
- `a5_ginterp_none`: arm 5 — interp-path cap (`g_interp_cap`), no anneal.
- `a45_ginterp_delayed`: arms 4+5 — interp-path cap + delayed anneal
  (closest to the ParticleGAN high-dim form with schedule).

## Ablation table (same cells, same gates as the locked harness)

| card | compiled | exam_score | field2d | sheet leftover | sheet gender | exam div/close/unused |
|---|---|---|---|---|---|---|
| `locked_b_cap_none` (b_cap/none) | **works** | 1.000 | FAIL (slider +0.596, leak +1.347) | PASS (leak -0.000, kept 0.930) | PASS (kept 0.995) | PASS/PASS/PASS (swing 1.000/1.000) |
| `a4_bcap_delayed` (b_cap/delayed) | **works** | 1.000 | FAIL (slider +0.596, leak +1.347) | PASS (leak -0.000, kept 0.932) | PASS (kept 0.996) | PASS/PASS/PASS (swing 1.000/1.000) |
| `a5_ginterp_none` (g_interp_cap/none) | **works** | 1.000 | FAIL (slider +0.597, leak +1.345) | PASS (leak -0.000, kept 0.931) | PASS (kept 0.996) | PASS/PASS/PASS (swing 1.000/1.000) |
| `a45_ginterp_delayed` (g_interp_cap/delayed) | **works** | 1.000 | FAIL (slider +0.597, leak +1.345) | PASS (leak -0.000, kept 0.932) | PASS (kept 0.996) | PASS/PASS/PASS (swing 1.000/1.000) |

## Verdict

**HOLD** — winning card: `none (tie vs locked)`.

- KEEP: a propose card beats or ties locked on every cell with no new
  failure mode (candidate to graduate toward a gated follow-up).
- HOLD: mixed vs locked (wins some cells, loses or ties others) — keep
  propose-only, needs a follow-up question answered first.
- DROP: loses to locked outright — do not pursue.

## Honesty notes

- The `field2d` column FAILs identically for all four cards *including*
  the locked reference: the harness runs the gated teacher on unpinned
  pairs there (`with_attrs=False`), and unpinned poles copy even
  leftover (leak ~1.3, predicted by `train_lm_adv`'s own docstring).
  The pinned-pairs field2d gate still passes and is pinned by
  `tests/test_lm_2d_adv.py::test_field2d_tracks_the_slider_without_gender_leak`;
  the propose-only CPU tests re-check finiteness + reporting (not the
  gate) on this config so the ablation comparison stays fair.
- `linear` anneal is wired through `make_grad_regularizer` and
  unit-tested (`tests/test_lm_adv_anneal.py`); the table uses `delayed`
  because the 60%-hold schedule is the ParticleGAN-faithful one.
- Honesty: toy hidden-state deltas only. No MiniMax weights, no audio
  render, no GPU. Live default stays `--lm_target v9`.

