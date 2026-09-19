# Fire #22 — Music-posture warn helper (2026-09-09)

## Goal
Scaffold an explicit, testable Music-posture warning when close-family Field3D cells
run under Music parts0 proxy (`n_particles==1`), without changing locked train defaults.

## Helper
`analysis.slider2d.field3d.music_close_posture_warn(cfg_or_n_particles, cell_kind=...) -> Optional[str]`

Warns when:
- `cell_kind` ∈ `{close, close_live_noise, tiny_slider_dom, close_with_leak}`
- AND `n_particles == 1`

Message cites harden path: prefer `n_particles>=2` OR multi-seed gate; optional
candidate `vicreg_weight=0@n=1` (not a silent default flip).

## Call site
`score_adv_field3d` attaches `music_close_posture_warn` to the result dict (None when safe).
Does not touch loss / AdvConfig defaults / CELLS_3D geometry.

## Tests
- `test_music_close_posture_warn_n1_close_family`
- `test_score_adv_field3d_emits_music_close_posture_warn_key`

## recipe_change
**NO**

## ping_user
**NO** — documents already-known Fire #21 harden path as an explicit warn; no new bite.

## Next
Optional: surface warn in Music→toy stressor runners / catalog dig scripts; keep
mandatory multi-seed gate for Music parts0 close in ops docs.
