# Studio combinations and energy — September 7

**Completed:** all 30 new recordings and CPU measurements passed the final audit.
Rendering took 33.06 minutes through the existing studio, with 16 clips on GPU 0
and 14 on GPU 1. All 60 audio URLs support byte-range playback, and all 30
generator log entries match the frozen adapter multipliers, seed and device.
The studio is idle on both workers and its prior settings were preserved.

**Metric selection:** following the user clarification, mean Content Enjoyment
across both seeds selects the best tested setting per pair. See the
[selected presets and tradeoffs](metric-selection.md),
[metric-selected results page](http://192.168.1.90:7860/studio-combo-energy-20260907/selected.html),
and [preset data](selected-presets.json). Listening is optional. The decision rule
was adopted after the sweep; it is not a fresh-seed validation or a production default.

Read the [results and per-seed changes](results.md), [full table](results.csv),
and [audit](audit.json).

User-authorized screen on both existing studio GPU workers. Ordinary addition,
the three previously heard pairs, and two fresh seeds (707 and 909). The existing
electronic song fixture is held fixed; this tests seed variation, not new lyrics.

Thirty new recordings: three pairs × two seeds × five settings. At equal balance,
test LM energies 2.4, 2.8, 3.2. At energy 2.8, test balances 40/60, 50/50, 60/40.
The shared center is rendered once. This small cross-shaped design does not test
energy/balance interactions, three-slider blends, alternative mergers or solos.

The production API freezes the sheet, seed, controls and energy per request.
It uses both already loaded workers, records the actual device, retains every
cut in the studio library, and archives full WAVs plus first-20-second excerpts.
The studio adds its normal 30-second ending grace to the 20-second target.
Remembered studio settings are restored immediately after submission; no
service restart, installed weights or production defaults are changed.

[Listen](http://192.168.1.90:7860/studio-combo-energy-20260907/).
The energy and balance sections contain six blind triplets each. The same center
recording appears in both. The private blind key is deterministic and no metric
ranking appears within the blind groups. The separate metric page presents the
selected settings, their scores, and reference audio.

Existing CPU measurements report CE/PQ/PC/CU, each requested concept margin,
and lyric diagnostics separately. Scores describe the first 20 seconds and do
choose numerical winners by mean CE. No compound metric is fitted. In particular,
complexity is not inherently better, and sparse intros can have poor ASR scores.

Reproduce with the `minimax-music3` environment: freeze using `build_screen.py`,
queue/archive using `render.py`, measure using `measure.py`, and inspect the saved
`screen.json`, `renders.json`, `measurements.json`, report and final audit.
Run `select_by_metric.py` to reproduce the preset selections and scored results page.
