# Cheap score-guided choices during creation

Feasibility check following the user's request for a cheap way to improve the
score during song creation. No new music was generated and no studio code,
weights or defaults were changed.

## Measured scorer cost

The pinned Audiobox Aesthetics model scored a prepared 10-second mono 16 kHz
excerpt on four CPU threads. Median warm forward time was **0.567 seconds** over
five calls, with a range of 0.553–0.589 seconds. Loading the cached model took
5.488 seconds; the first forward took 0.994 seconds. See `scorer-timing.json`.

The forward returns all four aesthetics heads; the proposed selection uses CE.
This excludes imports, resampling, file I/O, CLAP and ASR. Keeping the scorer
resident avoids repeated loading. The scorer cost is not the cost of generating
the excerpt.

## Early selection check using existing audio

For each of the 15 pair/settings combinations in the completed 30-clip sweep,
choose between seeds 707 and 909 using first-10-second CE. Evaluate that choice
on the disjoint next 10 seconds using the already cached scorer windows.

- The selected seed also has higher CE in the next window in **11/15** comparisons.
- Mean next-window CE gain over choosing one of the two seeds uniformly is **+0.086**.
- Country + indie rock: 5/5 agreements, +0.088 mean next-window gain.
- Female + pop: 2/5 agreements, +0.001 mean next-window gain.
- House + acoustic folk: 4/5 agreements, +0.169 mean next-window gain.

See `early-selection.json` and `source-windows.json`. First-window selection
and evaluation use separate audio intervals to avoid the part/whole correlation
of testing the first 10 seconds against a score that includes those same seconds.
These are dependent comparisons across five settings of three pairs on one song,
using the same two seeds. This is a feasibility signal, not a full-song or
fresh-song validation. The first 20 seconds share the normalization gain used
in the earlier study.

## Proposed runtime behavior

Generate two short beginnings with the same sheet and controls, score their
audio by CE, then resume only the selected state. Keep the choice bounded to
two candidates. Reuse the winning semantic state and rendered prefix.
Listening is not required for this numerical selection.

The extra generation would be roughly one additional short beginning plus the
context needed for overlapping audio chunks and scorer execution. Actual runtime
overhead must be measured after implementing the state handoff.

The installed Diffusers pipeline currently completes semantic generation before
denoising and decoding audio. Its semantic block retains frame hidden states but
does not expose the autoregressive KV cache as a resumable output. Therefore this
is not an existing studio flag. Implementing it requires a semantic pause/resume
state, retained random-generator state, and acoustic chunk/overlap handling.
Preview scoring must not consume the semantic RNG or discard the context needed
to continue the chosen prefix. Re-running a full song with the same seed is not
equivalent to resuming: the current acoustic noise is sampled after the full
semantic pass from the same generator.

Compare early selection with ordinary single-seed generation on fresh full songs,
using CE on later, non-selected sections as well as the full output and recording
actual added generation time. Existing metric-selected presets provide a separate
way to reuse the sweep's settings without generating additional candidates.

Model reference: [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics).
The pipeline inspection used the installed `diffusers/modular_pipelines/minimax_music3`
code, especially `encoders.py`, `before_denoise.py`, `denoise.py` and `decoders.py`.
