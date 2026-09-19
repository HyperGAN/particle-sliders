# Studio LoRA merger screen — September 7, 2026

**Complete:** 18 new renders, six verified reused references, and all 24 CPU
measurements. The [final audit](audit.json) passed source/artifact hashes,
actual runtime host assignment, exact excerpts and all 48 audio URLs. The
studio is restored to two GPUs with its Keep settings preserved. See the
[results](results.md) and [listening gallery](../../eval/listen/studio-mergers-20260907/index.html).
[Human preferences](listening-feedback.json) are recorded for all six groups.
Ordinary E2.8 wins three, KnOTS-TIES two, and TIES one uncertain comparison.
KnOTS-TIES beats ordinary for female + pop on both songs; ordinary wins
country + indie rock twice; house + acoustic folk splits between them. Keep ordinary
as the default and investigate KnOTS-TIES selectively. See the qualifications
and full rankings in the results; these counts do not measure preference strength.

The [metric agreement audit](metric-agreement.md) compares the saved metrics
with these six rankings and the earlier energy rankings. Content Enjoyment
matches 13/18 merger comparisons, versus 12/18 for a fixed method order, but
only 4/9 earlier energy comparisons. Production Complexity matches 12/18 and
8/9 respectively. No metric is established as a reliable predictor or causal
explanation; the studies are small and share recordings.

The listener preferred ordinary equal mixing at energy 2.8 on all three pairs
in the [earlier gain study](../studio_mix_20260907/listening-feedback.json).
This screen tests whether nonlinear merging improves on that baseline.
It does not change the production studio or its energy default.

## Frozen design

Three pairs: female + pop, country + indie rock, house + acoustic folk.
Two fixtures: the previous evaluation row 2 / seed 101 and the reserved
row 3 / seed 303. Each group uses identical caption, lyrics, sampling seed
and pipeline settings. The caption remains neutral; combined-caption
references from the earlier study remain separately available.

Each group compares ordinary energy 2.8, TIES, and KnOTS-TIES in a fixed
randomized A/B/C order. Ordinary energy 2 is an optional reference.
Six recordings are reused only after matching source, checkpoint, prompt,
resolved component, seed and audio hashes. Eighteen new recordings are rendered
through the studio pipeline on physical GPU 1. Full recordings are retained,
with exact first-20-second listening excerpts. The studio's 30-second ending
grace remains enabled, so requested duration 20 is not a hard 20-second cap.

## Methods and limits

Ordinary mixing forms `1.4 * delta_1 + 1.4 * delta_2`, where each delta includes
the checkpoint's own alpha/rank. TIES retains approximately the largest 30%
of entries, elects signs by summed magnitude, and averages the nonzero updates
agreeing with each elected sign. Threshold ties follow the official kthvalue
convention. Zero-sign resolution also follows the reference implementation.

KnOTS horizontally concatenates the full updates, computes a shared left SVD
basis, then applies TIES to the magnitude-bearing singular-value/right-vector
blocks. Our thin QR/SVD factorization is algebraically equivalent to the dense
concatenation and tested against a dense FP64 reference. Singular values at or
below 1e-5 are dropped, matching the reference. Plain summation in this basis
reproduces ordinary addition and is therefore not a separate candidate.

Both methods use **per-projection trimming and sign resolution**, rather than
the papers' globally flattened model representation. Every resulting dense
projection is rescaled to the Frobenius norm of that projection's ordinary
energy-2.8 update. FP64 norm accumulation controls numerical error on large
matrices; artifacts remain FP32. This controls weight strength and layer
allocation, not perceived intensity. These are explicitly norm-matched,
per-projection adaptations of the methods. Density 30% is a screening point,
not an optimized setting or grounds to reject every possible variant.

The dense outputs are retained without low-rank recompression. All 144 target
projections must match the runtime hosts. The dedicated renderer installs an
in-process research merger; no production inference code is edited. It restores
exact pristine weights before each change and on a partial failure, adds the
FP32 update, then casts to the host dtype once, as the studio normally does.

Sources: [TIES paper](https://arxiv.org/abs/2306.01708),
[TIES implementation](https://github.com/prateeky2806/ties-merging/tree/44e7891fc84f3de7e4caa52664cd864ca3715e91),
[KnOTS paper](https://arxiv.org/abs/2410.19735), and
[KnOTS implementation](https://github.com/gstoica27/KnOTS/tree/baa35e85768b8841820d35edc3e41b25304248c5).
The algorithms here are independently implemented; no dependency installation
or third-party training process is required.

## Selection rule

Listening preference comes first: retain both requested characters, clear
words, coherent groove, natural voice and enjoyable music. Compare each pair
on both songs. CPU CLAP, aesthetics and lyric recognition are supplementary
diagnostics and cannot declare the winner. If an alternative consistently wins,
confirm across more seeds and mixture ratios before integrating a studio mode.
If results are mixed, retain ordinary addition and investigate pair-specific
strength or blending controls before adding merger complexity. Solo energy is
not inferred from pair preferences.

The recorded listening results support a focused KnOTS-TIES versus ordinary
follow-up for female + pop. They do not support replacing ordinary mixing
globally or routing methods automatically by pair name. The user has not yet
ranked the energy-2 references, so the second-song gain comparison remains open.

## Files and execution

Use `/home/mikkel/anaconda3/envs/minimax-music3/bin/python`. Run
`test_mergers.py` and `test_dense_runtime.py` before building artifacts.
`build_artifacts.py` streams one projection at a time; `build_screen.py` freezes
inputs and verified reuse; `render.py` records every attempt. `measure.py`
runs on CPU. `report.py` and `publish.py` expose the listening gallery through
the studio. `audit.py` verifies outputs and delivery. `restore_studio.py`
returns the studio to two GPUs only once its queue is idle and Keep is off.

Before the frozen run, a strict normalization assertion caught FP32 norm
accumulation error on a full-size projection. Switching norm accumulation to
FP64 resolved it; the failed build attempt is preserved in `build.log` and no
audio was rendered from those partial artifacts.
