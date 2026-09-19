# Lo-fi: same prompt, a fresh continuation each update

This experiment continues the exact original UNI16 Lo-fi step-600 state for
60 updates. The fixed and fresh arms use the first original training prompt,
batch size one, the original GAN v2 baseline losses, and the same restored
adapter, critic, and optimizer states. The user authorized both GPUs.

- Fixed arm, GPU 0: reuse the neutral base-model continuation sampled with seed 7.
- Fresh arm, GPU 1: use seeds 1000001 through 1000060, one new continuation per
  optimizer update. The prompt and lyrics stay identical. The adapter is disabled
  while sampling and computing base-model teachers. Histories are cached only
  for exact restart and auditing; the seed schedule never cycles.

The style discriminator and feature matching read lyric/prompt states before
the generated audio continuation. Causal attention means that resampling the
continuation does not directly diversify these style targets. It changes the
history used by the ending penalty. This experiment tests that narrower effect,
not training on a distribution of generated audio style states.

Each arm checks the original source hashes, exact source adapter tensors, and
exact fixed-history replay before updating weights. Fresh history hashes,
sample seeds, teacher differences, losses, and full resumable states are retained.
The isolated runtime preserves the original training implementation; production
training code and the studio registry are unchanged.

The initial launch exposed missing transitive imports in the isolated runtime.
No completed update was exported. The missing dependencies were copied, the
optimizer limit was checked in the isolated runtime, and both arms restarted
from the full original step-600 state. Details are in `audit/`.

GPU 0 pre-renders the parent, published, and fixed-arm references while GPU 1
continues the fresh-history arm. Reference files are hash-checked when reused.
After both arms finish, the runner compares the original 600, published 660,
fixed-history 660, and fresh-history 660 at strength 1 on original held-out rows
2/3 and seeds 101/303, for 20 seconds. The primary comparison is **fresh versus
fixed mean Content Enjoyment over all four matched cases**. Every recording,
individual CE difference, and secondary diagnostic is retained. This is a small
continuation experiment, not evidence about training from scratch or other styles.

Live page: <http://100.90.104.57:8888/fresh-seed-lofi-20260910/>

The persistent user services are `music-seed-fixed-20260910`,
`music-seed-fresh-20260910`, `music-seed-prewarm-20260910`, and `music-seed-finish-20260910`.
`protocol.json` fixes the experiment, `runtime-snapshot.json` records dependencies,
and `summary.json` is written only after all comparisons have been scored.

## Completed result

Both arms completed updates 601–660. The fixed arm used one continuation;
the fresh arm used 60 different seeds and 60 distinct continuations. Both
exports match their full saved states, and all saved tensors are finite.
The recorded maximum change in the style teacher was exactly zero, while
the ending teacher changed (mean RMS difference 6.824).

| Checkpoint | Mean Content Enjoyment | Mean production quality | Mean lyric agreement |
| --- | ---: | ---: | ---: |
| Starting checkpoint 600 | 7.633611 | 8.133672 | 0.805145 |
| Published checkpoint 660 | 7.761235 | 8.299131 | 0.823799 |
| One prompt, fixed seed | 7.740034 | 7.926346 | 0.901789 |
| One prompt, fresh seed each update | 7.796379 | 8.145867 | 0.860838 |

The primary fresh-minus-fixed CE difference is **+0.056345**, with three
wins out of four cases. Every case is included with equal weight:

| Held-out arrangement | Generation seed | Fixed CE | Fresh CE | Fresh minus fixed |
| --- | ---: | ---: | ---: | ---: |
| 1 (YAML row 2) | 101 | 7.722405 | 7.804034 | +0.081629 |
| 1 (YAML row 2) | 303 | 7.609216 | 7.758409 | +0.149194 |
| 2 (YAML row 3) | 101 | 7.943488 | 7.763354 | -0.180134 |
| 2 (YAML row 3) | 303 | 7.685026 | 7.859717 | +0.174691 |

This is a modest, mixed result. Fresh histories increased mean predicted
production quality by 0.219521 but reduced mean ASR-based lyric agreement
by 0.040951. These are diagnostics, not replacements for the declared CE
comparison. Four cases and a single continued training state do not establish
a reliable general gain; no checkpoint is promoted by this experiment.
The original published checkpoint used the original multi-prompt recipe, so
the fixed and fresh arms are the controlled comparison for changing history.

All 16 candidate recordings and their original WAV downloads are available
on the listening page. `scores.json` retains complete per-recording metrics,
and `summary.json` retains the primary aggregate and all paired differences.
`evaluation-file-audit.json` verifies the 32 remotely served WAV/MP3 files,
checkpoint hashes, locked render settings, and identical Off/reference audio
within each case. `browser-audit.json` records decoding of all 16 MP3s,
playback, seeking, exclusive playback, score hiding, and mobile/desktop layout.

The renderer's GPU handover retries retained the same locked prompts and
seeds. No failed handover produced a replacement candidate selected for its
score. GPU rendering and measurement services have finished; the listener
continues to bind to `0.0.0.0:8888`.
