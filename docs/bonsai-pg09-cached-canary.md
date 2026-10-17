# Proposed bounded Bonsai cached-readout CPU canary

**Pending core-owner process admission.** This is a concrete reviewable worker,
not a launch receipt. No actual cached tensor payload or GGUF weights have been
loaded during preparation, and no server or GPU process has been started.

Worker: `scripts/verify_bonsai_pg09_cached.py`, SHA256
`9a3b987cba3d1b999ff8c9a4738be7d537630d01d1f53d1eefea46adacd79c36`.
Critic: `conceptmod/textsliders/pg09_readout_critic.py`, SHA256
`b83322a2ef2192ac3693fe136d019259c7fd8bf26fba25bb254931ba25462a57`.
Exact commands/environment are in `evidence/pg09-text-video/bonsai-cpu-admission-proposal.json`.

The worker calls the public profile validator before cached payload access, then
reuses existing `FrozenReadouts`, `BonsaiParticleHead`, the thin
`BonsaiReadoutResponseAdapter` and `ConditionalPG09`. The prior metadata preparation
pins all15 readouts and exact official model/runtime/pooling/normalization identity.
Cached serialized tensor SHA256 checks occur when the native cache loads each
readout. The5.9GB GGUF is never opened; no inference binary/server is invoked.

The readout head remains the original native128x4 routed bank and residual
branches. The small raw-logit critic sees full neutral context and native5120
responses; no row-ID condition or new dimensional projection is used. G is only
public adversarial loss plus required public prior; D is public adversarial plus
continuous penalty. MSE calls are forbidden on the optimized path.

For each preset, the produce process performs four updates, checkpointing after
two; a separate restore process replays the final two. Total proposal: four
CPU-only processes, twelve optimizer updates including replay, one Torch thread,
120seconds maximum per process with no retry/extension. Output directories are
new under `/tmp/pg09-bonsai-real-cache-20261017/`. No GPU lease is requested.

Required results are adapter/critic/actual-bank changes, positive continuous
penalties, immutable target/neutral readouts, exact full checkpoint/loss/held-out
vector continuation, safe weights-only checkpoint loading and inference export
reload. Evaluation preserves training streams. Historical three held-out pairs,
three canaries, positive-input identity diagnostic and0/.5/1 grid are reused.
Squared errors remain reporting diagnostics only. This four-update pilot cannot
qualify generation, convergence, in-model GGUF training or semantic quality.

Completed preparation evidence:

- Metadata-only preflight succeeds on the existing assets; no Torch/model import
  or cached tensor payload access occurs in this mode.
- **2 complete worker-control tests pass**, both PG presets, using a freshly
  generated synthetic temporary cache. Those tests exercise produce/restore,
  safe serialization, held-out helper context, frozen responses and export reload
  without touching the official cache.
- Broader bridge evidence remains50CPU checks, six fresh-process continuation
  cases and two cross-profile atomic rejection cases, on core `3e94b13`.

The worker never grants itself process/model admission. The core owner must
explicitly admit the exact source/command/budget before `--mode produce` or
`--mode restore` is run. Until then only `--mode preflight` is executed.
