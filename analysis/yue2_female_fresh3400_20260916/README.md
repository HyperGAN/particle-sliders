# Female YuE2, fresh sampling from update 1

The user liked the original 600-step slider and requested training and ranking
through 3400. They then requested a rerun with one consistent formulation and
varying seeds/prompts throughout, removing the historical step-601 transition.
The original weights and listening files remain intact as `reference600`.

The new run starts from a zero-output rank-8, alpha-8 AR adapter. It uses one
prompt per update throughout, balanced shuffled passes over the original four
training rows, and 3400 distinct history seeds starting at 3000001. Every update
generates a new frozen-base history of up to 250 semantic tokens. Prompt-state
teachers remain fixed; sampled histories supply ending supervision.

The loss stays RpGAN + batch-mean feature matching + native end-margin MSE,
all weighted 1. With batch size 1, the feature mean is that update's feature.
Generator/critic learning rates remain 0.0005/0.00075, beta1 0 and beta2 0.999.
Generator weight decay is 1e-6 and gradient values are clipped at 1. There is
no added parameter-update cap, no hidden-state MSE, no lyric hold, and no phase
change at 600. Milestones only save, render and compare the same ongoing game.

Base-history sampling uses the official native CUDA-graph decoder with the
adapter disabled and native off-mode CFG. It was repeatable in the local probe
and approximately twice as fast as eager sampling. Graph and eager sampling
were **not token-identical** in the probe; their attention arithmetic differs.
The chosen graph backend is pinned for the entire fresh run. Adapter training
and listening renders retain eager forwards so the live LoRA hooks remain active.

Training and rendering use physical GPU 1. CPU scoring uses cached CLAP and
Audiobox Aesthetics. The studio environment and GPU 0 are not modified.
The initial observed rate was about 7.2 seconds/update, mostly fresh generation;
3400 updates therefore take roughly 6.5–7 hours before milestone rendering.

## Automatic workflow

`campaign.py` resumes full game state and processes steps 600, 1000, 2000, 3000,
and 3400. Recovery state is saved every 20 updates; milestones retain immutable
full state, adapters and source/recipe signatures. SIGTERM/SIGINT requests a
graceful pause. Restarting the same command resumes saved state, exact sampler,
optimizers, critic, teachers, RNG and history accounting. Duplicate sampled
histories are rejected without replacing the seed.

`render.py` compares four matched cases: held-out rows 2/3, seeds 1709/2903,
strength 1, 500 semantic tokens (about 20 seconds). Off and explicit female-caption
controls are shared. The original 600 and all five fresh checkpoints give
32 total previews. Partial outputs are kept separately; complete outputs have
native manifests and checkpoint identities and are never silently overwritten.

`measure.py` reuses the existing `quality-later-v2` selector and technical checks.
It prefers the latest clean new-run checkpoint within 0.2 of the best clean
mean enjoyment and mean production separately. The original 600 is a separate
reference run and can be selected if no new candidate qualifies. Style is a
separate diagnostic and cannot change selection. Lyrics are informational only;
no new ASR run is required by this protocol. Near-identical waveform checks and
integrity gates remain in force. Near silence, severe clipping, non-finite state,
or all four candidate clips shorter than four seconds stop the campaign for review.

Selected native weights, sidecar and selection rationale are copied to the
listening page's `selected/` folder after each ranking pass. These are provisional
local choices; the workflow does not publish to the Hub or change studio defaults.

## Paths

- Training: `models/female-yue2-fresh3400-20260916/`
- Logs, pinned protocol, audits: this directory.
- Listening and ranking: `eval/listen/yue2-female-fresh3400-20260916/`
- Page: <http://100.90.104.57:8888/yue2-female-fresh3400-20260916/>
- Runner: `/ml2/music/.cache/yue2-test-env/bin/python -u analysis/yue2_female_fresh3400_20260916/campaign.py`

The initial training process was launched before the coordinator and is recorded
in `initial-process.json`. The coordinator waits for that process before using
the same GPU, then takes responsibility for the remaining steps and comparisons.

Verification before launch: 40 tests passed across fresh sampling/resume, the
YuE2 adapter, UNI16 objective and shared critic/feature-matching tests. The new
checks cover unique histories from the first update, unchanged style teachers,
exact adapter-off history sampling, balanced prompt sampling, and tensor-exact
resume of the adapter, critic, both optimizers and sampler.
