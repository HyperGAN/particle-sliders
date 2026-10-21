# Bounded image lifecycle integration proposal

The local review branch starts at exact shared PR142 f797be1, preserving its
cdfb80a parent and all original receipts. It must not be retargeted to main or
merged through PR141's 90-commit stack. No existing branch is rewritten.

Review the lifecycle/factory delta first. Then create a separate clean branch
from public main 20fdf98d25455582e7e7191eef8ef51cf73173ae (refresh this observed
head before publication). Transplant selected file contents from the reviewed
snapshot; do not cherry-pick the unrelated stack history. Review the resulting
file diff against main.

The minimum package dependency closure for the four image products is:

- Preserve main's existing legacy public modules; reconcile generic fixes in
  `_training.py`, `bispf_game.py`, `bispf_provenance.json`, `formulation.py`,
  `reference.py`, `recipe.py`, `grad_regularizers.py`, and package root exports.
  Keep explicitly named historical objectives; do not redirect their calls.
- Add `pg09.py`, `pg09_release_sources.json`, `pg09_continuous_profile.json`,
  `pg09_backends.py`, `pg09_images.py`, `pg09_readout_critic.py`, `pg09_hndl.py`,
  `field_critic.py`, `routed_projection.py`, `qwen_native.py`, `fibo_native.py`,
  `fibo_provenance.json`, `lifecycle.py`, and `session_cli.py`.
- Carry licenses/provenance, pyproject package data/dependency metadata,
  requirements-pg09.txt, MANIFEST.in, the conditional PG contract and image
  lifecycle guide. Regenerate source manifests from actual candidate bytes.
- Carry the conditional PG/state, native image/Fibo and lifecycle tests plus
  package import/sdist/wheel tests adapted to that deliberately bounded surface.
  Preserve previous receipts as historical evidence, never relabel them.

`field_critic` and `routed_projection` were extracted without altering their
mathematics so image products no longer need to import the MOSS campaign module
or the tiny causal-LM host to access these generic facilities. Historical paths
re-export their original class names. The clean branch does not need
`pg09_moss_campaign.py`, `pg09_audio.py`, `pg09_text_video.py`,
`pg09_remaining_campaigns.py`, Music3/audio campaigns, their launchers,
qualification directories, model outputs, unrelated docs or stack commits.
The generic `pg09_backends.py` remains one reusable response/scale/state module;
its definitions do not launch native jobs or campaigns.

Build the selected candidate wheel, verify its complete dependency closure and
run the bounded CPU tests in separate Iris, Qwen, Krea and Fibo environments.
Then establish ONE immutable common candidate commit/wheel and update all four
derivative draft dependencies/locks together. A different clean candidate pin
requires a fresh source cohort even when transplanted math is identical.

The current local derivative updates demonstrate the architecture on the
f797-based review snapshot; they are proposed draft updates, not permission to
merge it into main. Re-pin and regenerate locks once the clean main candidate
is approved. Keep original cdf/f797 pins and receipts available for old runs.
Native GPU/pretrained qualification, active training pins, publication and
merges remain separately held for exact-head review. Qwen and Fibo keep their
incompatible Diffusers revisions in separate environments.
