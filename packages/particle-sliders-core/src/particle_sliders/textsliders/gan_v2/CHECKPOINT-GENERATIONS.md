# Historical GANv2 checkpoint generations

This is a publication repair for the historical GANv2 source cohort. The GAN
implementation stays on its current legacy path; none is copied into the active
model-independent shared core. The generic standard-library publisher lives in
particle_sliders.execution.checkpoint_generation.

Prospective trainer saves write full training state, RAW safetensors weights and
their JSON sidecar into a new private staging directory. All required files and
the generation manifest are hashed and fsynced before the directory becomes an
immutable generations/step-NNNNNNNNN/ generation. Only then does an atomic,
fsynced latest.json reference advance. A status file never makes a partial
collection committed. The full state retains the existing modules, optimizers,
sampler, RNG, update count and source signature; no equations, schedules, weight
averaging or historical metric semantics change.

The new trainer does not replace historical top-level state.pt, milestone,
adapter or sidecar files. Every new generation is a complete immutable
milestone. Its latest receipt contains the full-state path and raw-export path.
Standalone export() retains its old raw-export-only compatibility contract;
the trainer uses publish_checkpoint() for the complete transaction.

Resume accepts a new latest.json, generation directory, generation.json, or
the generation's state.pt, verifying the complete manifest and the stored
step/signature before mutating a game. Explicit historical .pt paths remain
supported under their existing exact source/configuration checks. Old
checkpoints do not acquire new receipts or metric qualification.

For scripts/train_lm_gan_v2.sh, use a fresh run name and set
RESUME_STATE=/absolute/path/to/previous-run/latest.json for a new committed
generation, or explicitly supply the historical full .pt path with its complete
original source/configuration pin. The captured analysis/campaign callers retain
their old top-level state.pt assumptions and are migration inputs, not updated
entry points for this prospective format. No nontransactional state.pt alias is
created to make those historical protocols appear compatible.

On a staging/build/fsync/rename failure, partial directories remain evidence
and the prior latest receipt stays authoritative. If directory publication
succeeds but latest publication fails, the committed orphan generation remains
available for explicit verified resume; it is never silently deleted,
overwritten or adopted. A failure after the latest pointer replacement can
leave the new complete generation visible even though the final directory
fsync failed; inspect/verify the receipt rather than assuming rollback. Existing
generation names are refused, and a per-run lock excludes concurrent publishers.
No training/campaign elapsed-time limit is introduced.
