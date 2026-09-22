# Concept Slider Core

Source repository: [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders)
(formerly `sliders-conceptmod`). The Python import remains `concept_slider_core`.

The opt-in deterministic endpoint engine accepts a serializable `SliderRecipe`:

```python
from concept_slider_core import SliderRecipe, EndpointGame, require_same_critic

recipe = SliderRecipe.from_dict({"cover_weight": 1, "pole_weight": 1,
                                "grad_kappa": 1, "particles": 1})
recipe.require_locked_shared()
require_same_critic(production_critic_identity, experiment_critic_identity)
game = EndpointGame(adapter, existing_critic, recipe, count=len(training_rows))
game.update(predict_context, target_poles, step=1)
```

`predict_context(ids, sign)` is a context manager yielding full model predictions
in the declared training coordinates; keep the adapter sign active through
backward, including activation checkpoint recomputation. `target_poles(ids)`
returns the corresponding frozen positive and negative teachers. The consumer
supplies the critic: this engine never chooses or replaces its architecture.

RpGAN and the exact ParticleGAN gradient regularizer are shared implementations.
`grad_arm`, coefficient, kappa, norm and interval all affect actual backward.
Pole and coverage weights affect the generator loss. Without a stochastic prior,
both losses equal the sum of positive/negative endpoint MSEs; weights 1/1 thus
give total coefficient 2. D/G use independent replayable row streams. Unknown
or unsupported settings fail before optimization (FM, VIC and noise must be off
in this deterministic engine). This is an opt-in engine, not an in-place migration
of any existing model trainer.

`teacher_poles` exposes the original `faithful_guard_e` geometry, extracted
verbatim with source hashes in `locked_provenance.json`. With a declared leak
axis it requires a declared slider axis. Without a leak axis it explicitly
reports `no_declared_leak_raw_poles`, matching the Music wrapper; this posture
does **not** test leftover removal. The critic-specific `tx` prohibition remains
the Music consumer's responsibility; do not change a consumer's critic to evade
its compatibility guard.

Reusable particle math and fitting algorithms for model-specific concept slider
releases. The first consumer is [Anima](https://github.com/mikkel/anima-concept-sliders).
This package depends only on PyTorch; it does not install a model runtime.

```bash
# From the sliders-conceptmod checkout, inside your model's environment:
python -m pip install --no-deps ./packages/concept-slider-core
python -m pytest packages/concept-slider-core/tests
```

Published model repositories pin a full Git commit and this package's subdirectory
in their requirements. Install the model's tested PyTorch build first. Do not
install this research repository's legacy root requirements for Anima.

```python
from concept_slider_core import RoutedMLP, fit_routed_down

bridge = RoutedMLP(inputs=8, outputs=8)
# projection_inputs: [calibration_rows, projection_width]
# routed_targets: [calibration_rows, 8], before the teacher's up projection
down, ridge = fit_routed_down(projection_inputs, routed_targets)
# Keep the teacher's up matrix: delta = (x @ down.T) @ teacher_up.T
```

The shared implementation includes soft particle routing, the global-mix critic,
relativistic paired losses, the particle variance/covariance regularizer, noise
schedule, and the dual ridge fit used by Anima. The `reference.py` implementation
is byte-identical to the released Anima source; [extraction.json](extraction.json)
records its hash and origin. Its historical default constants remain unchanged.
Each consumer must pass its own recipe; those defaults are not a claim that one
hyperparameter set is best for every model.

Model repositories own target selection, trajectories/token collection, hook
placement, checkpoint schemas, runtime dependencies, training orchestration,
evaluation, and ComfyUI conversion. The [architecture and research guide](../../docs/shared-core.md)
describes the boundary and the requirements for adding a consumer. Anima's
sampler, optimizer schedule, exact gradient cap, normalization by diffusion
position, and frozen model provenance remain in Anima.

YuE2 and Music 3 have existing implementations. This initial extraction does
not migrate them or change their releases. Research for those models remains
elsewhere in this repository until a separately verified migration.

The CPU tests exercise particle gradients and permutation invariance, critic
double backward, held-out recovery of a known linear teacher, equivalence to an
independent primal solve, and invalid calibration rejection. Anima also checks
the original source fingerprint, training/resume tests and real ComfyUI modules.

Code is MIT; retained upstream notices are included in the package. A model's
weights and samples remain subject to that model's license.
