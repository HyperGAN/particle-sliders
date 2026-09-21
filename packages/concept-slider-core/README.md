# Concept Slider Core

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
