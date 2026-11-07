# particle-sliders-core 0.14.0

Model-independent slider infrastructure using exact **ParticleGAN 0.9.5**.
Products supply frozen models, native response implementations, adapter target
choices, media generation and presets. This package supplies shared lifecycle,
training state, generic adapters, evaluation and hyperparameter contracts,
execution, raw exports and inference. GAN mathematics stays in ParticleGAN.

Read [the distribution boundary and migration rules](SHARED-CORE-BOUNDARY.md)
before adopting this new source cohort. Both wheels and editable installs exclude
embedded native products and the old `conceptmod` namespace hooks. Old pinned
cohorts and checkpoints retain their original loaders.

Install this directory with Python 3.11+ and PyTorch 2.6+. Products should pin a
reviewed full Git commit. Fresh configurations use `ema_decay=None`; all model
weights remain raw. Rank by operand-first FP64 per-example pooled relative SSE,
then average examples equally.

- [Response sessions](RESPONSE-SESSIONS.md)
- [Image lifecycle](IMAGE-LIFECYCLE.md)
- [Conditional response contract](CONDITIONAL-PG09-CONTRACT.md)
- [Training commands](TRAINING-COMMAND.md)
- [Native inference](NATIVE-INFERENCE.md)
- [Evaluation and LR search](EVALUATION-AND-LR-SEARCH.md)
- [ParticleGAN 0.9.5 compatibility](COMPATIBILITY-PG095.md)

Wide continuous-vector backends explicitly declare
`penalty_input_domain="continuous"` with `response_domain="continuous_vector"`.
CPU fixture qualification does not establish native model or media quality.

The shared [raw parameter search](PARAMETER-SEARCH.md) provides model-owned
hyperparameter bindings, exact full-coordinate RAW metric ranking, retained
evidence, and step-budgeted resume without a native runtime dependency.
