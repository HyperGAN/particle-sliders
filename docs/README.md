# Documentation

To train or generate with a model, start in its [product repository](products.md).
This repository documents the shared core and its integration contracts.

## Start here

- [Project overview](../README.md): what a slider is and how teacher/student training works.
- [Math](math.md): paired responses, the conditional adversarial game, and search metrics.
- [Core API map](shared-core.md): ownership boundaries and integration points.
- [Development](development.md): install, build, test and contribute.

## Integration guides

- [Core distribution boundary](../packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md)
- [Conditional engine](../packages/particle-sliders-core/CONDITIONAL-PG09-CONTRACT.md)
- [Numerical response sessions](../packages/particle-sliders-core/RESPONSE-SESSIONS.md)
- [Image sessions](../packages/particle-sliders-core/IMAGE-LIFECYCLE.md)
- [Native inference](../packages/particle-sliders-core/NATIVE-INFERENCE.md)
- [Training command and execution](../packages/particle-sliders-core/TRAINING-COMMAND.md)
- [Search API](../packages/particle-sliders-core/PARAMETER-SEARCH.md)
- [Evaluation and learning rates](../packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md)
- [Learning-rate fields](../packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md)
- [Activation checkpoint ownership](../packages/particle-sliders-core/CHECKPOINT-OWNERSHIP.md)
- [Selective PEFT scaling](../packages/particle-sliders-core/PEFT-SCALING.md)
- [Conditioning cache identity](conditioning-cache-identity.md)
- [Optional HyperGAN reporting](../packages/particle-sliders-core/HYPERGAN-REPORTING.md)
- [Pinned ParticleGAN compatibility](../packages/particle-sliders-core/COMPATIBILITY-PG095.md)

## Historical material

[History and retained evidence](history.md) explains the older experiments,
source pins, audit findings and data still present in this tree. Old model usage
pages and superseded proposals are available at immutable Git revisions rather
than mixed into current instructions.
