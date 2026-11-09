# particle-sliders-core

Shared infrastructure for training and applying small adapters on frozen
models. Core 0.14.0 is model-independent and pins `particlegan==0.9.5`.

For model installation, prompts, recipes and generation, use the
[product repositories](../../docs/products.md). For a readable explanation of
teacher/student training, see the [project overview](../../README.md) and
[math guide](../../docs/math.md).

| Task | Contract |
|---|---|
| Understand core/product ownership and installed modules | [Distribution boundary](SHARED-CORE-BOUNDARY.md) |
| Supply native paired responses | [Conditional engine](CONDITIONAL-PG09-CONTRACT.md) |
| Train numerical responses | [Response sessions](RESPONSE-SESSIONS.md) |
| Use image operations | [Image sessions](IMAGE-LIFECYCLE.md) |
| Apply an adapter without training resources | [Native inference](NATIVE-INFERENCE.md) |
| Dispatch training in a product runtime | [Training command](TRAINING-COMMAND.md) |
| Search and compare candidates | [Search API](PARAMETER-SEARCH.md) · [Evaluation](EVALUATION-AND-LR-SEARCH.md) |
| Manage rates and scaling | [Learning rates](LEARNING-RATE-CONTRACT.md) · [PEFT scaling](PEFT-SCALING.md) |
| Own activation checkpoint wrappers | [Checkpoint ownership](CHECKPOINT-OWNERSHIP.md) |

Fresh sessions set `ema_decay=None`; all model weights and exports are raw.
Relative teacher/student error is the default search proxy. Measuring intended
effect in matched finished outputs is an optional improvement, not a requirement
or a reason to repeat completed searches. Historical checkpoints retain their
original source and environment pins.

See [development](../../docs/development.md) for build and test instructions.
