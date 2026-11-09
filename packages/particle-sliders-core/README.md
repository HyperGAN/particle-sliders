# particle-sliders-core

Shared infrastructure for training and applying small adapters on frozen
models. Core 0.14.1 is model-independent and pins `particlegan==0.9.5`.

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
| Select rates from intended effect | [Evaluation and saved-output workflow](EVALUATION-AND-LR-SEARCH.md) · [Historical SSE runner](PARAMETER-SEARCH.md) |
| Manage rates and scaling | [Learning rates](LEARNING-RATE-CONTRACT.md) · [PEFT scaling](PEFT-SCALING.md) |
| Own activation checkpoint wrappers | [Checkpoint ownership](CHECKPOINT-OWNERSHIP.md) |

Fresh sessions set `ema_decay=None`; all model weights and exports are raw.
Select rates only by equal-case intended-effect ON-minus-OFF mean in finished
outputs, retaining negatives, zeros and all tied maxima. Relative teacher/student
error is diagnostic only. Reuse completed training, RAW checkpoints, outputs and
first measurements without extra confirmation. The existing fixed-SSE
`particle-slider-search` runner has no output-effect callback; use the supported
product inference path and external frozen instrument described in the
evaluation guide. Historical protocols, results and checkpoint source/environment
pins remain unchanged.

See [development](../../docs/development.md) for build and test instructions.
