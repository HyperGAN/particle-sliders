# Preserved core 0.13 model bindings

These files preserve the exact replaced source from shared-core commit
`285180ed9813f12e912bb6bce6bada00496fd19a`. They are historical evidence, excluded
from the core 0.14 wheel and editable runtime. Use the original full source pin
for its old installed paths and checkpoint cohorts. New native implementations
belong to their model product repositories.

The `tests/` directory preserves the previous installed/native namespace tests,
including the full versions of files that mixed shared and native assertions.
Run those with their original pinned core, not the new 0.14 distribution.
The active package suite retains the generic assertions and checks the new
wheel/strict-editable boundary outside a source checkout.
