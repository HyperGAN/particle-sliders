# Scoped initialization for a new training cohort

`particle_sliders.initialization.initialize_fresh_components` is an explicit
fresh-only helper around the admitted ParticleGAN 0.9.5 public initializer APIs.
It does not change existing constructors, optimizer formulas, default recipes,
model-weight EMA policy, checkpoints, exports or inference. It does not adopt an
initialization policy for every product. Product integration remains explicit.

Call this **after constructing the owned new trainables and before optimizers**.
A product must route fresh-versus-restore intent before that point. Do not call it
in an inference loader, during resume, after loading RAW/FULL weights, or merely
because `completed_steps` happens to be zero. Existing completed and resumed
cohorts retain their original initializer, tensors and source identity.

## Explicit per-component choices

Every component declares its method. There is no QR default and no universal
initialization CLI flag. Supported mutating scopes are parameter-only leaves:
exact `nn.Linear`, ordinary/transposed `nn.Conv1d/2d/3d`, or a plain `nn.Module`
holder with explicit distribution declarations for its existing parameters.
Composite networks list their owned leaves separately. Subclasses and modules
requiring architecture-specific finalizers (for example embedding padding rows,
batch-feature readout correction, or calibrated MoG buffers) are refused. They
need their own reviewed integration rather than a silent parameter-only rewrite.
A `keep` scope may contain a whole pretrained/frozen module.

- `keep`: preserve the entire supplied scope. No tensor values are cloned or
  hashed; ownership/storage metadata is checked. No seeds are supplied.
- `deterministic_orthogonal`: call the official deterministic initializer using
  a complete mapping of parameter names to seeds, so unrelated scopes cannot
  shift draws. Registry declarations supply the constructor's distribution;
  call-local `distributions` can declare a custom bank or a reviewed host scale.
  Orthogonal matrices have the declared distribution's RMS. Existing constant
  and identity values also retain upstream's preservation behavior.
- `sample_distributions_v1`: literally sample declared `Uniform`/`Normal` laws
  through public `initialize_`, using separate helper-owned CPU generator streams.
  An `R2Normal` declaration needs an explicit Uniform/Normal override here.
- `xavier_uniform_zero_bias_v1`: public literal Xavier at an explicit `gain`,
  with zero biases. This is a distinct method, **not default-scale QR over a
  tuned Xavier host**. Requires a wholly owned, trainable exact Linear leaf.
- `identity_linear_v1`: public identity/zero operation on a square, wholly owned
  trainable exact Linear leaf.

All mutating methods preserve explicit `KEEP` and all-zero constructor tensors,
including a neutral LoRA up projection. This protection takes precedence even
for literal Xavier/identity. The down projection must remain nonzero so the
zero-up adapter has a useful gradient path. Do not zero arbitrary critic mixing
or readout parameters; retain their declared host law. Frozen parameters and
buffers are never staged or rewritten. This helper neither creates nor inserts
a particle bank; an existing bank needs its own explicit scope/distribution.
Partially frozen Linear leaves are supported for QR/sampling; literal
Xavier/identity refuse them instead of staging a frozen matrix.

```python
from torch import nn
from particlegan.init import R2Normal
from particle_sliders.initialization import (
    ComponentInitialization, initialize_fresh_components,
)

# Product supplies these newly constructed, owned leaves/parameters.
# No pretrained load, restore, optimizer construction or forward occurs here.
# adapter.down is nonzero; adapter.up was deliberately constructed as zero.
receipt = initialize_fresh_components(
    {
        "adapter.down": ComponentInitialization(
            adapter.down, "deterministic_orthogonal", {"weight": 101}),
        "adapter.up": ComponentInitialization(
            adapter.up, "deterministic_orthogonal", {"weight": 102}),
        "critic.input": ComponentInitialization(
            critic_input, "xavier_uniform_zero_bias_v1",
            {"weight": 201, "bias": 202}, gain=1.0),
        # existing_bank is a plain parameter-only Module holder exposing z.
        "bank": ComponentInitialization(
            existing_bank, "deterministic_orthogonal", {"z": 301},
            distributions={"z": R2Normal(mean=0.0, std=host_bank_std)}),
    },
    cohort_id="product-explicit-fresh-policy-v1",
    source_identity={"product_commit": reviewed_product_commit,
                     "initialization_policy": "reviewed-host-scales-v1"},
    completed_steps=0,
    restored_from=None,
    optimizers=(),
)
# Persist receipt.canonical_json / receipt.sha256 in the product's new cohort
# provenance before constructing the game's optimizers. Use ema_decay=None.
```

This is option placement, not a runnable product command or native model loader.
Products own exact leaf names, host scales, seed assignment, adapter neutrality,
source admission and durable integration. For a holder, register the existing
Parameter object without replacing it; do not pass overlapping parent/child
scopes. Every nonempty trainable name needs an integer seed in `[0, 2**63)`,
including protected KEEP/zero values. Unknown specs, shared/aliased storage,
existing gradients and repeated initialization of marked scopes are refused.

## Provenance and limits

The receipt records explicit cohort/source declarations, all methods, gains,
parameter shapes/dtypes, named seeds, resolved host laws and effective operations,
owned-generator state digests and owned-trainable before/after byte digests.
It also records the actual helper source and admitted ParticleGAN version/source
identity. The initializer must be the recorded 0.9.5 `init.py` (SHA-256
`3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b`).
Later source/version changes require explicit admission; no captured runtime is
upgraded here. The receipt is immutable canonical JSON; `as_dict()` returns a copy.

All validation and scratch initialization finish before original tensor copies.
A validation/staging error leaves original tensor bytes, markers and global RNG
unchanged. Successful copies retain original Parameter identities and state-dict
keys. Only owned trainables are cloned; the official literal API sees a small
scratch holder, never a frozen backbone or a host's bound forward references.
Copy failures attempt value rollback, but arbitrary hardware/copy failures and
Tensor version counters are not promised to be transactional.

The helper refuses supplied nonzero steps, any `restored_from`, nonempty
`optimizers` and its own repeated-scope/original-Parameter markers (including KEEP scopes). It **cannot discover arbitrary
past checkpoint loads or an optimizer the caller omits**, and a marker is not a
checkpoint-history oracle. Products must truthfully supply fresh/restore intent
and enforce it at their actual factory/restore boundary. There is no automatic
session/factory hook in this change, and no claim of native/model qualification.
