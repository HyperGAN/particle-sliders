# Optional HNDL conditional readout critics

The requested “handl” is **HNDL**, the [Human-readable Network Definition
Language](https://github.com/HyperGAN/HNDL), pronounced “handle.” Its distribution
and import name are `hndl`. The current HyperGAN `develop` snapshot
`a879c4fbff903ccefeba2194bcc7a13fe7b5abb7` declares `hndl>=0.8.2` in its training
extra and uses `hndl.torch.network` with explicit named shape contracts in
`src/hypergan/hndl_networks.py`. The older local HNDL checkout at `f7712af4`
contains 0.6.0 and is not the qualified dependency for this integration.

This integration uses the released **HNDL 0.8.2** public `hndl.resolve` and
`hndl.torch.build` APIs. The release tag is
`b68a401d7535cf150d1f1a01331aa4d0617a0cb0`; the PyPI wheel SHA256 is
`2ebed7d7ee89d5de255399518db601eb69d39d4ba383ff97b10538b1a12af3ac`.
See the [HNDL specification](https://hypergan.github.io/HNDL/SPEC/) for the
declarative syntax, shape constraints, named ports and resolved plans.

## Use with the shared game

Install the optional extra on Python 3.11–3.14:

```sh
python -m pip install 'particle-sliders-core[hndl]'
```

The base package retains its own Python support and does not import HNDL.
Only constructing the optional critic requires the extra and supported Python.
The integration is a caller-selected critic for a **new reviewed training
cohort**. Existing campaign constructors and their source pins stay unchanged.

```python
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic

critic = HNDLConditionalReadoutCritic(
    response_width=8, condition_width=3, hidden=16,
).to(device='cpu')

# Supply actual prepared responses, their matching frozen conditions and a
# backend owning only differentiable slider trainables, as for any shared game.
game = ConditionalPG09(
    backend, prepared_rows, critic, conditions=conditions,
    response_domain='continuous_field', batch_size=2, total_steps=100,
)
metrics = game.step()
```

`response_width` counts all non-batch response coordinates. The critic flattens
them internally; the shared game and ParticleGAN penalty receive the original
native field. It concatenates the response with the condition and builds
`Linear → SiLU → Linear` in HNDL, returning one **unbounded raw scalar** per
example. It does not change objectives, optimizers, signed poles, model
conditioning, active adapter parameters or sampling. The public ParticleGAN
loss and penalty still perform the training updates.

This is useful for bounded image/audio/text response readouts and declarative
critic experiments. A flat readout is not an efficient replacement for large
native fields. The existing `CampaignFieldCritic` uses convolutions, conditional
FiLM and mean pooling; active MOSS and other campaign constructors retain it.
Ming and other scaffolds without a native response/trainer need no HNDL import.
Native backend qualification and performance comparisons remain separate gates.

## Architecture experiments and checkpoint identity

Optional `source=` receives two named matrices, `response` and `condition`.
It supports the bounded HNDL readout operations listed in the module, including
linear layers, activations, joins and layer normalization. For example:

```python
critic = HNDLConditionalReadoutCritic(8, 3, source='''
joined = concat(response, condition)
linear(joined, 32)
silu()
linear(16)
silu()
linear(1)
''')
```

The final output must be a scalar linear readout. Pretrained provider calls and
custom operators are rejected before resolution. HNDL handles graph parsing,
shape inference, parameter construction and PyTorch execution. Invalid shape
constraints fail at construction.

Shape acceptance alone does not qualify a custom conditional critic. For
example, a single linear layer on concatenated inputs adds a condition-only
offset that cancels between paired real/fake logits; its response gradients do
not depend on the condition. New source proposals need their own numerical
conditioning and gradient checks before a cohort starts. The default SiLU
architecture has explicit condition-dependent gradient coverage.

`critic.architecture()` returns a JSON-compatible record containing the exact
source, resolved plan, input widths, HNDL version and HNDL Python source hashes.
Keep it with experiment records. The critic's `extra_repr()` includes their
digests; `ConditionalPG09` already signs this representation and module sources
in checkpoint configuration. Same-shaped source edits, dependency changes and
legacy Python critic checkpoints therefore cannot silently cross-resume.
No new non-tensor entries are added to the critic's ordinary `state_dict()`;
shared training packets retain the `weights_only=True` loading contract.

## CPU acceptance

`packages/particle-sliders-core/tests/test_pg09_hndl.py` checks matched-weight
scores, first and second derivatives against the existing Python MLP; it also
checks that response gradients depend on conditioning. Both public BISPF
presets run real shared game steps, update generator and critic parameters,
and resume exactly from a safe-loaded checkpoint. Additional checks cover
native field geometry at the penalty boundary, raw scalar output, source/legacy
checkpoint rejection before mutation, shape errors and lazy optional imports.

The release integration target is the package owner's ParticleGAN 0.9.1 proposal.
Build and run these tests against that exact reviewed package wheel before
promotion. CPU acceptance does not qualify new native/GPU training runs or
authorize replacing any active source pin. All merges remain routed through
Ada and the parent coordinator.
