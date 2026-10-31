# Explicit PEFT adapter scaling

`particle_sliders.peft_scaling.peft_adapter_scale_context(roots, *, adapter_names)`
is an additive opt-in scale owner for an already attached, stock PEFT `Linear`
on `torch.nn.Linear`. It requires PEFT only when called. It does not change the
historical `pg09_backends.lora_scale_context`, introduce a dependency pin, attach
adapters or choose product targets. H3/Pika integration belongs to its native
owner; this helper does not imply that its custom wrapper needs replacement.

## Selection and preservation

`roots` is a nonempty collection of `(canonical_native_path, nn.Module)` pairs.
`adapter_names` is an explicit nonempty collection of unique, nonempty exact
keys; a bare string is rejected. Module identity deduplicates overlapping roots.
Distinct managed leaves must have distinct canonical provenance paths.
**Every requested adapter must be present and active on every PEFT leaf beneath
the supplied roots.** A missing/incomplete name on one leaf refuses the entire
scope before any scaling dictionary changes. For intentionally sparse placement,
pass the exact narrower native modules; there is no silent partial coverage.

Bind once after native attachment/device placement, before temporary scale
overrides. The returned callable captures each selected key's existing numeric
factor, including native alpha/rank or RS-LoRA scaling. At strength `s` it sets
only those keys to `captured_factor * s`. It does not infer selection from
`requires_grad`, substrings or whichever adapter PEFT happens to activate.
Unselected keys retain their original values, active status and parameters.
Both base and fixed-companion parameters must be frozen by the product; optimizer
ownership is a separate `AdapterParameters` contract.

Finite signed, fractional and zero strengths are supported here; a product may
declare a narrower range. Nested scopes compute from captured factors, never
from an outer strength and never by dividing a scale back out. Normal exit and
exceptions restore the exact entry dictionary object and values for every owned
layer. This changes no weights, activation, modes or gradient flags.
Cleanup attempts every owned dictionary even if its preflight is interrupted.

The complete scope is preflighted before writes: missing/inactive adapters,
merged/disabled layers, nonfinite factors, unsupported PEFT layer/base types and
LoRA variants such as DoRA are refused. The bounded implementation supports
stock Linear LoRA only, including its native RS-LoRA factors; embeddings,
convolutions, quantized/custom leaf classes and variants are unqualified.
After binding, changing adapter/module/parameter topology, activation or fixed
scales requires a new scope. Trainable weight values and mode/grad flags may
change without rebinding.

Nested or body mutation of unselected scales/topology raises an error; it is
not silently accepted as a new baseline. Owned temporary dictionaries are
restored while reporting the error. If another owner replaces a scaling
dictionary, the helper reports an ownership conflict and does not overwrite
that external replacement; other owned siblings still unwind. Concurrent scope
use and external mutation are unsupported. A native owner must resolve such
conflicts before reuse; a failure does not undo user-performed weight merging or
topology edits. Keep model mutation outside active scales/checkpoint graphs.

## Checkpoint integration

The native factory supplies this context to the existing `AdapterParameters`
owner. The same owner must be captured by checkpoint recomputation:

```python
from contextlib import ExitStack
from particle_sliders.peft_scaling import peft_adapter_scale_context
from particle_sliders.pg09_backends import (
    AdapterParameters, checkpoint_with_adapter_scale,
)

# Product has already attached/activated both adapters, frozen the native base
# and fixed companion, and validated its exact selected trainable leaves.
scope = peft_adapter_scale_context([('transformer', native_model)],
                                   adapter_names=('slider',))
network = AdapterParameters(selected_named_parameters, scope)

with ExitStack() as checkpoints:
    for block in product_selected_checkpoint_blocks:
        checkpoints.enter_context(checkpoint_with_adapter_scale(block, network))
    with network.scaled(.5):
        response = native_model(native_input)
    # The scale scope has exited; block replay still captures the forward's .5.
    response.square().mean().backward()  # illustrative probe, not the GAN loss
```

This is a composition example, not a new product training loop. A model using a
supported native public checkpoint API may instead use the existing
`bind_native_gradient_checkpointing(native_model, network)`. Keep a single
checkpoint owner; do not wrap the same block twice. The common lifecycle still
keeps its outer scale owner active through the PG update.

The helper controls scale; it does not intercept arbitrary application calls to
`torch.utils.checkpoint`. A bare uncaptured checkpoint can replay after scope exit
at the wrong restored scale. Do not use that combination: use the captured owner
above or a qualified native hook. The generic helper cannot detect every hidden
application checkpoint call. Tests explicitly support backward after both scale
scope and checkpoint-handle exit, with a different outer strength during replay,
multiple forward strengths in one graph and repeated `autograd.grad` replay.

## Numerical zero and linearity

Numeric PEFT scale zero **still evaluates A/B and adds the scaled delta**. It is
not a true branch bypass: `0 * inf` can produce NaN. With finite deterministic
CPU fixture values, zero matches the base **plus the fixed companion**; it does
not mean disabling all adapters. Do not claim universal bit identity for native
mixed precision, stochastic forwards or nonfinite adapters. Core response and
gradient validation remain necessary.

At a single vanilla Linear leaf with fixed input, the selected residual is
linear in its scalar coefficient. A complete nonlinear multi-layer network,
trajectory or sampled image/audio need not be linear in slider strength.

## Provenance and historical compatibility

Include `scope.metadata()` in the product response metadata, for example:

```python
def metadata(self):
    return dict(existing_verified_product_metadata,
                adapter_scale_scope=self.scope.metadata())
```

The receipt records protocol, helper source SHA256, exact selected names,
canonical target paths, selected baseline factors, unselected factors and active
names. It preflights the scope again before returning a fresh snapshot. The
product still supplies verified base/fixed-adapter content and revision identities.
No new required field is injected into historical packet formats. Opting in
changes backend behavior/provenance and needs a new qualified source cohort;
do not rewrite an old checkpoint signature to admit it.

The old `lora_scale_context` intentionally continues to scale every PEFT name
under its roots. There is no new runtime warning: old callers have no explicit
selection declaration from which to distinguish intended all-adapter scaling
from fixed-companion misuse. Its behavior is preserved by a compatibility test,
not endorsed for mixed fixed/trainable hosts. New mixed hosts should use this
explicit helper or a separately verified product-owned selective wrapper.

`tests/test_peft_scaling.py` uses real, randomly initialized PEFT mixed adapters,
without model downloads. It exercises zero/fractional/signed/full strengths,
strict coverage, fixed-companion preservation, frozen weights after a selected
optimizer step, deduplication, nested/exception cleanup, mutation/topology
refusals, captured checkpoint gradients and numerical-zero limitations. PEFT is
an optional test dependency; this file skips explicitly when it is absent.
Run it with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and CPU-only bounded
thread settings. Passing tiny fixtures does not qualify H3/Pika or other native
backends, GPU kernels, waveform rendering or wide continuous response domains.

Executed source receipt on 2026-10-31: **186 passed in 8.97s**, including 43
real-PEFT cases and the previous 143 cut-point/lifecycle/provenance cases. The
isolated environment used Python 3.14.4, Torch 2.15.0+cpu, official ParticleGAN
0.9.3, PEFT 0.21.2, Transformers 5.22.1 and pytest 9.2.0. These are test-environment
versions, not new repository dependency pins or a cross-product qualification.
The installed PG source map retained the exact digest recorded in CUT-POINTS.md.
Importing `particle_sliders.peft_scaling` in a fresh process did not import PEFT.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q packages/particle-sliders-core/tests/{test_peft_scaling,test_cut_points,test_checkpoint_ownership,test_conditional_pg09,test_fibo_native,test_image_lifecycle,test_response_lifecycle,test_dependency_direction,test_compact_inference}.py
```
