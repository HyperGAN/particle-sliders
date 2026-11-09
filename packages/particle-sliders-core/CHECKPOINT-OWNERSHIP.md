# Activation checkpoint ownership

This guide concerns **activation recomputation during backward**, not disk
training checkpoints. A recomputed block must use the adapter strength captured
by its original forward, even if an outer strength scope has since exited.

`checkpoint_with_adapter_scale(block, network)` installs immediately and returns
an `AdapterScaleCheckpoint` handle with `close()` and context-manager support:

```python
from contextlib import ExitStack
from particle_sliders.pg09_backends import checkpoint_with_adapter_scale

with ExitStack() as checkpoints:
    for block in product_selected_blocks:
        checkpoints.enter_context(checkpoint_with_adapter_scale(block, network))
    # Product/session operation; network exposes scale and scaled(value).
```

Products select blocks; core supplies no model discovery policy. Earlier
installations unwind if a later one fails. The same block cannot be wrapped
twice. Nested scopes of one open handle close only on the outer exit.

Closing restores the exact prior instance `forward` and marker ownership,
including removing an override when `forward` originally came from the class.
Repeated close is safe; a closed handle cannot be reentered. A replaced forward
or marker is an ownership conflict: the helper refuses to overwrite it. Resolve
that conflict before retrying cleanup. A built autograd graph retains its
captured scale after close; closing does not cancel pending work or synchronize
concurrent calls.

## Native adapter cleanup

`ARLastHiddenResponseAdapter` installs selected blocks transactionally and keeps
an ownership ledger. `close_checkpointing()` attempts remaining handles in
reverse order, removes successful entries and raises the first failure after
attempting all siblings. Failed entries remain available for retry.

The product must retain that adapter until cleanup succeeds, and close wrappers
before restoring the broader native owner:

```python
def close_native():
    adapter.close_checkpointing()
    restore_native_owner()  # only after wrapper cleanup succeeds
```

Do not rely on a disposable `ExitStack` to remember failed cleanup callbacks;
it discards them. Retain individual handles or the adapter's retry ledger when
external ownership conflicts are possible. Factory cleanup still owns native
modes, device placement, checkpoint flags and adapter attachment. This helper
restores only its own attributes.

A qualified native public checkpoint API can instead use
`bind_native_gradient_checkpointing(model, network)`. Keep one checkpoint owner
per block. [PEFT scaling](PEFT-SCALING.md) shows composition with a selected
adapter scope. Ordinary uncaptured checkpoint calls can replay with the wrong
scale and are not equivalent.
