# Reversible scale-capturing checkpoints (local proposal)

This change is not in the published API or the previously qualified core pins.
It leaves ParticleGAN mathematics, response geometry and checkpoint packets
unchanged. New source bytes require their own reviewed source identity; do not
silently rebind a research cohort or replace an installed package.

`checkpoint_with_adapter_scale(block, network)` still installs immediately and
refuses a block already wrapped by this helper. It now returns an
`AdapterScaleCheckpoint` handle with `close()` and context-manager support:

```python
from contextlib import ExitStack
from particle_sliders.pg09_backends import checkpoint_with_adapter_scale

with ExitStack() as checkpoints:
    for block in native_blocks:
        checkpoints.enter_context(checkpoint_with_adapter_scale(block, adapter.network))
    # Run the existing training/session operation here.
```

`native_blocks` are selected native modules; `adapter.network` is the existing
scale owner exposing `scale` and `scaled(value)`. No model import or selection
policy is added. ExitStack unwinds earlier blocks if a later installation or
the operation raises, including interruption. Retain the individual handles if
cleanup may encounter an ownership conflict: ExitStack itself discards failed
callbacks and cannot serve as a retry ledger. Nested distinct blocks unwind in
reverse order. Nested installation on the same block is refused while preserving
the outer wrapper. Nested scopes of the same open handle close only at the outer
exit. Ignoring the return keeps the previous eager lifetime.

Closing restores the exact prior instance `forward`/marker ownership, including
removing an override when `forward` originally came from the class. Repeated
close is safe; a closed handle cannot be reentered. The handle refuses to
overwrite a forward or marker replaced by another owner. Resolve that ownership
conflict before retrying cleanup. Do not restore the broader native owner first.
An already built autograd graph retains its captured scale even after close;
closing does not cancel pending work or provide concurrency synchronization.

`ARLastHiddenResponseAdapter` installs its selected blocks transactionally and
owns a ledger of their handles. `close_checkpointing()` attempts all remaining
handles in reverse installation order. Successfully closed handles are removed;
failed ones remain owned. The first failure is raised after all attempts. Resolve
the external ownership conflict, then call the same adapter method again; closed
siblings are not replayed. Both inherited Music3/YuE2 adapters expose this method.
Handle construction and adapter owner assignment are protected by rollback; a
failed final transfer cannot discard the local cleanup ledger.

The native factory must retain the adapter while cleanup is unresolved and run
wrapper cleanup before broader model/slider restoration. A sequential close
callback makes that ordering explicit:

```python
adapter = make_existing_native_response_adapter()

def close_native():
    adapter.close_checkpointing()  # may raise; retains unfinished handles
    restore_native_owner()        # run only after all wrappers have closed

# Supply close_native as the existing Binding's close callback. On conflict,
# retain the adapter/native owner, resolve the conflict and retry this cleanup.
```

Do not rely on a disposable ExitStack callback to remember an unresolved adapter
cleanup or continue broader native restoration after that cleanup refuses. The
adapter method is retryable; this does not change session/factory close policies.

The factory still owns modes, device placement, native checkpoint flags and LoRA
attachment. This helper restores only its own two attributes. The separate
`bind_native_gradient_checkpointing` API is unchanged. The authored tiny CPU
ownership/scale tests have not run; runtime and installed-wheel qualification
remain pending resource admission on each exact PG line.
