# Native provider evaluation scope

`Binding.observation()` now encloses the lazy provider's `len(rows)` and row
fetch, scale substitution, and native generated response. It previously began
only after fetch. Product hooks can therefore restore native LRU/cache/buffer
state from capture as well as forward, including either failure. Ambient RNG,
adapter buffers/modes and game training state remain owned by the common
observation scope. Products must still restore native state outside the adapter
network through their explicit hook; the core does not copy an entire native base.

This narrow patch introduces no training-provider or GAN math change. Its four
regressions cover successful repeated seeded observation, fetch and forward
failures, and invalid bounds, including native state and ambient RNG restoration.
These cases have installed offline CPU coverage on this source line.
They do not qualify a native provider or permit checkpoint source rebinding;
reviewed `da4bae3` and `32f6830` remain untouched.
