# Conditioning cache identity

`particle_sliders.execution.cache_identity` binds frozen tensor caches to an
explicit model/runtime identity and exact request values. `fingerprint_file`
streams actual file bytes; paths, sizes and timestamps alone do not establish
weight identity. Cache keys use full canonical JSON digests. Loads verify the
versioned identity/request payload before admitting a finite floating tensor.
Publication is atomic and exclusive: an existing artifact is never overwritten.

A product supplies its architecture, weight/config fingerprints and generation
settings before either a cache hit or a runtime load. Compute these once per
preparation call, from staged immutable inputs. Do not infer identity from a
loaded host's presence or from a cache filename alone. Native execution and
runtime qualification remain the product's responsibility.

The historical `textsliders.train_lora_music3` consumer now includes staged
component/config/weight bytes, optional adapter bytes and sidecars, adapter
architecture, generation settings, implementation and runtime identifiers. Its
loaded and cache-only paths share the same explicit identity. Exact duration
and scale values no longer alias through display rounding. A cache-only run
requires the staged input bytes to establish that identity.

Old unidentified cache files remain unchanged at their old paths. They cannot
establish a valid hit and must be regenerated under the new identity schema.
Mismatched or malformed payloads at new paths are refused without rewriting
them. Cache identity does not change the teacher-generation, training or optimizer formula.

Core 0.14 excludes the historical model-specific consumer from its default
wheel. The helper lives in the shared execution namespace so product-owned
consumers can reuse it without importing a model implementation from core.
The original consumer and its caches remain available under the immutable
0.13 source cohort.
