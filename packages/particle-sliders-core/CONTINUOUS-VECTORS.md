# Exact upstream source for continuous vectors

Official PyPI ParticleGAN 0.9.4 automatically selects its categorical penalty
branch for two-dimensional responses of width at least 170. That source remains
the default and continues to refuse wide continuous vectors here. Never reshape
a hidden vector into an image to change its penalty branch.

The separate profile `source-801ae1e-continuous-v1` binds upstream commit
`801ae1ead8f88b88eca8f5398137e3e1bc01ad36` and every shipped Python file in
`pg094_continuous.json`. It is an unreleased source cohort, not the official
0.9.4 wheel. This upstream source adds an explicit data-domain option while
preserving every automatic default. Its existing continuous RMS gradient norm,
smooth cap, gradient floor and logit gauge then apply to wide native vectors.

Install that exact upstream source in an isolated environment, and explicitly
pass both `pg_source_profile=PG_CONTINUOUS_SOURCE_PROFILE` and
`penalty_input_domain='continuous'` to `ConditionalPG09` with
`response_domain='continuous_vector'`. The engine passes the domain to the
public upstream penalty factory. Both source profiles check all package Python
bytes; neither can alias the other merely because their version strings match.

The profile is admitted for CPU API/gradient/optimizer/checkpoint conformance,
including native vector widths 170, 1024, 2048 and 4096. It does not qualify a
pretrained product, GPU precision, renderer, learned rate or published control.
A product must bind a fresh source/model/data/numerical cohort and prove its own
teacher parity, real updates, full-state continuation and portable inference.
Its native response shape stays unchanged. Old checkpoints require their
original source environment and are never migrated by changing receipt hashes.

The checkpoint binds the PG source profile, all upstream Python hashes, native
geometry, explicit domain and complete resolved recipe. A changed declaration
is rejected before state mutation. The ordinary release profile and its small
continuous-vector/image paths keep their existing defaults and boundaries.
