# Core0.12.0: exact PG0.9.4 and HyperGAN develop compatibility candidate

This isolated source candidate pins official `particlegan==0.9.4`, source profile
`release-0.9.4`, with an exhaustive Python source manifest bound to the official
wheel SHA256 `ad21eeaea861cf0582fc39aa96eb3fd8ab8cf21e8e0f0179f0597a9c42eb99b6`.
The optional `fibo-hndl` and `hypergan-hndl` extras select released HNDL0.10.0
and the actual HyperGAN framework develop commit
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. These are immutable dependency pins,
not branch-name resolution and not ParticleGAN develop.

The new network profile is `hypergan-hndl-v2`. It verifies all HNDL Python sources,
the reviewed HyperGAN builder closure and the framework's clean build provenance.
A matching subset of Python files cannot falsely claim another framework commit.
The original `fibo_hndl_sources.json` and `pg093_release.json` remain unchanged.
Old HNDL-v1 and PG0.9.3 requests fail explicitly; existing artifacts require their
original core0.11.0 or earlier exact source pins. `pg094` is the explicit new entry
point; `pg093` refuses rather than aliasing a new release. No automatic checkpoint
migration or relaxed source/profile option exists.

The official PG0.9.4 wheel differs from0.9.3 only in `policy.py` immutable extra-state
validation. Loss, penalty, prior, optimizer and recipe source bytes are unchanged.
Four paired shared games (both public presets, with and without particles) match
exactly across releases for three updates: metrics, generator, critic and optimizer
states. Matched HNDL/Torch projections and spatial critics preserve outputs, first
and second derivatives; shared full/split continuation also matches exactly.
These are synthetic CPU contracts, not native Fibo or media qualification.

Install this candidate in an isolated Python>=3.11 environment. Native serialization
uses `serialization`; the modern network checks use `fibo-hndl,test` together. The
separately supported legacy readout critic uses `hndl` (HNDL0.8.2) in a different
environment. The two optional HNDL families cannot be co-installed. Keep native model
libraries and product-specific model staging outside this shared package.

Continuous-domain admission is unchanged: official auto refuses flat native vectors
of width>=170, including H3's reported1,398,528-wide response and native Music3/YuE2
responses. Neither PG0.9.4 nor the HyperGAN builder update qualifies that domain.
No reshape, categorical interpretation, unofficial overlay or bypass is introduced.
No weights were downloaded and no GPU, training, publication or deployment occurred.

The existing public main core0.11.0/PG0.9.3 and product core0.8.1/0.8.2/PG0.9.1
cohorts remain on their immutable commits. Products are not automatically repinned.
This candidate does not include or alter the separately owned cut-point API, H3
or VoxCPM2 product code. Its package README is this file so the other task's
README/MANIFEST edits can be integrated independently.
