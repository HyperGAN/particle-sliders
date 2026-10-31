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

Continuous-domain admission is unchanged: this shared product gate refuses flat native vectors
of width>=170, including H3's reported1,398,528-wide response and native Music3/YuE2
responses. Neither PG0.9.4 nor the HyperGAN builder update qualifies that domain.
No reshape, categorical interpretation, unofficial overlay or bypass is introduced.
No weights were downloaded and no GPU, training, publication or deployment occurred.

The existing public main core0.11.0/PG0.9.3 and product core0.8.1/0.8.2/PG0.9.1
cohorts remain on their immutable commits. Products are not automatically repinned.
This joint qualification branch includes the separately reviewed additive
cut-point docs/probes from e86c0969b04b5a7ee6de481a8f01458cf2e2b43a and the PEFT
scaling helper from 06ac5af9cb717c47dd10ce5d83bd9cc4258a8945. Original owner
branches are preserved. See [cut-point boundaries](CUT-POINTS.md) and
[explicit PEFT scale scopes](PEFT-SCALING.md). This does not include H3 or
VoxCPM2 product code or admit their wide continuous responses. The shared
compatibility package README stays this file; the cut-point owner's source
README and MANIFEST additions are retained independently.


## Lifecycle failure follow-up (separate candidate from4abc0eb)

Training restore now converts/clones an isolated EMA and stages metrics before
any live game mutation. Global RNG is scoped across validation/application and
rollback. Complete PG preflight is factored into a private helper so invalid
packets do not invoke live load hooks merely to roll back. A late live load-hook
failure restores the previous valid game, EMA/metrics remain unchanged, and all
captured modes/gradient/checkpoint flags are restored. If rollback itself fails,
the original exception receives a diagnostic note and the game is disabled until
a valid restore. An already-failed game has no valid rollback checkpoint; a
successful restore can recover it, while failed recovery stays disabled. This is
bounded owned-state rollback, not a guarantee against arbitrary external side
effects, broken custom hooks or hardware/allocation failure.

Native and observation cleanup attempts every captured restoration action even
when another fails. Cleanup failures are attached as notes to an active original
exception; absent an original failure, the first cleanup exception is raised
after the remaining actions run. Factory/context cleanup uses that same rule.
The operation exception is passed explicitly: an unrelated caller's active
`except` block never suppresses a normal cleanup failure. `close()` marks a
training or native-inference session closed only after its binding cleanup
succeeds, so failed native cleanup can be retried.

Incoming PG state is preflighted once before live application. Shadow optimizer
loads preserve the exact per-device official PG thermal channel, including on
failure; they do not publish incoming thermal state. Failed live application
also restores that channel, the penalty's last diagnostics/collection flag,
cooler state and transient observer flags independently of optimizer rollback.
If rollback fails, the game remains disabled; this does not claim recovery of
arbitrary external effects. EMA conversion copies only the EMA container before
allocating incoming values, avoiding an unnecessary clone of the old EMA.

Published regression coverage is in `tests/test_lifecycle_restore_failures.py`.
It includes EMA trial-load failure, one-shot and persistent live-load failures,
failed rollback disabling, original-error preservation during failed cleanup,
retryable training/inference close, unrelated ambient exceptions, thermal and
diagnostic restoration, and a checked-in actual4abc synthetic training packet
with provenance in `tests/fixtures/lifecycle/`. That packet is rejected before
live hooks with game/EMA/RNG/native flags unchanged; no hashes are rewritten.

This changes shared implementation/source identity. Old4abc0eb session packets
retain that exact environment; do not rewrite their source hashes or silently
migrate them. Downstream owners must repin the reviewed follow-up source and
requalify their own fresh cohorts. No PG mathematics or admission guard changed.

HNDL extras remain deliberately separate: `hndl` means the legacy0.8.2 direct
conditional-readout environment; `hypergan-hndl`/`fibo-hndl` means the verified
HyperGAN5f368a/HNDL0.10 network environment. They cannot co-install. Legacy tests
now explicitly skip in the modern environment, and are executed separately under
0.8.2; modern graph tests execute under0.10. Neither historical pin is widened.
