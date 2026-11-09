# Fix PRs for the state-machine audit

> Historical snapshot: findings, PR status and validation describe the date of
> this record. Subsequent fixes are in the current source. Use the
> [current core guides](../../README.md) for present API behavior.

The [audit](README.md) records `main` at `2b212c1`. The PRs below implement the seven priority findings as separate review units, splitting grouped ownership/recovery concerns. All 13 fix PRs were independently reviewed and merged into `main` on 2026-11-06. The additional device-visibility bug was discovered during regression work.

| Audit finding | Fix PR | Original review dependency |
|---|---|---|
| O15: retained/reversed GPU leases | [#172 — reap renderers and acquire leases in canonical order](https://github.com/HyperGAN/particle-sliders/pull/172) | main |
| A02: nested/concurrent activation ownership | [#173 — restore activation and reject competing owners](https://github.com/HyperGAN/particle-sliders/pull/173) | main |
| O06: unsuccessful calibration suppresses retry | [#174 — commit the fit digest after success](https://github.com/HyperGAN/particle-sliders/pull/174) | main |
| A01/A07/A25: partial host attachment | [#175 — roll back failed adapter attachment](https://github.com/HyperGAN/particle-sliders/pull/175) | #173 |
| O24: stranded running claims | [#176 — reconcile proven-dead claim owners without rerolling](https://github.com/HyperGAN/particle-sliders/pull/176) | main |
| O28: terminal decision before evidence archive | [#177 — commit archive before decision](https://github.com/HyperGAN/particle-sliders/pull/177) | main |
| T05–T09: partially committed training update | [#178 — quarantine interrupted updates](https://github.com/HyperGAN/particle-sliders/pull/178) | main |
| O35: incomplete final package occupies its name | [#179 — atomically publish complete package directories](https://github.com/HyperGAN/particle-sliders/pull/179) | main |
| A09–A11: partial load and unsafe PEFT fallback | [#180 — validate/roll back loads and fail closed](https://github.com/HyperGAN/particle-sliders/pull/180) | #175 |
| C03/C07: reuse after cleanup/restoration failure | [#181 — quarantine sessions while retaining cleanup retry](https://github.com/HyperGAN/particle-sliders/pull/181) | main |
| T16: partial legacy full-state restore | [#182 — transactional restore and quarantine recovery](https://github.com/HyperGAN/particle-sliders/pull/182) | #178 |
| C13/T21: prospective weight EMA | [#183 — enforce raw weights and preserve historical provenance](https://github.com/HyperGAN/particle-sliders/pull/183) | #182 |
| Additional: CPU visibility overwritten at import | [#184 — preserve an explicit CPU mask](https://github.com/HyperGAN/particle-sliders/pull/184) | main |

## Review and merge order

The review used two stacks: **#173 → #175 → #180** and **#178 → #182 → #183**. Parents were merged with merge commits, and their children were retargeted to `main`. The other fixes targeted `main` independently.

Integration retained both failure guards: a failed restore cannot clear an interrupted-update latch; only successful full restore enables continuation. Averaged historical artifacts remain preserved and are refused by prospective raw-only execution.

#180 and #183 overlapped at four checkpoint-loader decorators. The [reviewed integration](https://github.com/HyperGAN/particle-sliders/commit/192af4a26581f6ea62d886072a007eb2560a15c8) retains both imports and this order: `@classmethod`, then `@raw_checkpoint_admission`, then `@atomic_attachment(...)`, then the loader. Provenance is admitted before inspecting the host, and attachment rollback remains intact.

## Review results and merge receipts

Independent review caught two additional issues before their PRs were merged:

- **#182:** Endpoint recovery could clear failure flags without restoring adapter weights. Complete schema-2 packets now include the adapter and restore it within the transaction. Historical packets require explicitly supplied saved adapter weights; incomplete recovery leaves failure flags set. A real optimizer-mutation-then-error regression verifies exact continuation, and a late adapter-load failure verifies rollback and retained quarantine. See the [migration note](../../shared-core.md).
- **#183:** Removing averaged exports had also removed the raw `latest.json` discovery receipt. Publication and refresh are restored. Real exports at consecutive steps verify the pointer, exact current raw tensor bytes, metadata and retention of the earlier artifact.

Every code PR passed source review against its proper base and CPU regression checks. Reviewers asserted module paths to exclude editable-install masking. Focused checks below are per-PR evidence and overlap with each other; they are not additive totals. GPU and native pretrained models were not qualified.

| PR | Reviewed head | Focused evidence | Merged commit |
|---|---|---|---|
| [#172](https://github.com/HyperGAN/particle-sliders/pull/172) | `8f4a140e` | 7 lock/process tests | [adba5420](https://github.com/HyperGAN/particle-sliders/commit/adba542002a10c313a69c298f758f21e28a87cf9) |
| [#173](https://github.com/HyperGAN/particle-sliders/pull/173) | `adbf3c6e` | 103 activation/ownership tests | [586b51b5](https://github.com/HyperGAN/particle-sliders/commit/586b51b54f6e9b1d4f9135a2bfc493348b262dcc) |
| [#174](https://github.com/HyperGAN/particle-sliders/pull/174) | `60ff94db` | 2 retry regressions; broader server review | [7781c8fb](https://github.com/HyperGAN/particle-sliders/commit/7781c8fb06a83141db07b391729ca450d5d0bcd4) |
| [#175](https://github.com/HyperGAN/particle-sliders/pull/175) | `322e48f1` | 128 passed, 7 optional skips | [9e103a47](https://github.com/HyperGAN/particle-sliders/commit/9e103a472a80111a41b2d39a60f1fea2110c249d) |
| [#176](https://github.com/HyperGAN/particle-sliders/pull/176) | `f77e1b4c` | 11 regressions and a real foreign-owner refusal | [8fd1f624](https://github.com/HyperGAN/particle-sliders/commit/8fd1f624a0d9917b237a1430ec542ece2f9cb2e8) |
| [#177](https://github.com/HyperGAN/particle-sliders/pull/177) | `759d0e28` | 15 archive/decision tests | [16c27d94](https://github.com/HyperGAN/particle-sliders/commit/16c27d9441224e247e0684ad23c6dc4b90b9a4bb) |
| [#178](https://github.com/HyperGAN/particle-sliders/pull/178) | `ab31890d` | 75 passed, 9 optional skips | [bb5596a1](https://github.com/HyperGAN/particle-sliders/commit/bb5596a1afe01e077b7d5a29a21acb091c562abc) |
| [#179](https://github.com/HyperGAN/particle-sliders/pull/179) | `45e83236` | 18 regressions and 3 concurrency/durability probes | [8dfa21ff](https://github.com/HyperGAN/particle-sliders/commit/8dfa21ffb10f8d8f0ea258194ec8cb5a92ca3523) |
| [#180](https://github.com/HyperGAN/particle-sliders/pull/180) | `66cf815d` | 124 passed, 7 optional skips | [b48c0b2f](https://github.com/HyperGAN/particle-sliders/commit/b48c0b2f68db1913d57b1c73c0b67505d58dd69d) |
| [#181](https://github.com/HyperGAN/particle-sliders/pull/181) | `60cabc67` | 12 fault cases plus existing restoration/isolation suites | [21a4e177](https://github.com/HyperGAN/particle-sliders/commit/21a4e1770f52a61021bfebaa4fff4f244ec7e59b) |
| [#182](https://github.com/HyperGAN/particle-sliders/pull/182) | `f9e9dd0a` | 67 endpoint/restore tests; combined recovery checks | [59d80600](https://github.com/HyperGAN/particle-sliders/commit/59d80600cb51aea9473b1f3371977268b71adede) |
| [#183](https://github.com/HyperGAN/particle-sliders/pull/183) | `192af4a2` | 202 focused passed; 68 final integration passed | [50aae0d4](https://github.com/HyperGAN/particle-sliders/commit/50aae0d46b54d96d1c95990c40cb99b4a93409b6) |
| [#184](https://github.com/HyperGAN/particle-sliders/pull/184) | `34e96f67` | 18 isolated import-preamble cases | [01d4fae0](https://github.com/HyperGAN/particle-sliders/commit/01d4fae06e9541e0f614fde1cdcc23d8e89a4b85) |

## Validation

Each PR includes its exact local validation and limitations. Tests use CPU fixtures, owned temporary file locks/processes, failure injection, and immutable synthetic artifacts. They do not qualify pretrained/native model quality or change ParticleGAN equations, checkpoint history, the declared optimization metric, or foreign jobs.

Notable integration regressions include competing thread/async adapter owners; reversed-order concurrent GPU-lease controllers; corrupt/unknown queue ownership; native cleanup errors that preserve or suppress a body exception; failed archival without reevaluation; and partial copies across all eight research package writers.

**Final combined result: 754 passed, 18 subtests passed, 40 skipped, 1 deselected (64.93 seconds).** No failures or errors remained. Optional/native dependency cases are explicitly skipped. `git diff --check` passed and source conflict markers are absent.

The combined run covers the complete `packages/particle-sliders-core/tests` suite and these affected root tests:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  packages/particle-sliders-core/tests \
  tests/test_adapter_{scope_restoration,attachment_rollback,load_recovery}.py \
  tests/test_{package_publication,music3_particle_no_ema,attempt_archive_commit}.py \
  tests/test_{calibration_retry,generate_listen_visibility,finalist_gpu_locks}.py \
  tests/test_{reward_queue_recovery,reward_search,reward_game}.py \
  tests/test_reward_game_{complete_confirmation,joint_confirmation}.py \
  tests/test_gan_v2.py \
  -k 'not test_editable_wheel_owns_and_removes_startup_hook'
```

The environment has the combined core checkout installed editable, ParticleGAN 0.9.5, and the packaging build requirements. One nested editable-install/uninstall test is deselected because its fresh child environment inherits the machine's incompatible setuptools/backports installation. Ordinary wheel installation/uninstallation and namespace subprocess tests are included.

See [the 2026-11-07 shared-boundary and follow-up PR tracker](followups-20261107.md) for the subsequent separate `develop` PRs and new cohort validation.
