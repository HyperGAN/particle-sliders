# Fix PRs for the state-machine audit

The [audit](README.md) records `main` at `2b212c1`. The PRs below implement the seven priority findings as separate review units, splitting grouped ownership/recovery concerns. They are open proposals, not merged fixes. The additional device-visibility bug was discovered during regression work.

| Audit finding | Fix PR | Review dependency |
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

The overlapping adapter work is stacked **#173 → #175 → #180**. Training recovery/policy work is stacked **#178 → #182 → #183**. Each child PR targets the preceding feature branch so its review diff contains only its own fix. Merge stacked parents with merge commits, then retarget their children to `main`. If a parent is squash-merged or rebased during merge, first rebase the remaining child commits onto the updated `main`, then retarget and verify the diff. Other rows target `main` independently.

The stacks resolve overlapping source edits and retain both failure guards: a failed restore cannot clear an existing interrupted-update latch; only successful full restore enables continuation. Averaged historical artifacts are refused by prospective raw-only execution rather than migrated or relabelled.

#180 and #183 also overlap at four checkpoint-loader decorators. When combining them, retain both imports and this order: `@classmethod`, then `@raw_checkpoint_admission`, then `@atomic_attachment(...)`, then the loader. This admits provenance before inspecting the host while retaining attachment rollback. The combined validation checkout uses this resolution.

## Validation

Each PR includes its exact local validation and limitations. Tests use CPU fixtures, owned temporary file locks/processes, failure injection, and immutable synthetic artifacts. They do not qualify pretrained/native model quality or change ParticleGAN equations, checkpoint history, the declared optimization metric, or foreign jobs.

Notable integration regressions include competing thread/async adapter owners; reversed-order concurrent GPU-lease controllers; corrupt/unknown queue ownership; native cleanup errors that preserve or suppress a body exception; failed archival without reevaluation; and partial copies across all eight research package writers.

**Final combined result: 721 passed, 18 subtests passed, 40 skipped, 1 deselected (59.07 seconds).** No failures or errors remained. Optional/native dependency cases are explicitly skipped. `git diff --check` passed and source conflict markers are absent.

The combined run covers the complete `packages/particle-sliders-core/tests` suite and these affected root tests:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  packages/particle-sliders-core/tests \
  tests/test_adapter_{scope_restoration,attachment_rollback,load_recovery}.py \
  tests/test_{package_publication,music3_particle_no_ema,attempt_archive_commit}.py \
  tests/test_{calibration_retry,generate_listen_visibility,finalist_gpu_locks}.py \
  tests/test_{reward_queue_recovery,reward_search,reward_game}.py \
  tests/test_reward_game_{complete_confirmation,joint_confirmation}.py \
  -k 'not test_editable_wheel_owns_and_removes_startup_hook'
```

The environment has the combined core checkout installed editable, ParticleGAN 0.9.5, and the packaging build requirements. One nested editable-install/uninstall test is deselected because its fresh child environment inherits the machine's incompatible setuptools/backports installation. Ordinary wheel installation/uninstallation and namespace subprocess tests are included.
