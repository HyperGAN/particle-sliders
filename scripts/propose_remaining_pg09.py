#!/usr/bin/env python3
"""Create read-only-audit-bound proposals; never admit or start model work."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposals(audit, audit_path, bridge_path):
    ff, supra = audit['final_form'], audit['supra_final_boss']
    repo = bridge_path.resolve().parents[2]
    core = Path('/ml2/codex-storage/future-workspaces/2026-10-16/task-6/particle-sliders-pg09-core/packages/particle-sliders-core/src/particle_sliders/pg09.py')
    pins = [bridge_path.resolve(), repo/'conceptmod/textsliders/pg09_backends.py',
            repo/'conceptmod/textsliders/pg09_moss_campaign.py', core]
    common = dict(schema='remaining-pg09-campaign-proposal-v1', status='DRAFT_NOT_ADMITTED',
                  objective='pg09_gan_only', preset='bispf_optimized', pg_version='0.9.0',
                  pg_core_profile_commit='3e94b13eb5d6cd1ceee3f018f019a68603f35696',
                  response_domain='continuous_field', sampling='balanced', particles=None,
                  qualification='adapter-GAN-only; no disconnected particle bank',
                  source_pins={str(p): sha(p) for p in pins},
                  audit_sha256=sha(audit_path), no_mse_resume=True, no_mse_fallback=True,
                  resource_admission_owner='shared core/process owner',
                  probe=dict(stop_after_completed_steps=3, use_full_total_steps_recipe=True,
                             exact_in_memory_replay=True, exact_disk_backed_fresh_process_resume=True,
                             adapter_export_reload=True, frozen_base_stream_sha=True),
                  required_gates=['source/recipe/cache identity', 'training-only scale',
                                  'native frozen teacher/cache parity before update',
                                  'actual G/D gradient and parameter change',
                                  'frozen base stream SHA unchanged',
                                  'CPU RNG bytes and disk-backed fresh-process CUDA replay',
                                  'matched heldout semantics and preservation review'])
    final = dict(common, case='moss-final-form',
                 entrypoint='conceptmod.textsliders.pg09_remaining_campaigns.build_final_form_campaign',
                 teacher_cache=ff['cache'], cache_identity=ff['identity'],
                 model_revision='fc5de0d56fc910cbca6f6625a70f5f582b26deb1',
                 runtime=dict(python='/mnt/nvme/sliders/moss-image-sliders/.venv/bin/python', torch='2.9.1+cu128',
                              pg_import='source-verified released PG0.9 overlay; no package installation'),
                 total_steps=3840, batch_size=8, training_seed=1000, checkpoint_every=80, wall_seconds=604800,
                 response_scale=ff['scale'], rank=4, alpha=4, critic_width=32,
                 coverage=dict(signed_rows=15360, generator_exposures=30720, discriminator_exposures=30720,
                               full_balanced_cycles_per_phase=2, comparison_old254_signed_forwards=4064,
                               old_planned1600_signed_forwards=25600),
                 old_exports=ff['old_exports'], old_completed_step=254, old_rendered_step=200,
                 heldout=dict(rows=ff['heldout'], seeds=[29001], steps=50, width=512, height=512,
                              variants=['off', 'direct-positive', 'direct-positive-cfg1', 'minus', 'plus'],
                              arms=['GAN3840', 'MSE200', 'MSE254'], guidance=3.,
                              protocol_correction='conditional-only CFG with explicit empty negative; no skip22 for every arm'),
                 blockers=['resource admission after active BI/AI',
                           'new reviewed worker binding must accept FinalForm200/254, not inherited MSE1600 path',
                           'source-pin imported actual runtime and release overlay',
                           'actual native teacher parity + disk-backed probe; gate unchanged'])
    native = dict(common, case='supra-final-boss',
                  entrypoint='conceptmod.textsliders.pg09_remaining_campaigns.build_supra_campaign',
                  teacher_cache='/mnt/nvme/sliders/qualification/runs/supra-final-boss-20261017/benchmark/teacher_cache.pt',
                  original_source_sha256=supra['source_sha256'], model_revision='10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b',
                  text_revision='7bcac572ce56db69c1ea7c8af255c5d7c9672fc2',
                  vae_revision='31f26fdeee1355a5c34592e401dd41e45d25a493',
                  runtime=dict(python='/mnt/nvme/sliders/qualification/envs/pg090/bin/python',
                               packages=supra['original_run']['packages'], dtype='FP32 native DiT + CUDA BF16 autocast',
                               rank=16, alpha=16, stored_unconditional=True),
                  context_preparation=dict(max_unique_prompts=19, max_native_encode_calls=19,
                                           expected_T5_calls_with_stored_unconditional=18,
                                           entrypoint='export_live_supra_contexts',
                                           exact_unique_prompts=19,
                                           config_canonical_sha256='d32639ad9084f60d13faea1ef0e4d92b4bfc2e0420bc0fd15f4992c6270edd5b',
                                           model_revision='10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b',
                                           text_revision='7bcac572ce56db69c1ea7c8af255c5d7c9672fc2',
                                           frozen_export='new safetensors plus native-source/runtime/revision/tensor-hash receipt',
                                           online_downloads=False, placeholder_contexts=False,
                                           admission_required=True, training_updates=0),
                  total_steps=28800, batch_size=4, training_seed=7, checkpoint_every=400,
                  comparison_step=1600, wall_seconds=21600, response_scale=supra['response_scale'], critic_width=32,
                  coverage=dict(edited_states=240, preservation_states=30, logical_balanced_rows=600,
                                edit_repetitions=2, preservation_repetitions=4,
                                preservation_frequency=.2, full_balanced_cycles_per_phase=192,
                                generator_exposures=115200, discriminator_exposures=115200,
                                generator_edit_exposures=92160, generator_preservation_exposures=23040),
                  preservation=dict(objective='GAN-only native preservation target equals frozen base',
                                    archived_mse_weight=.1, new_optimized_mse_weight=None,
                                    intentional_change='remove0.1 MSE coefficient; retain1-in5 preservation exposure'),
                  validation=dict(cache='converged/validation_cache.pt', seeds=[39001,39002],
                                  edited_states=240, preservation_states=60, endpoints=12,
                                  used_for_training_or_scale=False, heldout_bridge_fruit_for_selection=False),
                  heldout=dict(rows=supra['config']['verification'], seeds=[42,1234], width=256, height=256,
                               steps=50, guidance=3., variants=['off','half','plus','teacher'],
                               arms=['GAN28800','MSE1600','MSE-selected27200'],
                               noise='native CUDA seeded SupraRuntime.noise',
                               protocol_correction='conditional-only CFG for new GAN and every old export arm'),
                  blockers=['resource admission for bounded19-context preparation, encoding not yet performed',
                            'frozen context export/source identity required before full campaign binding',
                            'original batchedCFG2 versus split conditional-only branch numerical parity must be measured',
                            'native fullmodel teacher parity and memory/timing canary; budget must fit6h bound',
                            'reviewed worker checkpoint/evaluation shell and refreshed source pins',
                            'disk-backed fresh-process CUDA replay then full budget then heldout semantics'])
    moss = Path('/mnt/nvme/sliders/moss-image-sliders-campaign/src/moss_sliders')
    final['source_pins'] = dict(common['source_pins'], **{str(p):sha(p) for p in
                                [moss/'vendor/pipeline_moss_image.py', moss/'lora.py']})
    native['source_pins'] = dict(common['source_pins'], **{v['current_path']:v['current_sha256'] for v in
                                 supra['native_source_comparison'].values()})
    return dict(final_form=final, supra_final_boss=native)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    bridge=Path(__file__).resolve().parents[1]/'conceptmod/textsliders/pg09_remaining_campaigns.py'
    result=proposals(json.loads(args.audit.read_text()), args.audit, bridge)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in result.items():
        path=args.output/(name+'.proposal.json')
        with path.open('x') as stream:
            json.dump(value,stream,indent=2)
            stream.write('\n')


if __name__=='__main__':
    main()
