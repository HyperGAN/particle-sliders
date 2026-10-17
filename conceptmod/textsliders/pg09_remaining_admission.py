"""Standard-library-only contracts for sequential owner-admitted FF/Supra work."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

CORE = Path('/ml2/codex-storage/future-workspaces/2026-10-16/task-6/particle-sliders-pg09-core')
QUALIFICATION = Path('/mnt/nvme/sliders/qualification')
OUTPUT_ROOT = QUALIFICATION/'runs/remaining-pg09-sequential-v1'
MODEL_REV = '10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b'
TEXT_REV = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'
VAE_REV = '31f26fdeee1355a5c34592e401dd41e45d25a493'
MOSS_REV = 'fc5de0d56fc910cbca6f6625a70f5f582b26deb1'
CONFIG_SHA = 'd32639ad9084f60d13faea1ef0e4d92b4bfc2e0420bc0fd15f4992c6270edd5b'
CASES = {
 'final-form': dict(total_steps=3840,batch_size=8,seed=1000,checkpoint_every=80,
  response_scale=.2522578997899252,python='/mnt/nvme/sliders/moss-image-sliders/.venv/bin/python',
  hf_home='/mnt/nvme/sliders/cache/huggingface',
  versions={'torch':'2.9.1+cu128','diffusers':'0.38.0','transformers':'5.3.0'},
  caps=dict(wall_seconds=604800,phase_probe_seconds=1800,phase_resume_seconds=1800,
            phase_evaluate_seconds=7200,max_reserved_gib=24,max_rss_gib=12,
            output_gib=16,admission_free_vram_gib=30,admission_host_available_gib=16)),
 'supra-final-boss': dict(total_steps=28800,batch_size=4,seed=7,checkpoint_every=400,
  response_scale=.32844100447280516,python='/mnt/nvme/sliders/qualification/envs/pg090/bin/python',
  hf_home='/mnt/nvme/ai/caches/huggingface/main',
  versions={'torch':'2.11.0+cu128','diffusers':'0.40.0.dev0','transformers':'5.15.0'},
  caps=dict(wall_seconds=21600,phase_probe_seconds=1800,phase_resume_seconds=1800,
            phase_contexts_seconds=1800,phase_evaluate_seconds=7200,
            max_reserved_gib=12,max_rss_gib=8,output_gib=8,
            admission_free_vram_gib=18,admission_host_available_gib=12)),
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class RemainingRecipe:
    data: dict

    @classmethod
    def read(cls,path):
        return cls.validate(json.loads(Path(path).read_text()))

    @classmethod
    def validate(cls,value):
        if value.get('schema')!='remaining-pg09-worker-v1' or value.get('objective')!='pg09_gan_only':
            raise ValueError('reviewed remaining GAN schema required')
        case=value.get('case')
        if case not in CASES:
            raise ValueError('only FinalForm and Supra FinalBoss are in scope')
        settings=CASES[case]
        for key in ('total_steps','batch_size','seed','checkpoint_every','response_scale','python','hf_home','versions','caps'):
            if value.get(key)!=settings[key] or type(value.get(key)) is not type(settings[key]):
                raise ValueError('fixed recipe field differs: '+key)
        if value.get('preset')!='bispf_optimized' or value.get('sampling')!='balanced' or value.get('particles','missing') is not None:
            raise ValueError('explicit GAN preset and adapter-only balanced protocol required')
        if type(value.get('physical_gpu')) is not int or value['physical_gpu'] not in (0,1):
            raise ValueError('owner must bind exactly physical card0 or1')
        card=value['physical_gpu']
        expected=[str(QUALIFICATION/f'admission-gpu-{card}.lock'),f'/mnt/nvme/sliders/locks/gpu-{card}.lock']
        if value.get('lease_paths')!=expected:
            raise ValueError('both exact owner admission lease targets required')
        cohort=Path(value.get('cohort_root','')).resolve()
        if not cohort.is_relative_to(OUTPUT_ROOT.resolve()) or not value.get('cohort_id'):
            raise ValueError('immutable new cohort identity/root required')
        output=Path(value.get('output_directory','')).resolve()
        if not output.is_relative_to(cohort) or output==cohort:
            raise ValueError('new immutable remaining-run output root required')
        if value.get('admission') not in ('DRAFT_NOT_ADMITTED','REVIEWED_REMAINING_GAN_RECIPE'):
            raise ValueError('explicit reviewed or draft admission state required')
        if not valid_sha(value.get('resolved_recipe_sha256')):
            raise ValueError('actual public recipe fields must be bound')
        pins=value.get('source_pins',{})
        needed={'pg09_remaining_admission.py','pg09_remaining_worker.py','pg09_remaining_campaigns.py',
                'pg09_backends.py','pg09_moss_campaign.py','pg09_moss_worker.py','pg09.py','moss_original_runtime.py'}
        if (not isinstance(pins,dict) or not needed.issubset({Path(p).name for p in pins}) or
                any(not Path(p).is_absolute() or Path(p).suffix!='.py' or not valid_sha(h) for p,h in pins.items())):
            raise ValueError('all actual worker/core/bridge/helper source pins required')
        guard=value.get('resource_guard',{})
        if not Path(guard.get('path','')).is_absolute() or not valid_sha(guard.get('sha256')):
            raise ValueError('original source-pinned resource guard required')
        if (not isinstance(value.get('input_sha256'),dict) or
                any(not Path(p).is_absolute() or not valid_sha(h) for p,h in value['input_sha256'].items())):
            raise ValueError('read-only input source/cache identities required')
        handoff=value.get('owner_priority_gate',{})
        if not Path(handoff.get('path','')).is_absolute() or (handoff.get('sha256') is not None and not valid_sha(handoff['sha256'])):
            raise ValueError('explicit missing or reviewed owner priority handoff required')
        if value.get('expected_base_parameter_sha256') is not None and not valid_sha(value['expected_base_parameter_sha256']):
            raise ValueError('explicit verified or missing frozen parameter identity required')
        if case=='final-form':
            if (not valid_sha(value.get('expected_base_parameter_sha256')) or value.get('model_revision')!=MOSS_REV or value.get('teacher_indices')!=[0,39] or
                    value.get('signed_rows')!=15360 or
                    Path(value.get('model_snapshot',''))!=Path('/mnt/nvme/sliders/cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-SFT/snapshots')/MOSS_REV or
                    set(value.get('old_adapters',{}))!={'MSE200','MSE254'}):
                raise ValueError('FinalForm full cache and actual200/254 comparison identities required')
        else:
            if (value.get('model_revision')!=MODEL_REV or value.get('text_revision')!=TEXT_REV or
                    value.get('vae_revision')!=VAE_REV or value.get('config_canonical_sha256')!=CONFIG_SHA or
                    value.get('teacher_indices')!=[0,239] or value.get('logical_rows')!=600 or
                    set(value.get('old_adapters',{}))!={'MSE1600','MSE-selected27200'}):
                raise ValueError('Supra exact model/T5/19-context protocol and actual baselines required')
            contexts=value.get('contexts',{})
            if (not Path(contexts.get('path','')).resolve().is_relative_to(OUTPUT_ROOT.resolve()) or
                    contexts.get('unique_prompts')!=19 or
                    any(contexts.get(k) is not None and not valid_sha(contexts[k]) for k in ('sha256','receipt_sha256','preparation_phase_sha256','case_budget_sha256'))):
                raise ValueError('explicit missing or reviewed exact19-context export required')
        return cls(json.loads(json.dumps(value)))

    @property
    def identity(self):
        return digest(self.data)

    def verify_sources(self):
        for path,expected in self.data['source_pins'].items():
            if sha(path)!=expected:
                raise ValueError('pinned source changed: '+path)
        guard=self.data['resource_guard']
        if sha(guard['path'])!=guard['sha256']:
            raise ValueError('resource guard source changed')

    def verify_origin(self,module):
        origin=str(Path(module.__file__).resolve())
        if self.data['source_pins'].get(origin)!=sha(origin):
            raise ValueError('actual imported source not pinned: '+origin)
        return origin

    def phase_reason(self,mode,previous):
        if self.data['admission']!='REVIEWED_REMAINING_GAN_RECIPE':
            return 'default denied: no owner-reviewed admission'
        if not valid_sha(self.data['owner_priority_gate'].get('sha256')):
            return 'default denied: missing owner MOSS-priority safe-handoff proof'
        if mode not in ('contexts','probe','resume-check','train','evaluate'):
            return 'unknown phase'
        if mode=='contexts':
            return None if self.data['case']=='supra-final-boss' and not previous else 'context phase requires fresh Supra output'
        if self.data['case']=='supra-final-boss' and any(self.data['contexts'].get(k) is None for k in ('sha256','receipt_sha256','preparation_phase_sha256','case_budget_sha256')):
            return 'missing reviewed19-context export; cannot encode during training'
        if self.data['case']=='supra-final-boss' and not valid_sha(self.data.get('expected_base_parameter_sha256')):
            return 'missing reviewed19-context preparation frozen-base identity'
        if mode=='probe':
            return None if not previous else 'prior probe outcome must be reviewed; no retry'
        if previous.get('recipe_sha256')!=self.identity:
            return 'previous phase/source/recipe identity differs'
        if mode in ('resume-check','train'):
            if (previous.get('status')!='probe_passed' or previous.get('completed_steps')!=3 or
                    not all(previous.get(k) is True for k in ('frozen_base_unchanged','generator_changed','critic_changed',
                                                            'exact_in_memory_replay','exact_export_reload','file_reference_written'))):
                return 'strict three-step probe gates incomplete'
            if mode=='resume-check' and previous.get('fresh_disk_resume_exact'):
                return 'prior disk resume result requires review; no retry'
            if mode=='train' and not previous.get('fresh_disk_resume_exact'):
                return 'separate-process file-backed exact proof missing'
        if mode=='evaluate' and (previous.get('status')!='trained' or
                previous.get('completed_steps')!=self.data['total_steps'] or not previous.get('frozen_base_unchanged')):
            return 'full GAN budget/frozen-base proof incomplete'
        return None
