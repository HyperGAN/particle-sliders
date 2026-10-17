"""Offline binding of an already owner-admitted native Supra context receipt.

This reads small derived contexts on CPU. It never imports a native runtime,
encodes text, loads pretrained weights, admits work or edits a prior recipe.
"""
from copy import deepcopy
import json
from pathlib import Path

from .pg09_remaining_admission import RemainingRecipe, digest, sha


def validate_preparation(draft, preparation, phase, receipt, budget):
    """Pure receipt gates, also exercised with clearly synthetic CPU fixtures."""
    if draft.data['case']!='supra-final-boss' or preparation.data['case']!='supra-final-boss':
        raise ValueError('Supra full/preparation contracts required')
    if draft.data['admission']!='DRAFT_NOT_ADMITTED':
        raise ValueError('only a new unadmitted draft may be bound offline')
    if preparation.data['admission']!='REVIEWED_REMAINING_GAN_RECIPE':
        raise ValueError('native preparation was not owner admitted')
    for key in ('cohort_id','cohort_root','source_pins','package_origins','package_source_sha256',
                'model_revision','text_revision','vae_revision','config_canonical_sha256',
                'physical_gpu','lease_paths','resource_guard','owner_priority_gate'):
        if draft.data[key]!=preparation.data[key]:
            raise ValueError('preparation/full source or cohort identity differs: '+key)
    if (phase.get('status')!='contexts_prepared' or phase.get('mode')!='contexts' or
            phase.get('recipe_sha256')!=preparation.identity or phase.get('training_updates')!=0 or
            phase.get('completed_steps')!=0 or phase.get('native_encode_calls_at_most')!=19):
        raise ValueError('successful zero-update native19-context phase proof required')
    if (budget.get('schema')!='remaining-cohort-wall-v1' or
            budget.get('cohort_id')!=draft.data['cohort_id'] or budget.get('case')!=draft.data['case'] or
            budget.get('wall_seconds')!=draft.data['caps']['wall_seconds'] or
            budget.get('deadline_unix')!=budget.get('started_unix',0)+budget.get('wall_seconds',0)):
        raise ValueError('same immutable cumulative case wall budget required')
    runtime=receipt.get('runtime_identity',{})
    if (receipt.get('schema')!='supra-frozen-contexts-v1' or receipt.get('unique_contexts')!=19 or
            receipt.get('max_encode_calls')!=19 or receipt.get('config_sha256')!=draft.data['config_canonical_sha256'] or
            receipt.get('stored_unconditional') is not True or
            runtime.get('model_revision')!=draft.data['model_revision'] or
            runtime.get('text_revision')!=draft.data['text_revision'] or
            runtime.get('preparation_recipe_sha256')!=preparation.identity or
            receipt.get('torch_version')!=draft.data['versions']['torch'] or
            draft.data['source_pins'].get(receipt.get('runtime_source'))!=receipt.get('runtime_source_sha256')):
        raise ValueError('exact native runtime/revision/config/context identity required')
    base=phase.get('frozen_base',{})
    from .pg09_remaining_admission import valid_sha
    if (not valid_sha(base.get('parameters_sha256')) or not valid_sha(base.get('buffers_sha256')) or
            base.get('parameters')!=104094736 or
            base.get('combined_sha256')!=digest({k:v for k,v in base.items() if k!='combined_sha256'})):
        raise ValueError('native frozen-base stream identity/count proof required')


def bind_prepared_contexts(draft_path, preparation_output, destination):
    """Write a NEW DRAFT only after actual native proof and CPU context validation."""
    import yaml
    from .pg09_remaining_campaigns import load_supra_contexts,validate_live_supra_context_preparation
    draft=RemainingRecipe.read(draft_path)
    output=Path(preparation_output).resolve()
    preparation=RemainingRecipe.read(output/'recipe.json')
    if output!=Path(preparation.data['output_directory']).resolve():
        raise ValueError('actual preparation output differs from its admitted recipe')
    if Path(destination).exists():
        raise FileExistsError(destination)
    draft.verify_sources()
    preparation.verify_sources()
    phase_path=output/'phase.json'
    phase=json.loads(phase_path.read_text())
    context_path=Path(draft.data['contexts']['path'])
    if context_path.resolve()!=Path(preparation.data['contexts']['path']).resolve() or not context_path.resolve().is_relative_to(output):
        raise ValueError('context path differs or escapes immutable preparation output')
    receipt_path=context_path.with_suffix('.json')
    receipt=json.loads(receipt_path.read_text())
    budget_path=Path(draft.data['cohort_root'])/'case-budget.json'
    budget=json.loads(budget_path.read_text())
    validate_preparation(draft,preparation,phase,receipt,budget)
    if (phase.get('context_sha256')!=sha(context_path) or receipt.get('sha256')!=sha(context_path) or
            phase.get('context_receipt_sha256')!=sha(receipt_path) or
            phase.get('cohort_budget_sha256')!=sha(budget_path)):
        raise ValueError('native preparation payload/receipt/budget checksum differs')
    attestation_path=output/'runtime-attestation.json'
    attestation=json.loads(attestation_path.read_text())
    if (attestation.get('recipe_sha256')!=preparation.identity or
            attestation.get('imports',{}).get('supra.runtime')!=receipt['runtime_source']):
        raise ValueError('actual native import attestation differs from preparation receipt')
    config=yaml.safe_load(Path(draft.data['config_path']).read_text())
    validate_live_supra_context_preparation(config,draft.data)
    bank=load_supra_contexts(context_path,receipt['sha256'],config)
    import torch
    if (len(bank)!=19 or any(a.shape!=(1,128,768) or b.shape!=(1,128) or
            a.dtype!=torch.float32 or b.dtype!=torch.float32 or
            not torch.isfinite(a).all() or not torch.isfinite(b).all() or not b.any()
            for a,b in bank.values())):
        raise ValueError('native nineteen full FP32 T5 tensors/masks differ')
    value=deepcopy(draft.data)
    value['expected_base_parameter_sha256']=phase['frozen_base']['parameters_sha256']
    value['contexts'].update(sha256=receipt['sha256'],receipt_sha256=sha(receipt_path),
                             preparation_phase_path=str(phase_path),preparation_phase_sha256=sha(phase_path),
                             case_budget_sha256=sha(budget_path))
    for path in (context_path,receipt_path,phase_path,budget_path,attestation_path,output/'recipe.json'):
        value['input_sha256'][str(path.resolve())]=sha(path)
    RemainingRecipe.validate(value)
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')
    return dict(status='BOUND_DRAFT_NOT_ADMITTED',path=str(destination),sha256=sha(destination),
                recipe_sha256=digest(value),live_executed=False)
