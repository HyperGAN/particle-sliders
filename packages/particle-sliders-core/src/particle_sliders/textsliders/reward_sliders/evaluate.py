"""Automatic dev calibration, composition controls and a fresh full-song test."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

from .specs import DEFAULT_RUN, WORKSPACE, ROOT, RewardSpec, digest, read_json, write_json, sha, validate_families
from .render import ResearchRenderer, resolve_styles
from .reward import CEReward
from .directions import RewardTeacher, choose_dev, paired_statistics
from .diagnostics import IntentDiagnostics, preservation_tolerances
from .experiment import status, verify
from .train import require_causal_evidence
from .transfer import families as transfer_families


def component(path, strength):
    path = Path(path)
    metadata = read_json(path.with_suffix('.json'))
    required = dict(kind='language_model',rank=8,alpha=8,target_replace=['Qwen3Attention'],
                    prefix='lora_te',delimiter='-',train_method='full',unit_scale=1.)
    if any(metadata.get(k) != value for k,value in required.items()) or sha(path) != metadata['weights_sha256']:
        raise ValueError('Reward checkpoint format or hash mismatch')
    return dict(required, weights=str(path),mtime=path.stat().st_mtime,multiplier=float(strength))


def attach_intent(observation, family, measurer, record_path):
    result = dict(observation,voice=family['voice'])
    if observation['status']=='complete':
        result['intent'] = measurer.measure(observation['audio'],family)
    write_json(record_path,result)
    return result


def composition_settings(styles, quality, maximum):
    """Return exact direct multipliers plus the equivalent studio E/shares."""
    energy = sum(styles.values())
    if quality < 0 or energy+quality > maximum:
        raise ValueError('Composition would exceed host energy')
    if energy <= 0 or quality >= energy:
        raise ValueError('Fixed-energy composition requires 0 < quality < existing style energy')
    reduced = {name:value*(energy-quality)/energy for name,value in styles.items()}
    arms = {
        'style-original': (styles,0.),
        'added-isolated': (styles,quality),
        'style-reduced': (reduced,0.),
        'added-fixed-energy': (reduced,quality),
    }
    return {name:dict(styles=dict(weights),quality=q,energy=sum(weights.values())+q,
                     shares={**{k:v/(sum(weights.values())+q) for k,v in weights.items()},
                             'reward-ce-v1':q/(sum(weights.values())+q)}) for name,(weights,q) in arms.items()}


def extra_style_hashes(manifest, components):
    sys.path.insert(0,str(WORKSPACE))
    from app.rewriter import _artist_name_hit
    for comp in components:
        path = Path(comp['weights'])
        sidecar = path.with_suffix('.json')
        meta = read_json(sidecar)
        prompt = Path(meta['prompts_file'])
        if not prompt.is_absolute(): prompt=ROOT/prompt
        if _artist_name_hit('',prompt.read_text()):
            raise ValueError('New transfer style has prohibited prompt provenance')
        for file in (path,sidecar,prompt):
            manifest['style_hashes'][str(file)] = sha(file)


def preservation_report(records, tolerances):
    pairs=defaultdict(dict)
    for record in records:
        pairs[(record['family'],record['seed'])][record['arm']['name']]=record
    details, issues=[] ,[]
    for (family,seed),arms in pairs.items():
        if any(name not in arms or arms[name]['status']!='complete' or not arms[name].get('intent',{}).get('valid')
               for name in ('off','lora')):
            issues.append(dict(family=family,seed=seed,reason='Output or diagnostic failure'))
            continue
        a,b=arms['off'],arms['lora']
        ad,bd=a['reward']['diagnostics'],b['reward']['diagnostics']
        changes=dict(PQ=bd['axes']['PQ']-ad['axes']['PQ'],
                     style_similarity=b['intent']['style_similarity']-a['intent']['style_similarity'],
                     clipped_fraction=bd['clipped_fraction']-ad['clipped_fraction'])
        voice=a['voice']
        if voice in ('female','male','instrumental'):
            changes['voice_similarity']=b['intent']['voice_similarities'][voice]-a['intent']['voice_similarities'][voice]
        if 'phrase_accuracy' in a['intent']['lyrics']:
            changes['phrase_accuracy']=b['intent']['lyrics']['phrase_accuracy']-a['intent']['lyrics']['phrase_accuracy']
        if 'instrumental_word_rate' in a['intent']['lyrics']:
            changes['instrumental_word_rate']=b['intent']['lyrics']['instrumental_word_rate']-a['intent']['lyrics']['instrumental_word_rate']
        for name in ('rms_db','crest_db','window_rms_cv'):
            changes[name]=b['intent']['dynamics'][name]-a['intent']['dynamics'][name]
        failures=[]
        for name,value in changes.items():
            limit=tolerances['values'].get(name)
            excursion=(abs(value) if name in ('rms_db','crest_db','window_rms_cv') else
                       value if name in ('clipped_fraction','instrumental_word_rate') else -value)
            if limit is None or excursion>limit:
                failures.append(name)
        if abs(bd['duration_s']/ad['duration_s']-1)>tolerances['duration_relative']:
            failures.append('duration')
        if bd['silent_fraction']-ad['silent_fraction']>tolerances['silent_fraction_increase']:
            failures.append('silence')
        details.append(dict(family=family,seed=seed,changes=changes,failures=failures))
        if failures: issues.append(dict(family=family,seed=seed,reason=','.join(failures)))
    diversity={}
    for arm in ('off','lora'):
        vectors=[r['intent']['embedding'] for r in records if r['arm']['name']==arm and r.get('intent',{}).get('valid')]
        if len(vectors)>1:
            x=np.array(vectors); x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            similarities=x@x.T
            diversity[arm]=float(similarities[np.triu_indices(len(x),1)].mean())
    if len(diversity)==2 and diversity['lora']-diversity['off']>tolerances['diversity_cosine_similarity_increase']:
        issues.append(dict(reason='Output embedding diversity decreased beyond tolerance'))
    return dict(passed=not issues,issues=issues,pairs=details,mean_pairwise_audio_cosine=diversity)


def evaluate(run):
    from .lifecycle import exclusive_stage
    with exclusive_stage(run):
        return _evaluate(run)


def _evaluate(run):
    run=Path(run); manifest=read_json(run/'manifest.json'); verify(manifest)
    require_causal_evidence(run,manifest)
    folder=run/'evaluation'; folder.mkdir(exist_ok=True)
    teacher=RewardTeacher(**torch.load(run/'teacher.pt',map_location='cpu',weights_only=True))
    arms=[dict(name='off')]
    for step in manifest['student']['steps']:
        path=run/'student'/f'reward-ce-v1_step{step}.safetensors'
        for strength in manifest['student']['strength_grid']:
            comp=component(path,strength)
            arms.append(dict(name=f'lora-step{step}-m{strength:g}',step=step,coefficient=strength,
                             checkpoint=str(path),checkpoint_sha256=sha(path)))
    evaluation_spec=dict(manifest_sha256=digest(manifest),arms=arms,
        dev_families=[f['family'] for f in manifest['families'] if f['split']=='dev'],seeds=manifest['pilot']['dev_seeds'],
        dev_lora_renders=72,capture_free_off_renders=8,composition_renders=16,
        pilot_test_student_renders=16,transfer_minimum_renders=48,
        selection='highest dev mean CE, off eligible, exact ties off then smaller strength then lower step',
        transfer_rule='one fixed checkpoint and absolute multiplier; keep existing style multipliers fixed within host max',
        source_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name not in ('report.py','campaign.py')})
    if (folder/'manifest.json').exists() and read_json(folder/'manifest.json')!=evaluation_spec:
        raise ValueError('Evaluation recipe/source changed')
    write_json(folder/'manifest.json',evaluation_spec)
    renderer=ResearchRenderer(run,manifest)
    scorer=CEReward(RewardSpec(**manifest['reward_spec']))
    intent=IntentDiagnostics(folder/'intent-cache')
    baselines=read_json(run/'baseline-observations.json')
    dev=[o for o in baselines if o['split']=='dev' and o['seed'] in manifest['pilot']['dev_seeds']]
    status(run,'dev_lora_calibration',completed=0,total=72)
    for family in [f for f in manifest['families'] if f['split']=='dev']:
        for seed in manifest['pilot']['dev_seeds']:
            for arm in arms[1:]:
                dev.append(renderer.observe(family,seed,arm,scorer,extra=component(arm['checkpoint'],arm['coefficient'])))
                status(run,'dev_lora_calibration',completed=len(dev)-8,total=72)
    selection=choose_dev(dev,arms)
    write_json(folder/'selection.json',selection)
    selected=selection['selected']
    if selected['name']=='off':
        write_json(run/'decision.json',dict(decision='failed_lora_transfer',reason='Dev CE selected Off over every checkpoint/strength',
                                          activation_teacher_supported=True,trained_lora='research artifacts retained'))
        status(run,'complete',decision='failed_lora_transfer'); return
    frozen_comp=component(selected['checkpoint'],selected['coefficient'])
    from .audit import audit_merged_adapter
    mix_family=next(f for f in manifest['families'] if f['split']=='dev' and len(f['style_multipliers'])>1)
    write_json(folder/'numerical-merge-audit.json',audit_merged_adapter(renderer.pipe,renderer.device,
               renderer.components(mix_family,frozen_comp),frozen_comp))
    write_json(folder/'selected-adapter.json',dict(arm=selected,component=frozen_comp,
               strength_rule='constant direct reward multiplier, existing style multipliers fixed',
               solo_control='sweep host energy; changing a lone fader does not change effective multiplier'))
    # Matched capture-free timing includes setup, merge and pipeline execution.
    overhead=[]
    for family in [f for f in manifest['families'] if f['split']=='dev']:
        for seed in manifest['pilot']['dev_seeds']:
            off=renderer.observe(family,seed,dict(name='timing-off'),scorer)
            lora=next(o for o in dev if o['family']==family['family'] and o['seed']==seed and o['arm']['name']==selected['name'])
            activation=read_json(run/'observations'/f"{family['family']}-s{seed}-{read_json(run/'dev-selection.json')['selected']['name']}.json")
            overhead.append(dict(family=family['family'],seed=seed,off=off['timing'],lora=lora['timing'],activation=activation['timing']))
    write_json(folder/'runtime.json',dict(pairs=overhead,target_fraction=.02,
        median_lora_overhead=float(np.median([r['lora']['total_seconds']/r['off']['total_seconds']-1 for r in overhead])),
        median_activation_overhead=float(np.median([r['activation']['total_seconds']/r['off']['total_seconds']-1 for r in overhead])),
        lora_overhead_p10_p90=np.quantile([r['lora']['total_seconds']/r['off']['total_seconds']-1 for r in overhead],[.1,.9]).tolist(),
        activation_overhead_p10_p90=np.quantile([r['activation']['total_seconds']/r['off']['total_seconds']-1 for r in overhead],[.1,.9]).tolist(),
        limitation='sequential matched runs with concurrent studio load on another GPU; report variability, not zero-cost claims'))
    # Direct addition and actual fixed-E sharing have explicitly different controls.
    compositions=[]
    original_manifest=renderer.manifest
    for family in [f for f in manifest['families'] if f['split']=='dev' and len(f['style_multipliers'])>1]:
        settings=composition_settings(family['style_multipliers'],selected['coefficient'],manifest['host_energy_max'])
        write_json(folder/f"composition-{family['family']}.json",settings)
        for seed in manifest['pilot']['dev_seeds']:
            for name,setting in settings.items():
                variant=dict(family,style_multipliers=setting['styles'])
                changed=deepcopy(manifest)
                changed['style_components'][family['family']]=resolve_styles(setting['styles'])
                renderer.manifest=changed
                arm=dict(name='composition-'+name,energy=setting['energy'],shares=setting['shares'])
                obs=renderer.observe(variant,seed,arm,scorer,
                         extra=component(selected['checkpoint'],setting['quality']) if setting['quality'] else None)
                # The comparison ID is intentional; retain actual cells/settings in observations.
                compositions.append(dict(obs,cell_hash=family['family']))
    renderer.manifest=original_manifest
    write_json(folder/'composition-results.json',dict(observations=compositions,
        isolated=paired_statistics(compositions,'composition-added-isolated','composition-style-original'),
        fixed_energy=paired_statistics(compositions,'composition-added-fixed-energy','composition-style-original'),
        matched_reduced_style=paired_statistics(compositions,'composition-added-fixed-energy','composition-style-reduced')))
    test=[o for o in baselines if o['split']=='test']
    for family in [f for f in manifest['families'] if f['split']=='test']:
        for seed in manifest['pilot']['seeds']:
            test.append(renderer.observe(family,seed,dict(name='lora'),scorer,extra=frozen_comp))
    write_json(folder/'pilot-test-lora.json',dict(statistics=paired_statistics(test,'lora'),
              limitation='These pilot families were already used by the activation gate; not fresh final validation'))
    # Measure training baselines to freeze preservation variability BEFORE transfer.
    status(run,'freezing_transfer_preservation')
    preservation_baselines=[]
    for obs in [o for o in baselines if o['split']=='train']:
        family=next(f for f in manifest['families'] if f['family']==obs['family'])
        preservation_baselines.append(attach_intent(obs,family,intent,folder/'baseline-intent'/f"{obs['id']}.json"))
    tolerances=preservation_tolerances(preservation_baselines)
    fresh=transfer_families()
    validate_families(manifest['families']+fresh)
    transfer_manifest=deepcopy(manifest)
    transfer_manifest.update(families=fresh,transfer=dict(adapter=selected,seeds=[5501,6607],
        primary='duration-weighted normalized CE across every nonoverlapping 10-second window including tail',
        tolerances=tolerances,required_full_songs=48,
        quality_claim='95% whole-family bootstrap lower bound > 0; preservation passes; >= two thirds of family means improve; every voice and temporal section mean is nonnegative'))
    for family in fresh:
        comps=resolve_styles(family['style_multipliers'])
        transfer_manifest['style_components'][family['family']]=comps
        extra_style_hashes(transfer_manifest,comps)
    transfer_dir=run/'transfer'; transfer_dir.mkdir(exist_ok=True)
    if (transfer_dir/'manifest.json').exists() and read_json(transfer_dir/'manifest.json')!=transfer_manifest:
        raise ValueError('Frozen transfer protocol changed')
    write_json(transfer_dir/'manifest.json',transfer_manifest)
    renderer.manifest=transfer_manifest; renderer.run=transfer_dir
    records=[]
    status(run,'fresh_full_song_transfer',completed=0,total=48)
    for family in fresh:
        for seed in transfer_manifest['transfer']['seeds']:
            for arm in ('off','lora'):
                observation=renderer.observe(family,seed,dict(name=arm),scorer,extra=frozen_comp if arm=='lora' else None,full_song=True)
                if observation['status']=='complete':
                    duration=observation['reward']['diagnostics']['duration_s']
                    # Full-song cap hits and grossly short outputs are failures, not selected excerpts.
                    if duration>=family['render_cap_seconds']-.5 or duration<family['intended_seconds']*.5:
                        observation.update(status='failed',error='Full song hit cap or ended before half its intended duration')
                        write_json(transfer_dir/'observations'/f"{observation['id']}.json",observation)
                records.append(attach_intent(observation,family,intent,transfer_dir/'intent'/f"{observation['id']}.json"))
                status(run,'fresh_full_song_transfer',completed=len(records),total=48)
    statistics=paired_statistics(records,'lora')
    preservation=preservation_report(records,tolerances)
    subgroups={}
    for attribute in ('voice','genre'):
        subgroups[attribute]={}
        for value in sorted({f[attribute] for f in fresh}):
            ids={f['family'] for f in fresh if f[attribute]==value}
            subgroups[attribute][value]=paired_statistics([r for r in records if r['family'] in ids],'lora')
    failures=[r['id'] for r in records if r['status']!='complete']
    improvement=(not failures and statistics['valid_pairs']==24 and statistics['family_bootstrap_95'][0]>0)
    section_changes={}
    for section in ('beginning','middle','ending'):
        section_rows=[]
        for record in records:
            copy=deepcopy(record)
            if copy['status']=='complete':
                copy['reward']['scalar']=copy['reward']['diagnostics']['sections_ce'][section]
            section_rows.append(copy)
        section_changes[section]=paired_statistics(section_rows,'lora')
    broad_support=dict(family_positive_fraction=float(np.mean([d>0 for d in statistics['family_deltas'].values()])) if statistics['family_deltas'] else 0.,
        every_voice_nonnegative=all(s['mean_delta'] is not None and s['mean_delta']>=0 for s in subgroups['voice'].values()),
        every_section_nonnegative=all(s['mean_delta'] is not None and s['mean_delta']>=0 for s in section_changes.values()))
    broad=improvement and preservation['passed'] and broad_support['family_positive_fraction']>=2/3 and broad_support['every_voice_nonnegative'] and broad_support['every_section_nonnegative']
    decision=('broad_quality_candidate' if broad else
              'scoped_ce_improvement_with_preservation_tradeoffs' if improvement else 'inconclusive_or_failed_transfer')
    if improvement and preservation['passed'] and not broad:
        decision='scoped_ce_improvement'
    result=dict(decision=decision,statistics=statistics,subgroups=subgroups,section_changes=section_changes,
                broad_support=broad_support,preservation=preservation,failures=failures,
                adapter=selected,production_registry_changed=False,default='off',
                interpretation='average result in this calibration domain; no universal quality guarantee')
    write_json(transfer_dir/'results.json',result); write_json(run/'decision.json',result)
    generator=renderer.host
    generator._merge_sliders(renderer.pipe,renderer.device,[])
    exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in generator._merge_state(renderer.device).pristine.items())
    write_json(folder/'final-merge-audit.json',dict(exact_off_restoration=exact,
               adapter=frozen_comp,production_compatible_rank=8,production_compatible_alpha=8))
    status(run,'complete',decision=decision)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    evaluate(parser.parse_args().run_dir)
