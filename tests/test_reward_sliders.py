"""Contract tests: position geometry, isolation, reward selection and failures."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from conceptmod.textsliders.reward_sliders.capture import ResidualCapture
from conceptmod.textsliders.reward_sliders.directions import choose_dev, causal_gate, paired_statistics, fit_direction
from conceptmod.textsliders.reward_sliders.reward import scoring_windows
from conceptmod.textsliders.reward_sliders.fixtures import pilot


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward(self, inputs_embeds, past_key_values=None):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h)
        return h


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()


class Cache:
    def __init__(self, length): self.length = length
    def get_seq_length(self): return self.length


def test_prefill_feedback_and_teacher_forcing_use_same_boundary():
    lm = Host()
    vector = torch.tensor([1., 0., 0.])
    kwargs = dict(layers=(1,2), layer=1, direction=vector, coefficient=.1, residual_norm=10.)
    prompt = torch.ones(2, 5, 3)
    feedback = torch.ones(2, 1, 3)
    with ResidualCapture(lm, **kwargs) as capture:
        assert torch.equal(lm.model(inputs_embeds=prompt), prompt)
        output = lm.model(inputs_embeds=feedback, past_key_values=Cache(5))
        assert torch.equal(output, feedback+vector)
    data = capture.trajectory()
    assert data['positions'].tolist() == [5]
    with ResidualCapture(lm, prompt_length=5, **kwargs):
        full = lm.model(inputs_embeds=torch.cat([prompt,feedback], 1))
    assert torch.equal(full[:, :5], prompt)
    assert torch.equal(full[:, 5:], output)


def test_single_token_prefill_is_not_mistaken_for_generation():
    lm = Host(); x = torch.ones(2,1,3)
    with ResidualCapture(lm, layers=(1,), layer=1, direction=torch.tensor([1.,0.,0.]), coefficient=1.):
        assert torch.equal(lm.model(inputs_embeds=x), x)


def test_zero_arithmetic_rng_exception_cleanup_and_isolation():
    lm, other = Host(), Host()
    x = torch.ones(2,6,3)
    rng = torch.get_rng_state().clone()
    with ResidualCapture(lm, coefficient=0., capture=False):
        assert torch.equal(lm.model(inputs_embeds=x), x)
    assert torch.equal(rng, torch.get_rng_state())
    kwargs = dict(layers=(1,), layer=1, direction=torch.tensor([1.,0.,0.]), coefficient=1., prompt_length=3)
    with pytest.raises(KeyboardInterrupt):
        with ResidualCapture(lm, **kwargs):
            with ThreadPoolExecutor(2) as pool:
                treated = pool.submit(lm.model, inputs_embeds=x).result()
                baseline = pool.submit(other.model, inputs_embeds=x).result()
            assert not torch.equal(treated, x)
            assert torch.equal(baseline, x)
            with pytest.raises(RuntimeError):
                with ResidualCapture(lm, **kwargs): pass
            raise KeyboardInterrupt
    assert torch.equal(lm.model(inputs_embeds=x), x)
    assert not lm.model._forward_pre_hooks and not lm.model.layers[1]._forward_hooks


def test_disjoint_tail_is_not_overlapped_or_discarded():
    assert scoring_windows(253,10,True) == [(0,100),(100,200),(200,253)]
    assert scoring_windows(253,10,False) == [(0,100),(100,200)]
    with pytest.raises(ValueError): scoring_windows(199,10)


def obs(family, seed, arm, value, split='test', status='complete'):
    return dict(id=f'{family}-{seed}-{arm}', family=family, seed=seed, arm=dict(name=arm),
                reward=dict(valid=status=='complete',scalar=value), status=status, cell_hash=family, split=split)


def test_off_wins_exact_ties_and_failed_arm_cannot_win():
    arms = [dict(name='off'),dict(name='a',layer=11,coefficient=.01),dict(name='b',layer=23,coefficient=.03)]
    rows = [obs(f, s, a['name'], 5.) for f in ('a','b','c','d') for s in (1,2) for a in arms]
    assert choose_dev(rows, arms)['selected']['name'] == 'off'
    rows[-1].update(status='failed', reward=dict(valid=False,scalar=10.))
    assert choose_dev(rows, arms)['mean_ce']['b'] is None


def test_family_bootstrap_and_failed_causal_arm():
    rows = [obs(f, s, a, v) for f in ('a','b','c','d') for s in range(4)
            for a,v in [('off',5.),('positive',5.2),('reversed',4.9),('random',5.)]]
    assert causal_gate(rows)['passed']
    statistics = paired_statistics(rows,'positive')
    assert statistics['valid_pairs'] == 16
    assert statistics['family_bootstrap_95'] == pytest.approx([.2,.2])
    rows[-1]['status'] = 'failed'
    assert not causal_gate(rows)['passed']


def test_direction_uses_within_cells_and_excludes_heldout():
    rows, features = [], {}
    for family, offset, split in [('a',100.,'train'),('b',-100.,'train'),('c',500.,'test')]:
        for seed, value in [(1,1.),(2,2.)]:
            row = obs(family,seed,'off',value,split)
            rows.append(row)
            features[row['id']] = {'11': dict(mean=torch.tensor([offset,value,900.*value if split=='test' else 0.]),
                                                       residual_norms=torch.ones(2,500)*10.)}
    teacher = fit_direction(rows,features,11)
    assert torch.equal(teacher.direction,torch.tensor([0.,1.,0.]))
    assert teacher.fitting_families == ['a','b']


def test_authored_fixture_split_and_balance():
    rows = pilot()
    assert [sum(r['split']==split for r in rows) for split in ('train','dev','test')] == [8,4,4]
    assert {r['voice'] for r in rows} == {'male','female','unspecified','instrumental'}


def test_real_qwen_generation_student_has_nonzero_target_and_trainable_lora(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from conceptmod.textsliders.reward_sliders.data import prepare_reward_rows, RewardStudentForward, set_scale
    from conceptmod.textsliders.reward_sliders.directions import RewardTeacher
    from conceptmod.textsliders.reward_sliders.specs import sha, WORKSPACE
    import sys
    sys.path.insert(0,str(WORKSPACE))
    from app.lora_runtime import LoRANetwork
    torch.manual_seed(123)
    lm = Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=3,
           num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=168100)).eval().requires_grad_(False)
    trajectory = tmp_path/'history.pt'
    torch.save(dict(prompt_embeds=torch.randn(2,5,16), frame_embeds=torch.randn(2,8,16)),trajectory)
    observation = dict(id='fixture',family='one',split='train',status='complete',seed=1,
                       trajectory=str(trajectory),trajectory_sha256=sha(trajectory))
    vector=torch.randn(16); vector/=vector.norm()
    teacher=RewardTeacher(1,.03,10.,vector,torch.zeros(16),['one'])
    rows=prepare_reward_rows(lm,[observation],teacher,lambda family:None,device='cpu',frames=8,stride=2)
    assert len(rows)==2 and rows[0]['real'].shape==(1,4,16)
    network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=0.,target_replace=['Qwen3Attention'],
                        prefix='lora_te',delimiter='-',train_method='full')
    forward=RewardStudentForward(lm,network,lambda family:None,'cpu')
    out=forward(rows[0])
    assert torch.count_nonzero(out['fake']) == 0
    loss=(out['fake']-rows[0]['real']).square().mean()
    loss.backward()
    assert sum(float(m.lora_up.weight.grad.abs().sum()) for m in network.unet_loras)>0
    assert all(p.grad is None for p in lm.parameters())
    from conceptmod.textsliders.gan_v2.critic import SpanCritic,pad_sequences
    from conceptmod.textsliders.gan_v2.engine import GANEngine
    from conceptmod.textsliders.gan_v2.train import arm_settings
    recipe,_=arm_settings('baseline',origin=0,horizon=660,diagnostics_every=0)
    critic=SpanCritic(16,width=8,layers=1,heads=2,conditioned=False,ordered=False,normalized_features=False)
    real,mask=pad_sequences([r['real'] for r in rows]); critic.calibrate_input_scale(real,mask)
    engine=GANEngine(network,critic,forward,rows,recipe)
    update=engine.update([0,1])
    assert update['step']==1 and np.isfinite(update['d_loss'])
    assert set(update['losses'])=={'adversarial','fm','end'}
    assert torch.count_nonzero(forward(rows[0])['fake'])>0
    network.detach()


def test_family_sampler_restores_and_weights_families_equally():
    from conceptmod.textsliders.reward_sliders.data import FamilySampler
    rows=[dict(family=f) for f in ('a','b','c') for _ in range(8)]
    sampler=FamilySampler(rows)
    batches=[sampler.next() for _ in range(6)]
    assert sorted(i for batch in batches for i in batch)==list(range(24))
    assert all(len({rows[i]['family'] for i in batch})==1 for batch in batches)
    restored=FamilySampler(rows)
    restored.load_state_dict(sampler.state_dict())
    assert restored.next()==sampler.next()


def test_composition_preserves_direct_strength_and_has_reduced_control():
    from conceptmod.textsliders.reward_sliders.evaluate import composition_settings
    settings=composition_settings({'female':1.68,'pop':1.12},.2,4.)
    assert settings['added-isolated']['energy']==pytest.approx(3.)
    assert settings['added-isolated']['shares']['female']==pytest.approx(.56)
    assert settings['added-fixed-energy']['energy']==pytest.approx(2.8)
    assert settings['style-reduced']['styles']==settings['added-fixed-energy']['styles']
    assert sum(settings['added-fixed-energy']['shares'].values())==pytest.approx(1.)
    with pytest.raises(ValueError): composition_settings({'style':3.9},.2,4.)


def test_full_song_normalizes_one_copy_and_weights_actual_tail(tmp_path):
    import soundfile as sf
    from conceptmod.textsliders.reward_sliders.reward import CEReward
    signal=np.concatenate([np.ones(100)*.05,np.ones(100)*.2,np.ones(53)*.1]).astype('float32')[:,None]
    path=tmp_path/'audio.wav';sf.write(path,signal,10,subtype='FLOAT')
    scorer=CEReward();scorer.model=object()
    copies=[]
    def score(data,rate,windows):
        copies.append(data.copy())
        return [dict(start_s=a/rate,end_s=b/rate,axes={'CE':float(i+2),'PQ':4.}) for i,(a,b) in enumerate(windows)]
    scorer._score=score
    result=scorer.measure(path,full_song=True)
    assert result['valid']
    assert result['scalar']==pytest.approx((20+30+4*5.3)/25.3)
    assert np.sqrt(np.mean(copies[0]**2))==pytest.approx(.1)
    assert copies[0][150,0]/copies[0][50,0]==pytest.approx(4.)
    assert np.array_equal(copies[1],signal)


def test_generic_lower_reward_orientation_is_not_just_a_label():
    from conceptmod.textsliders.reward_sliders.interfaces import fit_reward_teacher
    from conceptmod.textsliders.reward_sliders.specs import RewardSpec,digest
    from dataclasses import asdict
    rows=[obs('a',i,'off',float(i),'train') for i in (1,2)]
    features={r['id']:{'11':dict(mean=torch.tensor([float(r['seed']),0.]),residual_norms=torch.ones(2,500))} for r in rows}
    spec=replace(RewardSpec(),identifier='test-error',orientation='lower')
    for row in rows: row['reward']['reward_spec_sha256']=digest(asdict(spec))
    teacher,provenance=fit_reward_teacher(spec,rows,features,11)
    assert torch.equal(teacher.direction,torch.tensor([-1.,0.]))
    assert provenance['policy'].startswith('new reward requires')
    with pytest.raises(ValueError,match='different reward spec'):
        fit_reward_teacher(replace(spec,identifier='renamed-without-rescoring'),rows,features,11)


def test_checkpoint_formula_matches_full_merge_and_exact_off(tmp_path):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from safetensors.torch import save_file
    from types import SimpleNamespace
    from app.lora_runtime import LoRANetwork
    from conceptmod.textsliders.reward_sliders.audit import audit_merged_adapter
    from app import generator
    lm=Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=36,
                        num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=64)).bfloat16().eval()
    components=[]
    for label,multiplier in [('style',1.2),('reward',.3)]:
        network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=0.,target_replace=['Qwen3Attention'],
                            prefix='lora_te',delimiter='-',train_method='full',attach=False)
        with torch.no_grad():
            for module in network.unet_loras: module.lora_up.weight.normal_(std=.01)
        path=tmp_path/f'{label}.safetensors'
        save_file(network.state_dict(),str(path))
        components.append(dict(weights=str(path),mtime=path.stat().st_mtime,multiplier=multiplier,
             kind='language_model',rank=8,alpha=8.,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full'))
    result=audit_merged_adapter(SimpleNamespace(language_model=lm),'cpu',components,components[-1])
    assert result['modules']==144 and result['tensors']==432 and result['exact_off_restoration']
    assert all(p['max_error']==0 for p in result['projections'])
    assert generator._merge_state('cpu').signature==()


def test_training_gate_recomputes_evidence_and_requires_reversed_controls(tmp_path):
    from conceptmod.textsliders.reward_sliders.train import require_causal_evidence
    from conceptmod.textsliders.reward_sliders.specs import write_json,read_json,digest,sha
    manifest={'pilot':{'minimum_causal_ce_gain':.02}}
    teacher=tmp_path/'teacher.pt';teacher.write_bytes(b'test teacher identity')
    audio=tmp_path/'test.wav';audio.write_bytes(b'test audio identity')
    rows,jobs=[],[]
    for family in ('a','b','c','d'):
        for seed in range(4):
            for name,value in [('off',5.),('positive',5.2),('random',5.),('reversed',4.9)]:
                row=obs(family,seed,name,value)
                row.update(provenance={'manifest_sha256':digest(manifest)},audio=str(audio),audio_sha256=sha(audio))
                ident=f'{family}-s{seed}-{name}';row['id']=ident
                rows.append(row); jobs.append(dict(family=family,seed=seed,arm=row['arm']))
                write_json(tmp_path/'observations'/f'{ident}.json',row)
    gate=causal_gate(rows);gate.update(manifest_sha256=digest(manifest),teacher_sha256=sha(teacher))
    write_json(tmp_path/'causal-result.json',gate);write_json(tmp_path/'causal-arms.json',jobs)
    assert len(require_causal_evidence(tmp_path,manifest)['observation_sha256'])==64
    # A stale passing flag cannot conceal a changed scored observation.
    changed=tmp_path/'observations/a-s0-positive.json'
    row=read_json(changed);row['reward']['scalar']=4.;write_json(changed,row)
    with pytest.raises(ValueError,match='summary'):
        require_causal_evidence(tmp_path,manifest)
    write_json(tmp_path/'causal-arms.json',[j for j in jobs if j['arm']['name']!='reversed'])
    with pytest.raises(ValueError,match='64'):
        require_causal_evidence(tmp_path,manifest)
