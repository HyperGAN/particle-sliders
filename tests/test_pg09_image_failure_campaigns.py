"""CPU contract checks only. These do not establish actual image/sampler quality."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conceptmod.textsliders.pg09_backends import AdapterParameters, PreparedResponse
from conceptmod.textsliders.pg09_image_failure_campaigns import (
    CampaignRow, SanaHappyCampaignBackend, SD3SmileCampaignBackend,
    OnlineSnapshotProvider, online_campaign_step, render_archived_case, digest,
)


class SanaMath(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = nn.Linear(1, 1, bias=False)
        self.attn2.weight.data.fill_(2.)
        self.config = SimpleNamespace(timestep_scale=1., in_channels=1)
        self.masks = []

    def forward(self, x, *, encoder_hidden_states, encoder_attention_mask, timestep, return_dict):
        self.masks.append(encoder_attention_mask.detach().clone())
        return (x * self.attn2.weight.reshape(1,1,1,1) + encoder_hidden_states.mean(),)


def sana_fixture():
    model = SanaMath()
    backend = SanaHappyCampaignBackend(model, model_identity={'qualification':'cpu-contract-only'})
    def condition(v):
        return dict(embeds=torch.tensor([[[v],[v]]]), mask=torch.tensor([[1,0]]))
    row = CampaignRow(condition(1.), condition(3.), condition(-3.),
                      torch.ones(1,1,2,2), torch.tensor(500.), unconditional=condition(0.))
    return backend, row


def test_sana_preserves_raw_student_cfg_teacher_and_full_xattn():
    backend, row = sana_fixture()
    prepared = backend.prepare(row)
    assert torch.equal(prepared.neutral, torch.full_like(row.latent, 3.))
    assert torch.equal(prepared.target, torch.full_like(row.latent, 15.5))
    # Student remains raw neutral3, not neutral CFG4.5=6.5.
    assert torch.equal(backend.generated(prepared), prepared.neutral)
    assert all(torch.equal(mask, torch.ones_like(mask)) for mask in backend.transformer.masks)
    assert len(backend.network.weights) == 1 and not backend.transformer.attn2.weight.requires_grad
    with torch.no_grad():
        backend.network.weights[0].add_(.25)
    generated = backend.generated(prepared)
    gradient = torch.autograd.grad(generated.sum(), backend.network.weights)[0]
    assert torch.count_nonzero(gradient)
    assert torch.equal(backend.transformer.attn2.weight, torch.tensor([[2.]]))
    with backend.network.scaled(0):
        assert torch.equal(backend.network.call(row.latent, encoder_hidden_states=row.neutral['embeds'],
            encoder_attention_mask=row.neutral['mask'], timestep=row.timestep, return_dict=False)[0], prepared.neutral)
    assert all('attn2' not in name or name.startswith('weights') for name in backend.network.state_dict())


class RecordingPipe:
    def __init__(self, model):
        self.transformer = model
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def test_sana_native_render_preserves_actual_masks_seed_grid_and_restores_model():
    backend, row = sana_fixture()
    pipe = RecordingPipe(backend.transformer)
    result = render_archived_case(backend, pipe, row.neutral, row.unconditional,
                                 family='sana', seed=456, scale=.5)
    assert pipe.transformer is backend.transformer and backend.network.scale == 1
    assert result['generator'].initial_seed() == 456
    assert result['guidance_scale'] == 4.5 and result['num_inference_steps'] == 20
    assert torch.equal(result['prompt_attention_mask'], row.neutral['mask'])
    assert torch.equal(result['negative_prompt_attention_mask'], row.unconditional['mask'])


class RawSD3Math(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(1.), requires_grad=False)

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict):
        return (hidden_states*self.base + encoder_hidden_states.mean(),)


def test_sd3_uses_native_scheduler_endpoint_and_bf16_teacher_arithmetic():
    from contextlib import nullcontext
    from diffusers import FlowMatchEulerDiscreteScheduler
    model = RawSD3Math()
    leaf = nn.Parameter(torch.tensor(.1))
    network = AdapterParameters([('fixture',leaf)], lambda s: nullcontext())
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(50)
    backend = SD3SmileCampaignBackend(model, network, scheduler, model_identity={'qualification':'cpu-contract-only'})
    def c(v):
        return dict(embeds=torch.tensor([[[v]]]), pooled=torch.tensor([[v]]))
    row = CampaignRow(c(1.),c(3.),c(1.),torch.ones(1,1,2,2,dtype=torch.bfloat16),
        scheduler.timesteps[25], step_index=25,timesteps=scheduler.timesteps,sigmas=scheduler.sigmas)
    prepared = backend.prepare(row)
    native = deepcopy(scheduler)
    native._step_index = 25
    expected_pos = native.step(torch.full_like(row.latent,4.),row.timestep,row.latent,
                               return_dict=False)[0].to(torch.bfloat16)
    neutral = backend.endpoint(row.neutral,row,0.)
    negative = backend.endpoint(row.negative,row,0.)
    expected = (neutral+(expected_pos-negative)).float()
    assert torch.equal(prepared.target,expected)
    assert not torch.equal(prepared.target,torch.full_like(expected,4.))
    assert scheduler.step_index is None  # No cursor leak from per-call copies.
    with pytest.raises(ValueError,match='match'):
        backend.prepare(CampaignRow(**{**vars(row),'timestep':torch.tensor(123.)}))


class ContractBackend:
    def __init__(self):
        self.network = nn.Linear(1,1,bias=False)
        self.network.weight.data.fill_(.2)

    def generated(self, item):
        return item.row.latent * self.network.weight.reshape(1,1,1,1)

    def metadata(self):
        return dict(qualification='cpu-contract-only', protocol='online-current-adapter-snapshot-proposal-v2')


def build_contract_row(backend, descriptor, step):
    noise = torch.full((1,1,2,2), descriptor['noise'])
    prefix = noise + backend.network.weight.detach().reshape(1,1,1,1)
    c = dict(embeds=torch.tensor([[[descriptor['context']]]]))
    row = CampaignRow(c,c,c,prefix,torch.tensor(500.),provenance=dict(step=step))
    return PreparedResponse(row,prefix+.1,prefix)


def make_provider(backend, builder=build_contract_row):
    descriptors = [dict(noise=1.+i,context=.5+i,initial_noise_sha256=digest(torch.full((1,1,2,2),1.+i))) for i in range(4)]
    return OnlineSnapshotProvider(backend, descriptors, torch.tensor([[.5+i,500.] for i in range(4)]),
        (1,1,2,2), builder, protocol_identity=dict(qualification='cpu-contract-only',all_conditions_collision_checked=True))


def test_online_snapshot_is_stable_within_step_and_current_adapter_between_steps():
    backend = ContractBackend()
    provider = make_provider(backend)
    identity = provider.identity()
    with pytest.raises(RuntimeError,match='snapshot'):
        provider[0]
    provider.begin_step(0)
    first = provider[0]
    assert provider[0] is first and len(provider.audit) == 1
    with torch.no_grad():
        backend.network.weight.add_(.1)
    with pytest.raises(RuntimeError,match='changed'):
        provider[0]
    provider.end_step()
    provider.begin_step(1)
    assert not torch.equal(first.row.latent,provider[0].row.latent)
    assert provider.identity() == identity
    assert provider.audit[0]['step'] == 1
    assert provider.audit[0]['g_version_sha256'] == digest(backend.network.state_dict())
    assert provider.audit[0]['prefix'] == digest(provider[0].row.latent)
    provider.end_step()


def bad_random_builder(backend, descriptor, step):
    torch.rand(1)
    return build_contract_row(backend,descriptor,step)


def test_online_builder_rejects_hidden_rng():
    provider = make_provider(ContractBackend(),bad_random_builder)
    provider.begin_step(0)
    previous = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='global RNG'):
        provider[0]
    assert torch.equal(previous,torch.get_rng_state())
    provider.end_step()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6,8),nn.SiLU(),nn.Linear(8,1))

    def forward(self,response,conditions):
        return self.net(torch.cat([response.flatten(1),conditions],dim=1)).squeeze(-1)


def make_game(preset):
    from particle_sliders.pg09 import ConditionalPG09
    torch.manual_seed(19)
    backend = ContractBackend()
    provider = make_provider(backend)
    game = ConditionalPG09(backend,provider,Critic(),conditions=None,particles=None,
        response_domain='continuous_field',preset=preset,batch_size=2,total_steps=5,
        pg_source_profile='release',penalty_input_domain='auto')
    return game,provider


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        for x,y in zip(a,b):
            equal(x,y)
    else:
        assert a==b


@pytest.mark.parametrize('preset',['bispf','bispf_optimized'])
def test_online_provider_core_owns_updates_and_rebuilds_exact_resume(preset):
    torch.set_num_threads(1)
    game,provider = make_game(preset)
    assert not provider.audit
    online_campaign_step(game,provider)
    online_campaign_step(game,provider)
    state = deepcopy(game.state_dict())
    expected = [online_campaign_step(game,provider) for _ in range(3)]
    final = deepcopy(game.state_dict())
    row_receipts = deepcopy(provider.audit)
    other,other_provider = make_game(preset)
    other.load_state_dict(state)
    equal(expected,[online_campaign_step(other,other_provider) for _ in range(3)])
    equal(final,other.state_dict())
    equal(row_receipts,other_provider.audit)
    assert not torch.cuda.is_initialized()


def mutating_builder(backend, descriptor, step):
    backend.network.weight.add_(.1)
    return build_contract_row(backend,descriptor,step)


def test_builder_cannot_change_generator_inside_snapshot():
    provider = make_provider(ContractBackend(),mutating_builder)
    provider.begin_step(0)
    with pytest.raises(RuntimeError,match='changed inside'):
        provider[0]
    provider.end_step()


def test_archived_proposals_preserve_cases_and_do_not_read_weights(monkeypatch):
    from pathlib import Path
    from scripts.prepare_image_failure_campaigns_pg09 import proposals
    original = Path.read_bytes
    def guarded(path):
        assert path.suffix not in ('.pt','.safetensors','.bin')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    document = proposals()
    assert document['dispatch_allowed'] is False and len(document['campaigns']) == 2
    sana,sd3 = document['campaigns']
    assert [r['seeds'] for r in sana['heldout']] == [[123,456],[123,456],[789]]
    assert [r['seeds'] for r in sd3['heldout']] == [[123],[124],[125]]
    assert sana['evaluation']['strength_grid'] == [0.,.5,1.]
    assert sd3['evaluation']['strength_grid'] == [0.,-1.,.5,1.]
    assert len(sana['frozen_baseline_assets']) == 9 and len(sd3['frozen_baseline_assets']) == 6
    assert len(sana['training_prompts']['rows']) == 2 and len(sd3['training_prompts']) == 4
    for item in document['campaigns']:
        assert item['heldout'][-1]['control']
        assert item['evaluation']['fruit_is_evaluation_only']
        assert item['checkpoint_content_loaded'] is False
        assert [a['preset'] for a in item['arms']] == ['bispf','bispf_optimized']
        assert all(a['g_updates']==1000 and a['full_g_cycles']==4 for a in item['arms'])
        assert item['response_coordinates']['response_scale']['value'] is None
        assert item['pending_admission_gates']


def test_sana_unqualified_native_checkpointing_rejects():
    model = SanaMath()
    model.is_gradient_checkpointing = True
    with pytest.raises(ValueError,match='checkpoint qualification'):
        SanaHappyCampaignBackend(model,model_identity={'qualification':'cpu-contract-only'})


def test_sd3_prefix_builder_calls_original_algorithm_with_current_g_and_frozen_noise():
    from contextlib import nullcontext
    from diffusers import FlowMatchEulerDiscreteScheduler
    from conceptmod.textsliders.pg09_image_failure_campaigns import build_sd3_current_prefix_snapshot
    model = RawSD3Math()
    leaf = nn.Parameter(torch.tensor(.1))
    network = AdapterParameters([('fixture',leaf)],lambda s:nullcontext())
    observed = []
    def original_prefix(transformer,scheduler,latent,**kwargs):
        assert not torch.is_grad_enabled()
        observed.append((network.scale,float(leaf.detach()),kwargs))
        return latent+leaf.detach().to(latent.dtype)
    backend=SD3SmileCampaignBackend(model,network,FlowMatchEulerDiscreteScheduler(),
        model_identity={'qualification':'cpu-contract-only'},archive_prefix=original_prefix)
    def c(v):
        return dict(embeds=torch.tensor([[[v]]]),pooled=torch.tensor([[v]]))
    row=CampaignRow(c(1.),c(3.),c(1.),torch.ones(1,1,2,2,dtype=torch.bfloat16),torch.tensor(0.))
    descriptor=dict(input_row=row,initial_noise_sha256=digest(row.latent),
                    selected_step_index=25,unconditional=c(0.))
    first=build_sd3_current_prefix_snapshot(backend,descriptor,0)
    with torch.no_grad():
        leaf.add_(.1)
    second=build_sd3_current_prefix_snapshot(backend,descriptor,1)
    assert not torch.equal(first.row.latent,second.row.latent)
    assert observed[0][0]==observed[1][0]==1.
    assert observed[0][1]!=observed[1][1]
    assert all(o[2]['total_timesteps']==25 and o[2]['guidance_scale']==1 for o in observed)
    assert torch.equal(observed[0][2]['text_embeddings'],torch.cat([c(0.)['embeds'],c(1.)['embeds']]))
    assert first.row.provenance['initial_noise_sha256']==second.row.provenance['initial_noise_sha256']
    assert first.row.provenance['prefix_sha256']!=second.row.provenance['prefix_sha256']
    bad=dict(descriptor,initial_noise_sha256='bad')
    with pytest.raises(ValueError,match='initial noise'):
        build_sd3_current_prefix_snapshot(backend,bad,1)


def test_sana_minus_canary_cannot_substitute_for_empty_teacher():
    from dataclasses import replace
    backend,row=sana_fixture()
    value=backend.prepare(row).target
    assert torch.equal(value,backend.prepare(replace(row,negative=dict(embeds=torch.full((1,2,1),100.)))).target)
    with pytest.raises(ValueError,match='explicit empty'):
        backend.prepare(replace(row,unconditional=None))


def test_campaign_conditions_include_empty_teacher_and_native_interval():
    from dataclasses import replace
    from conceptmod.textsliders.pg09_image_failure_campaigns import campaign_conditions
    backend,row=sana_fixture()
    item=backend.prepare(row)
    first=campaign_conditions([item],family='sana')
    changed=replace(row,unconditional={**row.unconditional,'embeds':row.unconditional['embeds']+1})
    second=campaign_conditions([backend.prepare(changed)],family='sana')
    assert not torch.equal(first,second)


class RngRecordingPipe(RecordingPipe):
    def __call__(self,**kwargs):
        import random
        import numpy as np
        torch.rand(1)
        random.random()
        np.random.rand()
        return super().__call__(**kwargs)


def test_native_evaluation_restores_all_training_rng_streams():
    import random
    import numpy as np
    backend,row=sana_fixture()
    pipe=RngRecordingPipe(backend.transformer)
    cpu,python,numpy=torch.get_rng_state().clone(),random.getstate(),np.random.get_state()
    render_archived_case(backend,pipe,row.neutral,row.unconditional,family='sana',seed=123,scale=1.)
    assert torch.equal(cpu,torch.get_rng_state()) and python==random.getstate()
    actual=np.random.get_state()
    assert numpy[0]==actual[0] and np.array_equal(numpy[1],actual[1]) and numpy[2:]==actual[2:]


def test_image_critic_uses_genuine_field_and_conditions_without_output_bound():
    from conceptmod.textsliders.pg09_image_failure_campaigns import ImageFailureCritic
    critic=ImageFailureCritic((1,2,2,2),3)
    response=torch.ones(2,2,2,2,requires_grad=True)
    conditions=torch.ones(2,3,requires_grad=True)
    logits=critic(response,conditions)
    gradients=torch.autograd.grad(logits.sum(),[response,conditions])
    assert logits.shape==(2,) and all(torch.count_nonzero(g) for g in gradients)
    assert critic.score_bound==0
    with pytest.raises(ValueError,match='native4D'):
        ImageFailureCritic((1,8192),3)


def test_online_descriptor_and_conditions_remain_checkpoint_bound():
    backend=ContractBackend()
    provider=make_provider(backend)
    provider.begin_step(0)
    provider.descriptors[0]['noise']=999.
    with pytest.raises(RuntimeError,match='immutable descriptor changed'):
        provider[0]
    provider.end_step()
    provider=make_provider(backend)
    provider._conditions.add_(1.)
    with pytest.raises(RuntimeError,match='conditions changed'):
        provider.begin_step(0)


def descriptor_mutating_builder(backend, descriptor, step):
    descriptor['noise'] += 1
    return build_contract_row(backend, descriptor, step)


def test_builder_descriptor_mutation_is_rejected_without_changing_bound_inputs():
    provider = make_provider(ContractBackend(), descriptor_mutating_builder)
    before = deepcopy(provider.descriptors)
    provider.begin_step(0)
    with pytest.raises(RuntimeError, match='descriptor changed'):
        provider[0]
    equal(before, provider.descriptors)
    equal(before, provider._descriptors)
    assert not provider.audit
    provider.end_step()


@pytest.mark.parametrize('kind', ['parameter_data', 'persistent_buffer', 'nonpersistent_buffer', 'new_buffer'])
def test_complete_generator_state_guard_includes_data_writes_and_all_buffers(kind):
    backend = ContractBackend()
    backend.network.register_buffer('persistent', torch.tensor(1.))
    backend.network.register_buffer('transient', torch.tensor(2.), persistent=False)
    provider = make_provider(backend)
    provider.begin_step(0)
    provider[0]
    if kind == 'parameter_data':
        backend.network.weight.data.add_(1.)
    elif kind == 'persistent_buffer':
        backend.network.persistent.data.add_(1.)
    elif kind == 'nonpersistent_buffer':
        backend.network.transient.add_(1.)
    else:
        backend.network.register_buffer('new', torch.tensor(3.))
    with pytest.raises(RuntimeError, match='generator changed'):
        provider[0]
    provider.end_step()
    assert provider._g_state is None and provider._g_buffers is None


def test_conditions_reject_drift_even_after_row_was_cached():
    provider = make_provider(ContractBackend())
    provider.begin_step(0)
    provider[0]
    provider._conditions[0].add_(1.)
    with pytest.raises(RuntimeError, match='conditions changed'):
        provider.conditions(torch.tensor([0]))
    provider.end_step()
