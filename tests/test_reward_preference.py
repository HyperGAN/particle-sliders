import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from conceptmod.textsliders.reward_preference.objective import (
    guided_log_probs,selected_mean,preference_loss,detached_pair_coefficients,reference_kl,screen_summary)


def test_sequential_preference_backward_matches_joint_loss():
    values=torch.tensor([-.8,-1.1],requires_grad=True)
    refs=torch.tensor([-.9,-1.])
    gradient=torch.autograd.grad(1.7*preference_loss(*values,*refs,beta=5.),values)[0]
    coefficients=detached_pair_coefficients(*values,*refs,beta=5.,weight=1.7)
    torch.testing.assert_close(torch.stack(coefficients),gradient)
    assert gradient[0]<0<gradient[1]


def test_cfg_distribution_uses_both_branches_and_excludes_text_tokens():
    hidden=torch.tensor([[[1.,0.]],[[0.,1.]]]);head=torch.arange(16,dtype=torch.float32).reshape(8,2)/10
    actual=guided_log_probs(hidden,head,offset=2,vocabulary=3,eos=7)
    logits=hidden@torch.cat((head[2:5],head[7:8])).T
    expected=(1.5*logits[0]-.5*logits[1]).log_softmax(-1)
    torch.testing.assert_close(actual,expected)
    assert actual.shape==(1,4) and abs(float(actual.exp().sum())-1)<1e-6
    assert reference_kl(actual,actual)==0
    with pytest.raises(ValueError):selected_mean(actual,torch.tensor([0,1]))


def test_preference_update_increases_margin_without_changing_base():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from app.lora_runtime import LoRANetwork
    from conceptmod.textsliders.reward_sliders.data import generation_hidden
    torch.manual_seed(17);torch.set_num_threads(2)
    lm=Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=32)).eval().requires_grad_(False)
    base={k:v.detach().clone() for k,v in lm.named_parameters()}
    network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],
        prefix='lora_te',delimiter='-',train_method='full').requires_grad_(True)
    histories=[torch.randn(2,7,16) for _ in range(2)]
    def score(i):
        h=generation_hidden(lm,histories[i])[:,3:]
        logp=guided_log_probs(h,lm.lm_head.weight,offset=4,vocabulary=16,eos=21)
        return selected_mean(logp,torch.full((4,),i,dtype=torch.long))
    reference=[score(i).detach() for i in range(2)]
    optimizer=torch.optim.Adam(network.parameters(),lr=.002)
    for _ in range(6):
        optimizer.zero_grad();loss=preference_loss(score(0),score(1),*reference);loss.backward();optimizer.step()
    assert float(((score(0)-reference[0])-(score(1)-reference[1])).detach())>.01
    assert all(torch.equal(base[k],v) for k,v in lm.named_parameters())
    assert any(torch.count_nonzero(m.lora_up.weight) for m in network.unet_loras)


def test_recovered_tokens_are_shifted_past_warmup_and_reject_changed_feedback(monkeypatch):
    from diffusers.modular_pipelines.minimax_music3 import encoders
    from conceptmod.textsliders.reward_preference.replay import recover
    class Block(torch.nn.Module):
        def forward(self,inputs_embeds,**kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds,past_key_values=None)
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.model=Block();self.config=SimpleNamespace(vocab_size=10)
        def lm_head(self,h):return torch.zeros(2,10)
    monkeypatch.setattr(encoders,'_AUDIO_CODE_OFFSET',2);monkeypatch.setattr(encoders,'_SEMANTIC_VOCAB_SIZE',4)
    monkeypatch.setattr(encoders,'_AUDIO_END_TOKEN_ID',9);monkeypatch.setattr(encoders,'_AR_CFG_TOP_K',3)
    monkeypatch.setattr(encoders,'_sample_top_k',lambda logits,generator:torch.randint(2,6,(1,),generator=generator))
    monkeypatch.setattr(encoders,'_generate_depth_codes',lambda pipe,h,code,generator:(code[:,None],None))
    monkeypatch.setattr(encoders,'_embed_audio_frame',lambda pipe,codes:codes[:,:,None].float())
    generator=torch.Generator().manual_seed(71);state=generator.get_state()
    original=torch.cat([torch.randint(2,6,(1,),generator=generator)-2 for _ in range(5)])
    trajectory=dict(prompt_embeds=torch.zeros(2,3,1),frame_embeds=original[None,:,None].repeat(2,1,1).float(),
                    rng_before={'generator':state})
    result=recover(SimpleNamespace(language_model=LM()),trajectory,frames=4)
    assert torch.equal(result['tokens'],original[1:]) and result['verified_feedback_frames']==5
    trajectory['frame_embeds'][:,2]+=1
    with pytest.raises(ValueError,match='frame 2'):recover(SimpleNamespace(language_model=LM()),trajectory,frames=4)


def test_screen_rejects_large_average_gain_with_many_losses_and_retains_failures():
    controls={(str(i),7):dict(reward={'scalar':7.}) for i in range(4)}
    def rows(deltas):return [dict(family=str(i),seed=7,status='complete',reward={'valid':True,'scalar':7+d}) for i,d in enumerate(deltas)]
    assert not screen_summary(rows([2.,-.1,-.1,-.1]),controls)['passed']
    assert not screen_summary(rows([.5,.5,.5,-.4]),controls)['passed']
    assert screen_summary(rows([.1,.1,.1,-.01]),controls)['passed']
    broken=rows([.1,.1,.1,.1]);broken[0]['status']='failed'
    assert not screen_summary(broken,controls)['passed']
    assert not screen_summary(broken[:3],controls)['passed']


def test_fresh_confirmation_is_balanced_and_sound_only():
    from conceptmod.textsliders.reward_preference.setup import fresh_confirmation
    from conceptmod.textsliders.reward_sliders.specs import validate_families
    rows=fresh_confirmation();validate_families(rows)
    assert {r['voice'] for r in rows}=={'female','male','unspecified','instrumental'}
    assert len({r['lyrics'] for r in rows})==4
