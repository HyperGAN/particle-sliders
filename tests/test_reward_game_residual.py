"""Compare the proposed richer objective with the actual sequential decoder."""
from types import SimpleNamespace
import torch

from conceptmod.textsliders.reward_game.residual_policy import log_probs,selected_mean


def test_residual_teacher_forcing_matches_sequential_decoder_and_keeps_gradients():
    from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
    from diffusers.modular_pipelines.minimax_music3.encoders import _generate_depth_codes,_AUDIO_CODE_OFFSET
    torch.manual_seed(91);torch.set_num_threads(2)
    decoder=MiniMaxMusic3RVQDepthDecoder(hidden_size=16,num_layers=2,num_attention_heads=2,
        intermediate_size=32,audio_vocab_size=64,num_codebooks=8).eval().requires_grad_(False)
    embedding=torch.nn.Embedding(_AUDIO_CODE_OFFSET+10,16).requires_grad_(False)
    pipe=SimpleNamespace(rvq_depth_decoder=decoder,language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)),
                         num_codebooks=8,audio_vocab_size=64)
    hidden=torch.randn(2,16);captured=[]
    handles=[head.register_forward_hook(lambda module,args,output:captured.append(output.detach().clone())) for head in decoder.audio_heads]
    with torch.no_grad():codes,_=_generate_depth_codes(pipe,hidden,torch.tensor([3,3]),torch.Generator().manual_seed(92))
    for handle in handles:handle.remove()
    assert torch.equal(codes[0],codes[1])
    sequential=torch.stack(captured,dim=1)
    expected=(sequential[1]+1.5*(sequential[0]-sequential[1])).log_softmax(-1)
    inputs=hidden[:,None].clone().requires_grad_(True)
    actual=log_probs(pipe,inputs,codes[:1])
    torch.testing.assert_close(actual[0],expected,atol=2e-6,rtol=2e-6)
    (-selected_mean(actual,codes[:1])).backward()
    assert torch.isfinite(inputs.grad).all() and all(inputs.grad[branch].abs().sum()>0 for branch in range(2))
    assert all(p.grad is None for p in decoder.parameters())
