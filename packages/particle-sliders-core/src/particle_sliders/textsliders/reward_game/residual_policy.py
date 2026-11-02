"""Differentiable residual-code teacher forcing, separate from audio likelihood.

The frozen depth decoder consumes the LM hidden state plus already chosen frame
codes. Gradients stay connected to both CFG branches of the LM hidden state.
Callers must first verify all eight targets against exact captured feedback.
"""
import torch


def log_probs(pipe,hidden,codes,*,cfg=1.5,semantic_offset=None):
    """Return [positions, seven residual codebooks, vocabulary] log probabilities.

    Hidden states are [two CFG branches, positions, hidden width]. Codes are
    [positions, eight codebooks], after excluding the discarded warmup frame.
    This uses untruncated probabilities; it does not reproduce sampling top-k.
    """
    if semantic_offset is None:
        from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CODE_OFFSET
        semantic_offset=_AUDIO_CODE_OFFSET
    if hidden.ndim!=3 or hidden.shape[0]!=2 or codes.shape!=(hidden.shape[1],8):
        raise ValueError('Expected two CFG branches and all eight aligned codebooks')
    decoder=pipe.rvq_depth_decoder
    if len(decoder.audio_heads)!=7:raise ValueError('Expected seven residual heads')
    vocabulary=decoder.audio_heads[0].out_features
    if torch.any(codes<0) or torch.any(codes[:,1:]>=vocabulary):raise ValueError('Invalid captured residual target')
    positions=hidden.shape[1];targets=codes.to(device=hidden.device,dtype=torch.long)
    repeated=targets.repeat(2,1)
    semantic=pipe.language_model.model.embed_tokens(repeated[:,0]+semantic_offset)
    offsets=torch.arange(6,device=hidden.device)*vocabulary
    residual=decoder.audio_embeddings(repeated[:,1:7]+offsets)
    inputs=torch.cat((hidden.reshape(2*positions,1,-1),semantic[:,None],residual),dim=1)
    output=decoder(decoder.projection(inputs))[:,1:]
    branches=torch.stack([head(output[:,index]) for index,head in enumerate(decoder.audio_heads)],dim=1)
    branches=branches.reshape(2,positions,7,vocabulary).float()
    guided=branches[1]+cfg*(branches[0]-branches[1])
    return guided.log_softmax(-1)


def selected_mean(logp,codes):
    return logp.gather(-1,codes[:,1:].to(logp.device,dtype=torch.long)[...,None]).mean()
