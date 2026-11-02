"""A bounded semantic-policy preference surrogate, not full audio likelihood."""
import torch
import torch.nn.functional as F


def guided_log_probs(hidden, head, *, offset, vocabulary, eos, cfg=1.5):
    """Both CFG branches; exclude text vocabulary, retain the end token.

    Do not apply sampling top-k: it has discontinuous support during training.
    Residual-code and acoustic likelihoods are not part of this surrogate.
    """
    weights = torch.cat((head[offset:offset+vocabulary], head[eos:eos+1]), 0)
    logits = (hidden @ weights.T).float()
    conditional, unconditional = logits[0], logits[1]
    return (unconditional + cfg * (conditional-unconditional)).log_softmax(-1)


def selected_mean(log_probs, tokens):
    if log_probs.shape[0] != len(tokens):
        raise ValueError('Each emitted semantic token needs its own preceding history')
    return log_probs.gather(-1, tokens.to(log_probs.device).long()[:, None]).mean()


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta=5.):
    margin = (chosen-reference_chosen)-(rejected-reference_rejected)
    return -F.logsigmoid(beta * margin)


def detached_pair_coefficients(chosen, rejected, ref_chosen, ref_rejected, beta=5., weight=1.):
    """Exact derivatives of preference_loss, allowing one take's graph at a time."""
    margin = (chosen-ref_chosen)-(rejected-ref_rejected)
    coefficient = -weight * beta * torch.sigmoid(-beta*margin.detach())
    return coefficient, -coefficient


def reference_kl(log_probs, reference_log_probs):
    reference = reference_log_probs.to(log_probs).log_softmax(-1)
    return (reference.exp()*(reference-log_probs)).sum(-1).mean()


def screen_summary(rows, controls):
    expected = set(controls)
    if len(rows) != len(expected) or {(r['family'],r['seed']) for r in rows} != expected:
        return dict(passed=False,reason='incomplete declared screen')
    if any(r['status']!='complete' or not r.get('reward',{}).get('valid') for r in rows):
        return dict(passed=False,reason='invalid output retained')
    deltas = [r['reward']['scalar']-controls[(r['family'],r['seed'])]['reward']['scalar'] for r in rows]
    wins = sum(x>0 for x in deltas)
    mean = sum(deltas)/len(deltas)
    worst = min(deltas)
    return dict(passed=wins>=3 and mean>=.02 and worst>=-.30, wins=wins,pairs=len(deltas),
                mean_ce_gain=mean,worst_ce_delta=worst,deltas=deltas,
                rule='at least 3/4 wins, mean gain >= .02, no loss worse than -.30 CE',
                interpretation='small development screen, not a shipping claim')
