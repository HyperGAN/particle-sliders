"""TIES and joint-SVD alignment, independently implemented for this study.

Trimming/sign election are per projection. Disjoint mean excludes zeros.
This is explicitly a per-projection, norm-matched adaptation, not the papers'
globally flattened benchmark. KnOTS uses magnitude-bearing S V blocks.
"""
import torch


def ties(values, density=.3):
    if not 0 < density <= 1:
        raise ValueError('density must be in (0, 1]')
    if not values or any(v.shape != values[0].shape for v in values):
        raise ValueError('TIES requires equally shaped updates')
    stack = torch.stack([v.flatten() for v in values])
    if not torch.isfinite(stack).all():
        raise ValueError('Nonfinite input')
    if density < 1:
        # Match official threshold convention, including threshold ties.
        k = max(1, stack.shape[1]-int(stack.shape[1]*density))
        threshold = stack.abs().kthvalue(k, dim=1, keepdim=True).values
        stack *= stack.abs() >= threshold
    signs = stack.sum(0).sign()
    majority = signs.sum().sign()
    signs = torch.where(signs == 0, majority, signs)
    selected = torch.where(torch.where(signs > 0, stack > 0, stack < 0), stack, 0.)
    merged = selected.sum(0) / (selected != 0).sum(0).clamp_min(1)
    return merged.reshape(values[0].shape)


def joint_basis(ups, downs, cutoff=1e-5):
    """Exact thin SVD of [B1 A1, B2 A2, ...], without a dense concatenation.

    QR of [B1,B2] reduces the SVD to at most summed LoRA rank rows. FP64
    decomposition matches the reference precision; output remains FP32.
    Caller bakes alpha/rank and desired input weights into ups.
    """
    if len(ups) != len(downs) or not ups:
        raise ValueError('Missing factors')
    if len({b.shape[0] for b in ups}) != 1 or len({a.shape[1] for a in downs}) != 1:
        raise ValueError('Incompatible host shapes')
    q, r = torch.linalg.qr(torch.cat(ups, dim=1).double(), mode='reduced')
    blocks = []
    start = 0
    for b, a in zip(ups, downs):
        rank = b.shape[1]
        blocks.append(r[:, start:start+rank] @ a.double())
        start += rank
    u, s, vh = torch.linalg.svd(torch.cat(blocks, dim=1), full_matrices=False)
    keep = s > cutoff
    left = (q @ u[:, keep]).float()
    aligned = (s[keep, None]*vh[keep]).float()
    return left, list(aligned.split(downs[0].shape[1], dim=1)), s[keep].float()


def knots_ties(ups, downs, density=.3):
    left, blocks, singular = joint_basis(ups, downs)
    if not len(singular):
        return torch.zeros((ups[0].shape[0], downs[0].shape[1])), 0
    return left @ ties(blocks, density), len(singular)


def match_norm(delta, reference):
    target = torch.linalg.vector_norm(reference, dtype=torch.float64)
    before = torch.linalg.vector_norm(delta, dtype=torch.float64)
    if not torch.isfinite(target+before):
        raise ValueError('Nonfinite update norm')
    if before == 0 and target != 0:
        raise ValueError('Cannot normalize a zero merged update to nonzero strength')
    scale = float(target/before) if before else 0.
    result = delta*scale
    actual = torch.linalg.vector_norm(result, dtype=torch.float64)
    cosine = float(torch.sum(result*reference, dtype=torch.float64)/(actual*target)) if target else 0.
    return result, dict(reference_norm=float(target), raw_norm=float(before),
                       scale=scale, matched_norm=float(actual), cosine_to_linear=cosine)
