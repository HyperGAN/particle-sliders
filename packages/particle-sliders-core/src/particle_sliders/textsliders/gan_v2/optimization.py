"""Gradient accounting, functional LoRA step limits, and effective-weight EMA."""
from __future__ import annotations

import math
import torch


def finite_positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def gradient_vector(loss, parameters, *, retain_graph=True):
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(p).flatten() for p in parameters])
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten()
                      for p, g in zip(parameters, grads)])


def current_gradient(parameters):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad.detach()).flatten()
                      for p in parameters])


def install_gradient(parameters, vector):
    if vector.numel() != sum(p.numel() for p in parameters) or not torch.isfinite(vector).all():
        raise FloatingPointError("Invalid assembled generator gradient")
    start = 0
    for p in parameters:
        p.grad = vector[start:start + p.numel()].view_as(p).clone()
        start += p.numel()


def add_bounded_fm(other_gradient, fm_gradient, maximum):
    """Limit the weighted full-batch FM parameter gradient, after accumulation."""
    finite_positive(maximum, "FM gradient limit")
    if not torch.isfinite(other_gradient).all() or not torch.isfinite(fm_gradient).all():
        raise FloatingPointError("Nonfinite component gradient")
    norm = float(fm_gradient.double().norm())
    factor = min(1., maximum / max(norm, 1e-30))
    return other_gradient + factor * fm_gradient, {"raw_norm": norm, "factor": factor,
                                                  "applied_norm": factor * norm}


def gradient_statistics(vectors):
    norms = {key: float(value.double().norm()) for key, value in vectors.items()}
    if not all(math.isfinite(v) for v in norms.values()):
        raise FloatingPointError("Nonfinite gradient accounting")
    cosine = {}
    keys = list(vectors)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            denominator = norms[a] * norms[b]
            cosine[f"{a}:{b}"] = (None if denominator == 0 else
                float(torch.dot(vectors[a].double(), vectors[b].double()) / denominator))
    return {"norms": norms, "cosine": cosine, "scope": "weighted full minibatch, before clipping"}


def cosine_rate(step, *, origin, horizon, hold_fraction=.6, floor=.05):
    """Absolute update-index schedule. Extending a run cannot move its horizon."""
    if (horizon <= origin or step < 0 or not 0 <= hold_fraction < 1
            or not 0 < floor <= 1 or not math.isfinite(floor)):
        raise ValueError("Invalid fixed schedule")
    phase = min(1., max(0., (step - origin) / (horizon - origin)))
    if phase <= hold_fraction:
        return 1.
    phase = (phase - hold_fraction) / (1 - hold_fraction)
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * phase))


def factor_norm_squared(b, a):
    """||BA||_F² without materializing a dense adapted matrix."""
    b, a = b.double(), a.double()
    return ((b.T @ b) * (a @ a.T)).sum().clamp_min(0.)


def factor_distance_squared(b, a, old_b, old_a):
    # Stable difference representation: dB*A_old + B_new*dA. No subtraction
    # of almost equal large squared norms, and no dense 4096-square matrix.
    left = torch.cat([b.double() - old_b.double(), b.double()], dim=1)
    right = torch.cat([old_a.double(), a.double() - old_a.double()], dim=0)
    return factor_norm_squared(left, right)


def adapter_pairs(network):
    return [(m.lora_up.weight, m.lora_down.weight, float(m.scale)) for m in network.unet_loras]


@torch.no_grad()
def snapshot_pairs(pairs):
    return [(b.detach().clone(), a.detach().clone(), scale) for b, a, scale in pairs]


@torch.no_grad()
def effective_norm(pairs):
    return math.sqrt(sum(float(factor_norm_squared(b, a)) * scale**2 for b, a, scale in pairs))


@torch.no_grad()
def effective_distance(pairs, before):
    if len(pairs) != len(before):
        raise ValueError("Adapter topology changed")
    if any(p[2] != q[2] for p, q in zip(pairs, before)):
        raise ValueError("Adapter scaling changed during a step")
    return math.sqrt(sum(float(factor_distance_squared(b, a, old_b, old_a)) * scale**2
                         for (b, a, scale), (old_b, old_a, old_scale) in zip(pairs, before)
                         ))


@torch.no_grad()
def limit_effective_step(pairs, before, *, maximum):
    """Shorten a finite factor update until ||delta(BA)|| fits the budget.

    The norm is invariant to the initial factor gauge. The interpolation path
    and optimizer can still depend on factorization. Adam moments are retained.
    """
    finite_positive(maximum, "Effective step limit")
    raw = effective_distance(pairs, before)
    if not math.isfinite(raw):
        raise FloatingPointError("Nonfinite effective adapter step")
    if raw <= maximum:
        return {"proposed_norm": raw, "actual_norm": raw, "factor": 1., "maximum": maximum}
    proposed = snapshot_pairs(pairs)
    # Along factor interpolation delta_W(t)=t*L+t²*Q. Precompute its squared
    # norm polynomial once instead of 24 costly model-wide tensor reductions.
    c2=c3=c4=0.
    for (b,a,scale),(old_b,old_a,_) in zip(proposed,before):
        db,da=b.double()-old_b.double(),a.double()-old_a.double()
        left=torch.cat([db,old_b.double()],1)
        right=torch.cat([old_a.double(),da],0)
        c2+=scale**2*float(factor_norm_squared(left,right))
        c3+=2*scale**2*float(((left.T@db)*(right@da.T)).sum())
        c4+=scale**2*float(factor_norm_squared(db,da))

    def move(fraction):
        for (b, a, _), (old_b, old_a, _), (new_b, new_a, _) in zip(pairs, before, proposed):
            b.copy_(old_b + fraction * (new_b - old_b))
            a.copy_(old_a + fraction * (new_a - old_a))

    low, high = 0., 1.
    for _ in range(24):
        middle = (low + high) / 2
        squared=middle**2*(c2+middle*c3+middle**2*c4)
        if squared <= maximum**2:
            low = middle
        else:
            high = middle
    # Leave a small margin for float32 factor rounding, then verify the real
    # tensor displacement. The norm criterion never relies only on the model.
    low*=.99999
    move(low)
    actual = effective_distance(pairs, before)
    for _ in range(8):
        if actual<=maximum:break
        low*=.5;move(low);actual=effective_distance(pairs,before)
    if actual > maximum:
        move(0.);actual=effective_distance(pairs,before);low=0.
    if actual>maximum:raise FloatingPointError("Effective step failed its postcondition")
    return {"proposed_norm": raw, "actual_norm": actual, "factor": low, "maximum": maximum}


@torch.no_grad()
def compress_factors(b, a, rank):
    """Best truncated SVD of BA via thin QR; report discarded squared energy."""
    if rank <= 0 or b.shape[1] != a.shape[0]:
        raise ValueError("Invalid low-rank factors")
    qb, rb = torch.linalg.qr(b.double(), mode="reduced")
    qa, ra = torch.linalg.qr(a.double().T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    keep = min(rank, len(s))
    root = s[:keep].clamp_min(0).sqrt()
    left = (qb @ u[:, :keep]) * root
    right = root[:, None] * (vh[:keep] @ qa.T)
    total = float(s.square().sum())
    discarded = float(s[keep:].square().sum())
    return left.float(), right.float(), {"discarded_squared_norm": discarded,
        "total_squared_norm": total, "relative_frobenius_error": math.sqrt(discarded / max(total, 1e-30))}


class EffectiveEMA:
    """EMA of effective delta weights, approximated with a recorded rank budget.

    It averages BA, not B and A independently. Updates may be thinned in time;
    decay is raised to the number of intervening optimizer updates. Between
    observations it assumes the newest state represents that interval.
    """
    def __init__(self, *, rank=32, decay=.995):
        if rank <= 0 or not 0 <= decay < 1:
            raise ValueError("Invalid EMA settings")
        self.rank, self.decay = rank, decay
        self.factors, self.step, self.last_error = {}, None, None

    @torch.no_grad()
    def update(self, network, step):
        if self.step is not None and step <= self.step:
            raise ValueError("EMA observations must advance")
        rate = 0. if self.step is None else self.decay ** (step - self.step)
        discarded = total = 0.
        for m in network.unet_loras:
            b = m.lora_up.weight.detach().cpu().float() * float(m.scale)
            a = m.lora_down.weight.detach().cpu().float()
            if m.lora_name in self.factors:
                old_b, old_a = self.factors[m.lora_name]
                b = torch.cat([math.sqrt(rate) * old_b, math.sqrt(1 - rate) * b], 1)
                a = torch.cat([math.sqrt(rate) * old_a, math.sqrt(1 - rate) * a], 0)
            b, a, errors = compress_factors(b, a, self.rank)
            self.factors[m.lora_name] = (b, a)
            discarded += errors["discarded_squared_norm"]
            total += errors["total_squared_norm"]
        self.step = step
        self.last_error = math.sqrt(discarded / max(total, 1e-30))

    def state_dict(self):
        return dict(rank=self.rank, decay=self.decay, step=self.step,
                    factors=self.factors, last_error=self.last_error)

    def load_state_dict(self, state):
        if (state["rank"], state["decay"]) != (self.rank, self.decay):
            raise ValueError("EMA configuration changed")
        self.factors, self.step, self.last_error = state["factors"], state["step"], state["last_error"]

    def inference_state(self):
        output = {}
        for name, (b, a) in self.factors.items():
            # Uniform exported rank, with zero padding for narrow matrices.
            left = F_pad(b, (0, self.rank - b.shape[1]))
            right = F_pad(a, (0, 0, 0, self.rank - a.shape[0]))
            output[f"{name}.lora_up.weight"] = left.contiguous()
            output[f"{name}.lora_down.weight"] = right.contiguous()
            output[f"{name}.alpha"] = torch.tensor(float(self.rank))
        return output


def F_pad(value, padding):
    return torch.nn.functional.pad(value, padding)
