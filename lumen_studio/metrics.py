"""Fixed full-field projections and noise; raw residual and best-so-far separate."""
from collections import defaultdict
import math

import torch


def make_fixtures(dimension, count, seed=7013):
    generator = torch.Generator().manual_seed(seed)
    projections = torch.randn(dimension, 256, generator=generator)
    projections /= projections.norm(dim=0, keepdim=True)
    noise = torch.randn(count, dimension, generator=generator)
    return dict(seed=seed, projections=projections, noise=noise)


def swd(a, b, projections):
    pa, pb = a.float() @ projections, b.float() @ projections
    return float((pa.sort(dim=0).values - pb.sort(dim=0).values).abs().mean())


def measure(student, teacher, scale, fixtures):
    student, teacher = student.float(), teacher.float()
    error = (student - teacher) / scale
    s, t = student / scale, teacher / scale
    denominator = t.square().sum(-1).clamp_min(1e-12)
    gain = (s * t).sum(-1) / denominator
    orthogonal = s - gain[:, None] * t
    p = fixtures["projections"].to(s.device)
    n = fixtures["noise"][:len(s)].to(s.device)
    return dict(residual_rms=float(error.square().mean().sqrt()),
        residual_p95=float(error.abs().quantile(.95)),
        edit_cosine=float(torch.nn.functional.cosine_similarity(s, t, dim=-1).mean()),
        edit_gain=float(gain.mean()), orthogonal_rms=float(orthogonal.square().mean().sqrt()),
        teacher_swd_256=swd(s, t, p),
        game_swd={str(level): swd(n * level + error, n * level, p) for level in (.5, 1., 2.)})


def residual_breakdown(errors, records):
    groups = defaultdict(list)
    for error, rec in zip(errors, records, strict=True):
        row = rec["row"]
        squared = float(error.float().square().mean())
        for field, value in (("definition", row["definition"]), ("character", row["character"]),
                             ("framing", row["shared"]["framing"]), ("timestep", rec["position"]),
                             ("bare", row.get("bare", False))):
            groups[f"{field}:{value}"].append(squared)
    return {k: math.sqrt(sum(v) / len(v)) for k, v in groups.items()}


def best_so_far(values):
    best = math.inf
    result = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Non-finite residual")
        best = min(best, value)
        result.append(best)
    return result
