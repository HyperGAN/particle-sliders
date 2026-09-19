"""Music metal instability → CPU 2-D / particle-bridge stressor.

Live Music 3 particle-bridge (2026-09-17): female stayed calm; metal kept
dipping ``cos_pos`` while ``d_adv → 0`` and ``||g||`` blew up. The particle
forward bug was already fixed. The remaining difference is prompt geometry:

- **female** = attribute insert (``female ``, ~7 chars) on a shared caption.
  Target deltas across the four rows stay aligned (min pairwise cos ~0.55).
- **metal** = genre rewrite (whole style block → ``modern melodic metal``).
  Same four neutrals, but positives pull in different directions (min pairwise
  cos ~0.33) with larger ``||pos − neu||``.

This module freezes that geometry as toy targets + caption-edit stats so the
suite can fail a recipe / prompt pack before another GPU catalog run.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders import particle_bridge_gan as shared

# Floors measured on Music 3 prepared targets (state-step100, seed 7 catalog).
MUSIC_FEMALE_MIN_PAIRWISE = 0.5486
MUSIC_METAL_MIN_PAIRWISE = 0.3250

# Toy gates: attribute-insert must clear the Music female floor; genre
# rewrite must sit at or below the Music metal floor.
ATTR_MIN_PAIRWISE = 0.54
REWRITE_MAX_PAIRWISE = 0.42

PROMPTS = Path(__file__).resolve().parents[2] / "analysis/uni16_fresh3400_20260912/prompts"


@dataclass(frozen=True)
class CaptionEdit:
    """Char-level neu↔pos rewrite size (no model)."""

    shared_prefix: int
    mid_from: int
    mid_to: int
    len_delta: int

    @property
    def mid_growth(self) -> int:
        return int(self.mid_to - self.mid_from)


def caption_edit(neutral: str, positive: str) -> CaptionEdit:
    pre = 0
    while pre < len(neutral) and pre < len(positive) and neutral[pre] == positive[pre]:
        pre += 1
    suf = 0
    while (
        suf < len(neutral) - pre
        and suf < len(positive) - pre
        and neutral[-(suf + 1)] == positive[-(suf + 1)]
    ):
        suf += 1
    a_mid = neutral[pre : len(neutral) - suf if suf else len(neutral)]
    b_mid = positive[pre : len(positive) - suf if suf else len(positive)]
    return CaptionEdit(pre, len(a_mid), len(b_mid), len(positive) - len(neutral))


def load_uni16_edits(name: str) -> list[CaptionEdit]:
    raw = yaml.safe_load((PROMPTS / f"{name}-train.yaml").read_text())
    return [caption_edit(row["neutral"], row["positive"]) for row in raw["rows"]]


def pairwise_delta_cos(deltas: torch.Tensor) -> torch.Tensor:
    """``deltas`` is ``[R, D]``; return ``[R, R]`` cosine gram of row deltas."""
    unit = F.normalize(deltas.float(), dim=-1)
    return unit @ unit.T


def alignment_stats(deltas: torch.Tensor) -> dict:
    gram = pairwise_delta_cos(deltas)
    r = gram.shape[0]
    off = gram[~torch.eye(r, dtype=torch.bool)]
    return dict(
        min_pairwise=float(off.min()),
        median_pairwise=float(off.median()),
        mean_norm=float(deltas.float().norm(dim=-1).mean()),
        row_norms=[float(x) for x in deltas.float().norm(dim=-1)],
        gram=gram,
    )


def attribute_insert_deltas(dim: int = 8, rows: int = 4) -> torch.Tensor:
    """Female-like: one shared axis, tiny per-row jitter."""
    axis = F.normalize(torch.randn(dim), dim=0)
    jitter = torch.randn(rows, dim) * 0.08
    return 1.4 * axis + jitter


def genre_rewrite_deltas(dim: int = 8, rows: int = 4) -> torch.Tensor:
    """Metal-like: weak shared genre pull + large per-row orthogonal rewrite."""
    shared_axis = F.normalize(torch.randn(dim), dim=0)
    out = []
    for i in range(rows):
        ortho = torch.randn(dim)
        ortho = ortho - (ortho @ shared_axis) * shared_axis
        ortho = F.normalize(ortho, dim=0)
        # Tuned so min pairwise lands near Music metal (~0.33), not female (~0.55).
        scale = 0.9 + 0.12 * i
        out.append(1.05 * shared_axis + 1.55 * ortho * scale)
    return torch.stack(out)


def rewrite_risk(deltas: torch.Tensor) -> dict:
    stats = alignment_stats(deltas)
    stats["flag"] = stats["min_pairwise"] < ATTR_MIN_PAIRWISE
    return stats


class _Student(nn.Module):
    """Same routed particle student as ``yue2_particle_exam.Student``."""

    def __init__(self, dim: int):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.down = nn.Linear(dim, 8, bias=False)
        self.up = nn.Linear(8, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=1)
        nn.init.zeros_(self.up.weight)
        self.bridge = shared.RoutedMLP(8, 8)
        self.scale = 0.0

    def forward(self, x):
        return x + self.scale * self.up(self.bridge(self.down(x), self.particles))

    @contextmanager
    def scaled(self, scale):
        previous = self.scale
        self.scale = float(scale)
        try:
            yield
        finally:
            self.scale = previous


class _Backend:
    def __init__(self, neus: torch.Tensor, network: _Student):
        self.neus = neus
        self.network = network
        self.model = nn.Module()
        self.model.register_parameter(
            "device_anchor", nn.Parameter(torch.zeros(()), requires_grad=False)
        )
        self.model.config = SimpleNamespace(hidden_size=int(neus.shape[-1]))

    def hidden(self, ids, checkpointing=False):
        return self.network(self.neus[ids[0]])[None, None]


@isolated_seed("seed")
def run_particle_bridge_toy(deltas: torch.Tensor, *, steps: int = 120, seed: int = 0) -> dict:
    """Short particle-bridge train; returns cos volatility / critic death proxies."""
    deltas = deltas.float()
    rows = int(deltas.shape[0])
    dim = int(deltas.shape[1])
    neus = torch.randn(rows, dim)
    targets = neus + deltas
    network = _Student(dim)
    backend = _Backend(neus, network)
    fixed = [
        dict(ids=[i], prefix_len=1, neutral=neus[i : i + 1], targets=targets[i : i + 1])
        for i in range(rows)
    ]
    torch.manual_seed(seed + 1000)
    critic, g, d = shared.build_game(network, targets)
    sampler = shared.BridgeSampler(rows, seed)

    def predict(index, phase):
        with network.scaled(1.0):
            pred = backend.hidden([index]).float().reshape(1, -1)
        return critic.normalize(pred)

    cos_hist = []
    d_adv_hist = []
    g_adv_hist = []
    grad_hist = []
    for step in range(1, steps + 1):
        metrics = shared.update(network, critic, g, d, targets, predict, sampler=sampler, step=step)
        with network.scaled(1.0), torch.no_grad():
            pred = network(neus)
            cos = F.cosine_similarity(pred - neus, deltas, dim=-1).mean()
        cos_hist.append(float(cos))
        d_adv_hist.append(float(metrics["d_adv"]))
        g_adv_hist.append(float(metrics["g_adv"]))
        grad_hist.append(float(metrics["grad_norm"]))

    cos_t = torch.tensor(cos_hist)
    late = cos_t[len(cos_t) // 2 :]
    return dict(
        steps=steps,
        seed=seed,
        cos_last=float(cos_t[-1]),
        cos_min=float(cos_t.min()),
        cos_std=float(cos_t.std(unbiased=False)),
        cos_late_median=float(late.median()),
        cos_late_std=float(late.std(unbiased=False)),
        d_adv_zero_frac=sum(1 for x in d_adv_hist if x <= 1e-8) / len(d_adv_hist),
        g_adv_median=float(torch.tensor(g_adv_hist).median()),
        g_adv_late_median=float(torch.tensor(g_adv_hist[len(g_adv_hist) // 2 :]).median()),
        grad_median=float(torch.tensor(grad_hist).median()),
        alignment=alignment_stats(deltas),
    )
