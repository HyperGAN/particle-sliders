"""Paired-error ParticleGAN game from model-glue's anneal-routed benchmark.

Reference: model-glue df70ccb, configs/particle-toy-20260917.json;
ParticleGAN 441fdf42. No output reconstruction loss is used.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from analysis.slider2d.adv import rp_d_loss, rp_g_loss, make_grad_regularizer
from conceptmod.textsliders.lm_adv import _SetBlock, param_grad_norm

REFERENCE = dict(particles=128, z_dim=4, width=48, router_width=16,
    batch_size=64, g_lr=.0006, d_lr=.0009, particle_lr=.006,
    betas=(0., .999), vic_coeff=1., cap_coeff=1., cap_kappa=1., cap_every=4,
    ema=.995, noise_start=1., noise_decay_steps=8000, noise_floor=.03,
    target_std_floor=1e-4,
    # Paired-edit whitening: std of (target−neutral), then a scalar gain so
    # median per-row normalized edit RMS equals edit_rms_target. Noise starts
    # at edit_rms / edit_noise_ratio so initial edit/noise matches the metal
    # pilot (~0.28). Absolute-state whitening buried voice edits (~0.05).
    edit_rms_target=1., edit_noise_ratio=.28,
    # Sigma hold: the anneal stops at edit_rms * noise_hold_ratio. Per run, not a
    # constant: the paired critic stays balanced only while sigma is ~3-4x the
    # reachable residual (pop at 1.0 hold: ratio 3.7, flat d_loss 0.45; gender v2
    # at 1.0: ratio 1.9, d_loss eroding 0.45->0.18 with G grad norm 5->15).
    noise_hold_ratio=1.3,
    # Critic: 'sn_mlp' is the near-term default (spectral-norm residual MLP).
    # 'gmix' is the global-mix hybrid ablation (H→T·W then attend).
    # Legacy: 'mlp' thin head; 'mix' fat first layer; patch/bottleneck/…
    critic='mlp', critic_patch=64, critic_width=48, critic_layers=1, critic_heads=4,
    critic_tokens=8, critic_queries=4, critic_rank=64, critic_score_bound=8.0)


def mlp(inputs, outputs, width):
    layers = []
    for n in (inputs, width, width):
        layers.extend((nn.Linear(n, width), nn.LeakyReLU(.2)))
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


def bound_score(score, bound):
    """Keep relativistic logits in the softplus-sensitive band."""
    bound = float(bound)
    if bound <= 0:
        return score
    return bound * torch.tanh(score / bound)


def register_paired_error_norm(module, targets, neutrals=None):
    """Whiten by paired edit, not absolute target-state distribution.

    When ``neutrals`` is provided, per-coordinate scale is ``std(target−neutral)``,
    then a scalar gain pins median row edit RMS to ``edit_rms_target``. Sets
    ``module.noise_start = edit_rms / edit_noise_ratio``. Without neutrals,
    keeps legacy absolute-state whitening and ``noise_start = REFERENCE['noise_start']``.
    """
    if targets.ndim != 2 or len(targets) < 2 or not torch.isfinite(targets).all():
        raise ValueError('Expected finite training targets [rows, width]')
    floor = float(REFERENCE['target_std_floor'])
    if neutrals is None:
        scale = targets.std(0).clamp_min(floor)
        module.register_buffer('target_mean', targets.mean(0))
        module.register_buffer('target_std', scale)
        module.register_buffer('input_scale', scale.square().mean().sqrt())
        module.register_buffer('edit_rms', module.input_scale.detach().clone())
        module.noise_start = float(REFERENCE['noise_start'])
        module.normalization = 'absolute_target_per_coordinate_sample_std'
        return module
    if neutrals.shape != targets.shape or not torch.isfinite(neutrals).all():
        raise ValueError('Expected finite neutrals matching targets')
    edits = targets - neutrals
    scale = edits.std(0).clamp_min(floor)
    normed = edits / scale
    row_rms = normed.pow(2).mean(-1).sqrt()
    median_rms = row_rms.median().clamp_min(floor)
    gain = float(REFERENCE['edit_rms_target']) / float(median_rms)
    scale = scale / gain
    normed = edits / scale
    edit_rms = normed.pow(2).mean().sqrt().detach()
    module.register_buffer('target_mean', targets.mean(0))
    module.register_buffer('target_std', scale)
    module.register_buffer('edit_rms', edit_rms.to(dtype=torch.float32))
    module.register_buffer('input_scale', module.edit_rms.clone())
    ratio = float(REFERENCE['edit_noise_ratio'])
    module.noise_start = max(float(edit_rms) / ratio, float(REFERENCE['noise_floor']))
    module.normalization = 'paired_edit_per_coordinate_std_median_rms_gain'
    return module


class RoutedMLP(nn.Module):
    """Every input uses the same softmax route at train and inference time."""
    def __init__(self, inputs, outputs, z_dim=4, width=48, router_width=16):
        super().__init__()
        self.router = mlp(inputs, z_dim, router_width)
        self.net = mlp(inputs + z_dim, outputs, width)

    def forward(self, x, particles):
        q = self.router(x)
        weights = (q @ particles.T / math.sqrt(particles.shape[1])).softmax(-1)
        z = weights @ particles
        return self.net(torch.cat((x, z), dim=-1))


def particle_vic(z):
    """Pinned upstream VICRegLikeLoss (target_std=1, eps=1e-4).

    Sample standard deviation and off-diagonal covariance, on particles only.
    Source SHA256 d2563bcab93d443cbf2cf2260946a3ba470f1fc584c231b704ec5ac154f668c4.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    variance = F.relu(1. - std).mean()
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / (len(z) - 1)
    dim = z.shape[1]
    if dim == 1:
        return variance
    off = covariance.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    return variance + off.square().sum() / dim


def noise_std(step, start=None, decay_steps=None, hold=None):
    """Geometric anneal from ``start`` to the floor, done at ``decay_steps``.

    The reference horizon is 8000. A training run passes its own step budget
    so the floor is reached when that run ends, not thousands of steps later.

    ``hold`` is the normalized residual scale (about 1). Once the curve would
    pass under it, sigma stays there. A start already at or below the hold
    is left on the geometric curve, so the hold never raises noise.
    """
    if step < 0:
        raise ValueError('Step must be nonnegative')
    start = float(REFERENCE['noise_start'] if start is None else start)
    floor = float(REFERENCE['noise_floor'])
    horizon = float(REFERENCE['noise_decay_steps'] if decay_steps is None else decay_steps)
    if start <= 0 or horizon <= 0 or floor <= 0:
        raise ValueError('noise start, floor, and decay horizon must be positive')
    t = min(step / horizon, 1.)
    sigma = start * (floor / start) ** t
    if hold is None:
        return sigma
    hold = float(hold)
    if hold <= 0:
        raise ValueError('noise hold must be positive')
    if start > hold:
        sigma = max(sigma, hold)
    return sigma


class BridgeSampler:
    """CPU random streams, fresh rows/noise for each D and G minibatch."""
    def __init__(self, count, seed):
        if count < 2:
            raise ValueError('At least two training targets are needed for sample std')
        self.count = count
        self.rngs = {name: torch.Generator().manual_seed(seed + offset)
                     for name, offset in [('data', 10), ('noise', 30), ('vic', 40)]}

    def batch(self, dim, device, sigma):
        rows = torch.randint(self.count, (REFERENCE['batch_size'],), generator=self.rngs['data'])
        noise = torch.randn(len(rows), dim, generator=self.rngs['noise']).to(device) * sigma
        return rows, noise

    def vic_rows(self, count, device):
        return torch.randperm(count, generator=self.rngs['vic'])[:64].to(device)

    def state_dict(self):
        return dict(count=self.count, rngs={k: r.get_state() for k, r in self.rngs.items()})

    def load_state_dict(self, state):
        if state['count'] != self.count or state['rngs'].keys() != self.rngs.keys():
            raise ValueError('Incompatible particle-bridge sampler')
        for name, value in state['rngs'].items():
            self.rngs[name].set_state(value)


class ErrorCritic(nn.Module):
    """Thin MLP over the full error vector (legacy Yue2 particle-bridge default)."""

    def __init__(self, targets, *, neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        self.net = mlp(targets.shape[1], 1, REFERENCE['width'])

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        return self.net(coordinates).squeeze(-1)


class _ResidualSNBlock(nn.Module):
    """Spectral-normalized residual MLP block: x + Linear(SiLU(Linear(x)))."""

    def __init__(self, width):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(width, width))
        self.fc2 = spectral_norm(nn.Linear(width, width))

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(x)))


class ResidualSNErrorCritic(nn.Module):
    """Global residual MLP with spectral norm and bounded score.

    whitened error → Linear(H, W) → SiLU → residual block(s) → Linear(W, W/2)
    → SiLU → Linear(W/2, 1), each linear spectrally normalized; score bounded
    with ``bound * tanh(score / bound)``. No R1/R2 — ParticleGAN keeps b_cap.
    """

    def __init__(self, targets, *, width=None, layers=None, neutrals=None, score_bound=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        width = int(128 if width is None else width)
        layers = int(1 if layers is None else layers)
        if width < 2 or layers < 1:
            raise ValueError('width/layers must be positive (width>=2)')
        mid = max(width // 2, 1)
        self.width = width
        self.in_proj = spectral_norm(nn.Linear(dim, width))
        self.blocks = nn.ModuleList([_ResidualSNBlock(width) for _ in range(layers)])
        self.mid = spectral_norm(nn.Linear(width, mid))
        self.head = spectral_norm(nn.Linear(mid, 1))
        self.score_bound = float(
            REFERENCE.get('critic_score_bound') if score_bound is None else score_bound
        )

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        h = F.silu(self.in_proj(coordinates))
        for block in self.blocks:
            h = block(h)
        h = F.silu(self.mid(h))
        return bound_score(self.head(h).squeeze(-1), self.score_bound)


class PatchErrorCritic(nn.Module):
    """Feature-patch transformer over the paired-error vector.

    Chunks ``R^H`` into ordered patches, projects each to ``critic_width``,
    runs the same ``_SetBlock`` stack as ``SpanTransformerD`` (handwritten
    attention for ``b_cap``), mean-pools, scores. Same normalize contract as
    ``ErrorCritic`` so the particle-bridge game is unchanged.

    Contiguous patches assume local structure across hidden dims — that is a
    bad inductive bias for LM last-hidden errors. Prefer ``gmix`` / ``sn_mlp``.
    """

    def __init__(self, targets, *, patch=None, width=None, layers=None, heads=None, neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        patch = int(REFERENCE['critic_patch'] if patch is None else patch)
        width = int(REFERENCE['critic_width'] if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if patch < 1 or width < 1 or layers < 1:
            raise ValueError('patch/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.patch = patch
        self.width = width
        n_patches = (dim + patch - 1) // patch
        self.proj = nn.Linear(patch, width)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(width, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def _tokens(self, coordinates):
        # coordinates: (B, H) → (B, T, patch), zero-pad the tail patch.
        bsz, dim = coordinates.shape
        pad = (self.patch - dim % self.patch) % self.patch
        x = F.pad(coordinates, (0, pad)) if pad else coordinates
        return x.view(bsz, -1, self.patch)

    def forward(self, coordinates):
        patches = self._tokens(coordinates)
        tokens = self.proj(patches) + self.pos[:, : patches.shape[1]]
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)
        pooled = self.out_norm(tokens.mean(dim=1))
        return self.head(pooled).squeeze(-1)


class MixErrorCritic(nn.Module):
    """Nonlocal mix-then-attend critic (fat first layer — prefer bottleneck).

    ``Linear(H → T·W)`` is T× an MLP stem at H=4096, so D can memorize
    metal rewrite coordinates before G moves. Kept for ablations; new
    defaults should use ``GlobalMixErrorCritic`` / ``ResidualSNErrorCritic``.
    """

    def __init__(self, targets, *, tokens=None, width=None, layers=None, heads=None, neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(REFERENCE.get('critic_tokens', 16) if tokens is None else tokens)
        width = int(REFERENCE['critic_width'] if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if tokens < 1 or width < 1 or layers < 1:
            raise ValueError('tokens/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.width = width
        self.mix = nn.Linear(dim, tokens * width)
        self.pos = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(width, 1)
        self.score_bound = float(REFERENCE.get('critic_score_bound') or 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Match the early fat-mix baseline (U-curve to cos~0.93 by 300):
                # plain xavier, live head — zero-head / gain<1 kill the early spike
                # that run actually climbed out of.
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.mix.bias, std=0.02)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        tokens = self.mix(coordinates).view(bsz, self.tokens, self.width) + self.pos
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)
        score = self.head(self.out_norm(tokens.mean(dim=1))).squeeze(-1)
        return bound_score(score, self.score_bound)


class GlobalMixErrorCritic(nn.Module):
    """Learned global mix before tokenizing, then attention (recommended hybrid).

    ``2048 → Linear(H, T·W) → T tokens → 1–2 attn blocks → mean+max pool →
    bounded score``. Every token is a mixture of the full hidden state, so
    token adjacency is learned rather than invented by patching.
    """

    def __init__(self, targets, *, tokens=None, width=None, layers=None, heads=None,
                 neutrals=None, score_bound=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(8 if tokens is None else tokens)
        width = int(48 if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if tokens < 1 or width < 1 or layers < 1:
            raise ValueError('tokens/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.width = width
        self.mix = nn.Linear(dim, tokens * width)
        self.pos = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(2 * width, 1)
        self.score_bound = float(
            REFERENCE.get('critic_score_bound') if score_bound is None else score_bound
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.mix.bias, std=0.02)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        tokens = self.mix(coordinates).view(bsz, self.tokens, self.width) + self.pos
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)
        mean = self.out_norm(tokens.mean(dim=1))
        peak = self.out_norm(tokens.amax(dim=1))
        return bound_score(self.head(torch.cat((mean, peak), dim=-1)).squeeze(-1), self.score_bound)


class BottleneckMixErrorCritic(nn.Module):
    """Capacity-matched transformer: H→W stem, then token expand in W-space.

    Same down-projection cost as ``ErrorCritic`` / ``SpanTransformerD``; the
    nonlocal part is a cheap ``W → T·W`` map + attention. Hypothesis: fat
    ``H → T·W`` was the metal collapse, not attention itself.
    """

    def __init__(self, targets, *, tokens=None, width=None, layers=None, heads=None, neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(REFERENCE.get('critic_tokens', 8) if tokens is None else tokens)
        width = int(REFERENCE['critic_width'] if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if tokens < 1 or width < 1 or layers < 1:
            raise ValueError('tokens/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.width = width
        self.stem = nn.Sequential(nn.Linear(dim, width), nn.LeakyReLU(0.2))
        self.expand = nn.Linear(width, tokens * width)
        self.pos = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(width, 1)
        self.score_bound = float(REFERENCE.get('critic_score_bound') or 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.expand.bias, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        h = self.stem(coordinates)
        tokens = self.expand(h).view(bsz, self.tokens, self.width) + self.pos
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)
        score = self.head(self.out_norm(tokens.mean(dim=1))).squeeze(-1)
        return bound_score(score, self.score_bound)


class LowRankMixErrorCritic(nn.Module):
    """``H → r → T·W`` then attend. Caps the first-layer rank below full H."""

    def __init__(self, targets, *, tokens=None, width=None, layers=None, heads=None, rank=None,
                 neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(REFERENCE.get('critic_tokens', 8) if tokens is None else tokens)
        width = int(REFERENCE['critic_width'] if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        rank = int(REFERENCE.get('critic_rank', 64) if rank is None else rank)
        if min(tokens, width, layers, rank) < 1:
            raise ValueError('tokens/width/layers/rank must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.width = width
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, tokens * width)
        self.pos = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(width, 1)
        self.score_bound = float(REFERENCE.get('critic_score_bound') or 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.up.bias, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        tokens = self.up(self.down(coordinates)).view(bsz, self.tokens, self.width) + self.pos
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)
        score = self.head(self.out_norm(tokens.mean(dim=1))).squeeze(-1)
        return bound_score(score, self.score_bound)


class HybridErrorCritic(nn.Module):
    """MLP score + gated bottleneck-attention residual.

    Starts near the winning MLP (gate≈0); attention can grow if it helps
    metal's dispersed rewrite geometry without owning the early game.
    """

    def __init__(self, targets, *, tokens=None, width=None, layers=None, heads=None, neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        width = int(REFERENCE['critic_width'] if width is None else width)
        self.mlp = mlp(targets.shape[1], 1, width)
        self.attn = BottleneckMixErrorCritic(
            targets, tokens=tokens, width=width, layers=layers, heads=heads, neutrals=neutrals
        )
        # Share normalization with the outer module (attn has its own copies).
        self.gate = nn.Parameter(torch.tensor(-2.0))  # softplus(-2)≈0.13
        self.score_bound = float(REFERENCE.get('critic_score_bound') or 0.0)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        base = self.mlp(coordinates).squeeze(-1)
        # Bypass attn's outer bound; bound the sum once.
        raw_attn = self.attn.head(
            self.attn.out_norm(
                self._attn_tokens(coordinates).mean(dim=1)
            )
        ).squeeze(-1)
        score = base + torch.nn.functional.softplus(self.gate) * raw_attn
        return bound_score(score, self.score_bound)

    def _attn_tokens(self, coordinates):
        bsz = coordinates.shape[0]
        h = self.attn.stem(coordinates)
        tokens = self.attn.expand(h).view(bsz, self.attn.tokens, self.attn.width) + self.attn.pos
        for block in self.attn.blocks:
            tokens = block(tokens, key_padding_mask=None)
        return tokens


class QueryErrorCritic(nn.Module):
    """Learned queries cross-attending to a dense mix of the error vector.

    ``Linear(H → K·W)`` builds a key/value bank from the full vector; ``Q``
    learnable queries attend (handwritten, b_cap-safe), then score.
    """

    def __init__(self, targets, *, tokens=None, queries=None, width=None, layers=None, heads=None,
                 neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(REFERENCE.get('critic_tokens', 16) if tokens is None else tokens)
        queries = int(REFERENCE.get('critic_queries', 8) if queries is None else queries)
        width = int(REFERENCE['critic_width'] if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if min(tokens, queries, width, layers) < 1:
            raise ValueError('tokens/queries/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.queries = queries
        self.width = width
        self.heads = heads
        self.kv_mix = nn.Linear(dim, tokens * width)
        self.query = nn.Parameter(torch.zeros(1, queries, width))
        nn.init.normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList([_CrossBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(width, 1)
        self.score_bound = float(REFERENCE.get('critic_score_bound') or 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.kv_mix.bias, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        kv = self.kv_mix(coordinates).view(bsz, self.tokens, self.width)
        q = self.query.expand(bsz, -1, -1)
        for block in self.blocks:
            q = block(q, kv)
        score = self.head(self.out_norm(q.mean(dim=1))).squeeze(-1)
        return bound_score(score, self.score_bound)


class BottleneckQueryErrorCritic(nn.Module):
    """Query critic with H→W stem before the KV bank (capacity-matched)."""

    def __init__(self, targets, *, tokens=None, queries=None, width=None, layers=None, heads=None,
                 neutrals=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(REFERENCE.get('critic_tokens', 8) if tokens is None else tokens)
        queries = int(REFERENCE.get('critic_queries', 4) if queries is None else queries)
        width = int(REFERENCE['critic_width'] if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if min(tokens, queries, width, layers) < 1:
            raise ValueError('tokens/queries/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.queries = queries
        self.width = width
        self.stem = nn.Sequential(nn.Linear(dim, width), nn.LeakyReLU(0.2))
        self.kv_expand = nn.Linear(width, tokens * width)
        self.query = nn.Parameter(torch.zeros(1, queries, width))
        nn.init.normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList([_CrossBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(width, 1)
        self.score_bound = float(REFERENCE.get('critic_score_bound') or 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.kv_expand.bias, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        kv = self.kv_expand(self.stem(coordinates)).view(bsz, self.tokens, self.width)
        q = self.query.expand(bsz, -1, -1)
        for block in self.blocks:
            q = block(q, kv)
        score = self.head(self.out_norm(q.mean(dim=1))).squeeze(-1)
        return bound_score(score, self.score_bound)


class _CrossBlock(nn.Module):
    """Pre-norm cross-attention (Q←KV) + FFN; handwritten for b_cap."""

    def __init__(self, width: int, n_heads: int, ffn_mult: float = 2.0):
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f'width {width} must split over {n_heads} heads')
        self.n_heads = int(n_heads)
        self.norm_q = nn.RMSNorm(width, eps=1e-3)
        self.norm_kv = nn.RMSNorm(width, eps=1e-3)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.RMSNorm(width, eps=1e-3)
        hidden = max(1, int(round(width * ffn_mult)))
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))

    def forward(self, query, kv):
        bsz, q_len, width = query.shape
        heads = self.n_heads
        dh = width // heads
        q = self.q(self.norm_q(query)).view(bsz, q_len, heads, dh).transpose(1, 2)
        k, v = self.kv(self.norm_kv(kv)).chunk(2, dim=-1)
        k = k.view(bsz, -1, heads, dh).transpose(1, 2)
        v = v.view(bsz, -1, heads, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        a = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype) @ v
        a = a.transpose(1, 2).reshape(bsz, q_len, width)
        h = query + self.out(a)
        return h + self.ffn(self.norm2(h))


CRITIC_CONFIG_KEYS = frozenset({
    'patch', 'width', 'layers', 'heads', 'tokens', 'queries', 'rank',
    'score_bound',
})


def make_critic(targets, kind=None, config=None, *, neutrals=None):
    kind = REFERENCE['critic'] if kind is None else kind
    config = dict(config or {})
    unknown = set(config) - CRITIC_CONFIG_KEYS
    if unknown:
        raise ValueError(f'Unknown critic config keys: {sorted(unknown)}')
    score_bound = config.pop('score_bound', None)
    accepted = {
        'mlp': set(),
        'sn_mlp': {'width', 'layers'},
        'patch': {'patch', 'width', 'layers', 'heads'},
        'mix': {'tokens', 'width', 'layers', 'heads'},
        'gmix': {'tokens', 'width', 'layers', 'heads'},
        'bottleneck': {'tokens', 'width', 'layers', 'heads'},
        'lowrank': {'tokens', 'width', 'layers', 'heads', 'rank'},
        'hybrid': {'tokens', 'width', 'layers', 'heads'},
        'query': {'tokens', 'queries', 'width', 'layers', 'heads'},
        'bquery': {'tokens', 'queries', 'width', 'layers', 'heads'},
    }
    if kind not in accepted:
        raise ValueError(
            f"Unknown particle-bridge critic {kind!r}; "
            "expected mlp|sn_mlp|patch|mix|gmix|bottleneck|lowrank|hybrid|query|bquery"
        )
    unused = set(config) - accepted[kind]
    if unused:
        raise ValueError(f'Critic {kind!r} does not accept: {sorted(unused)}')
    if kind == 'mlp':
        critic = ErrorCritic(targets, neutrals=neutrals)
    else:
        classes = {
            'sn_mlp': ResidualSNErrorCritic,
            'patch': PatchErrorCritic,
            'mix': MixErrorCritic,
            'gmix': GlobalMixErrorCritic,
            'bottleneck': BottleneckMixErrorCritic,
            'lowrank': LowRankMixErrorCritic,
            'hybrid': HybridErrorCritic,
            'query': QueryErrorCritic,
            'bquery': BottleneckQueryErrorCritic,
        }
        critic = classes[kind](targets, neutrals=neutrals, **config)
    if score_bound is not None:
        if not hasattr(critic, 'score_bound'):
            if float(score_bound) != 0:
                raise ValueError(f'Critic {kind!r} does not support score_bound')
        else:
            critic.score_bound = float(score_bound)
    return critic


def build_game(network, targets, *, critic=None, critic_config=None, neutrals=None):
    critic_mod = make_critic(targets, critic, critic_config, neutrals=neutrals).to(targets.device)
    parameters = [p for name, p in network.named_parameters() if name != 'particles']
    if not parameters or network.particles.shape != (128, 4):
        raise ValueError('Expected generator parameters and one 128x4 particle cloud')
    g = torch.optim.Adam([
        dict(params=parameters, lr=REFERENCE['g_lr'], role='generator'),
        dict(params=[network.particles], lr=REFERENCE['particle_lr'], role='particles'),
    ], betas=REFERENCE['betas'])
    d = torch.optim.Adam(critic_mod.parameters(), lr=REFERENCE['d_lr'], betas=REFERENCE['betas'])
    return critic_mod, g, d


def update(network, critic, g, d, targets, predict, *, sampler, step):
    """predict(row, phase) returns normalized G output, retaining its graph.

    Repeated deterministic rows are forwarded once per phase; their independent
    noise samples still contribute separately to the batch-64 mean and gradient.
    """
    device = critic.target_mean.device
    real_targets = critic.normalize(targets.to(device)).detach()
    sigma = noise_std(step, start=getattr(critic, 'noise_start', None),
                      decay_steps=getattr(critic, 'noise_decay_steps', None),
                      hold=float(getattr(critic, 'edit_rms', REFERENCE['edit_rms_target']))
                      * float(REFERENCE['noise_hold_ratio']))
    critic.requires_grad_(True)
    di, dn = sampler.batch(targets.shape[1], device, sigma)
    dn = dn.to(real_targets)
    with torch.no_grad():
        unique = {i: predict(i, 'd') for i in sorted(set(di.tolist()))}
        predictions = torch.cat([unique[i] for i in di.tolist()])
        fake = dn + predictions - real_targets[di.to(device)]
    d.zero_grad(set_to_none=True)
    da = rp_d_loss(critic(dn), critic(fake))
    capper = make_grad_regularizer(arm='b_cap', coeff=1., kappa=1., norm='l2',
                                  lazy_k=4, target_anneal='none')
    cap, _ = capper.penalty(critic, dn, fake, step=step, collect_stats=False)
    d_loss = da + cap
    if not torch.isfinite(d_loss):
        raise FloatingPointError('Non-finite particle-bridge D loss')
    d_loss.backward()
    if not math.isfinite(param_grad_norm(critic.parameters())):
        raise FloatingPointError('Non-finite particle-bridge D gradient')
    d.step(); d.zero_grad(set_to_none=True); critic.requires_grad_(False)

    gi, gn = sampler.batch(targets.shape[1], device, sigma)
    gn = gn.to(real_targets)
    with torch.no_grad():
        real_scores = critic(gn)
    g.zero_grad(set_to_none=True)
    ga = 0.
    for i in sorted(set(gi.tolist())):
        mask = (gi == i).to(device)
        # Keep native adapter scale active through checkpointed backward.
        context = network.scaled(1.) if hasattr(network, 'scaled') else _identity_context()
        with context:
            prediction = predict(i, 'g')
            fake = gn[mask] + prediction - real_targets[i:i+1]
            loss = rp_g_loss(real_scores[mask], critic(fake)) * int(mask.sum()) / len(gi)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite particle-bridge G loss')
            loss.backward()
            ga += float(loss.detach())
    particle_gan_norm = param_grad_norm([network.particles])
    pv = particle_vic(network.particles[sampler.vic_rows(len(network.particles), device)])
    if not torch.isfinite(pv):
        raise FloatingPointError('Non-finite particle VIC')
    pv.backward()
    norm = param_grad_norm(network.parameters())
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite particle-bridge G gradient')
    particle_norm = param_grad_norm([network.particles])
    g.step()
    return dict(loss=ga + float(pv.detach()), g_adv=ga, particle_vic=float(pv.detach()),
        d_loss=float(d_loss.detach()), d_adv=float(da.detach()), d_pen=float(cap.detach()),
        grad_norm=norm, particle_grad_norm=particle_norm, particle_gan_grad_norm=particle_gan_norm, noise_std=sigma,
        penalty_center=1., d_rows=di.tolist(), g_rows=gi.tolist())


@contextmanager
def _identity_context():
    yield


def initialize_ema(network):
    return {k: value.detach().clone() for k, value in network.state_dict().items()}


@torch.no_grad()
def update_ema(ema, network):
    parameters = set(dict(network.named_parameters()))
    for name, value in network.state_dict().items():
        if name in parameters:
            ema[name].lerp_(value, 1. - REFERENCE['ema'])
        else:
            ema[name].copy_(value)
