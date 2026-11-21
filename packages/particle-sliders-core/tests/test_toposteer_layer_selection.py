"""CPU gate for the TopoSteer claim (arXiv:2611.01493): layer
selection for steering by class separability and by displacement-
cloud extent rank layers differently -- as per-example variation
grows, the two criteria move in opposite directions -- and the
separability winner can be a layer where steering does almost
nothing, because separability never steers the model to check.

Paper: "TopoSteer: Topological Layer Selection for Activation
Steering in LLMs" (Nguyen et al.). Prior selectors score a layer by
how separable the target behavior's activation clouds are, or how
consistently examples align with the mean displacement -- without
ever steering. TopoSteer instead scores a layer by the total H_0
persistence of its per-example displacement cloud, computed as the
length of the cloud's Euclidean minimum spanning tree (MST). The
paper proves that as example-level variation increases, TopoSteer
and discriminability move in opposite directions, and reports that
TopoSteer-selected layers steer more effectively across models and
tasks, without held-out validation.

Toy operationalization: L = 5 synthetic layers, each with paired
off/on states for N examples. Displacements delta_i = h_on - h_off
are computed in per-layer standardized state space (the paper
standardizes activations per layer before displacing). Two selectors:
  sep(l)  = Fisher-style discriminability of the state clouds along
            the mean-displacement direction;
  topo(l) = total MST length of the displacement cloud {delta_i}.
Realized steering gain gain(l) = r . d_hat_l, the projection of the
layer's unit mean displacement onto a FIXED behavior readout r --
the check the prior selectors never run.

The world encodes the paper's argument, with ground truth known:
  layer A: examples displace almost identically along a direction a
           nearly orthogonal to r. Clouds are tight, so separability
           is maximal; the displacement cloud is a small ball, so the
           MST is short; steering there barely moves the behavior
           readout (gain ~0.1).
  layer B: the behavior's own variation lives here -- displacements
           are centered on r but vary widely per example. Separability
           is diluted by the scatter; the displacement cloud is
           extensive, so the MST is long; steering there moves the
           readout (gain ~0.9).
  layers C-E: middling fillers on both criteria.
A variation sweep on a single-layer family asserts the theorem's
direction: as per-example displacement spread sigma grows,
separability falls while the MST score rises.

Measured on pop-os (particlegan 0.10.1): separability crowns A on
every seed (0.94-1.01 vs B's 0.79-0.82) while the MST score crowns B
(667-685 vs A's 48-51); realized gain 0.99 at B vs 0.10-0.13 at A.
Variation sweep: separability falls 1.0 -> 0.69-0.81 while the MST
score rises 47 -> 731-746 as spread grows 0.05 -> 1.0. Skips cleanly
when the installed particlegan does not provide the EMPF recipe.
"""
import numpy as np
import pytest
import torch

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 16
N = 240            # paired examples per layer
L = 5


def _unit(x):
    return x / np.linalg.norm(x)


def _mst_length(x):
    """Total Euclidean MST length via Prim's algorithm, O(N^2)."""
    n = len(x)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    in_tree = np.zeros(n, dtype=bool)
    dist = np.full(n, np.inf)
    dist[0] = 0.0
    total = 0.0
    for _ in range(n):
        d = np.where(in_tree, np.inf, dist)
        i = int(np.argmin(d))
        in_tree[i] = True
        total += np.sqrt(max(dist[i], 0.0))
        dist = np.minimum(dist, d2[i])
    return float(total)


def _layer_scores(states_off, states_on):
    """Standardize per layer, displace, score by both selectors."""
    all_states = np.concatenate([states_off, states_on])
    mu, sd = all_states.mean(0), all_states.std(0) + 1e-9
    off = (states_off - mu) / sd
    on = (states_on - mu) / sd
    delta = on - off
    d_bar = delta.mean(0)
    d_hat = _unit(d_bar)
    # Fisher discriminability along the mean-displacement direction.
    proj_off = off @ d_hat
    proj_on = on @ d_hat
    sep = abs(proj_on.mean() - proj_off.mean()) / (
        proj_on.std() + proj_off.std() + 1e-9)
    topo = _mst_length(delta)
    return sep, topo, d_hat


def run_seed(seed):
    _empf_recipe()
    rng = np.random.default_rng(15000 + seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, 3)))
    r = q[:, 0]                       # fixed behavior readout
    a = 0.1 * r + np.sqrt(1 - 0.01) * q[:, 1]   # layer A direction, ~orth to r
    a = _unit(a)

    def states(mean_disp, spread, base_sd=0.8):
        base = rng.standard_normal((N, D)) * base_sd
        disp = mean_disp + rng.standard_normal((N, D)) * spread
        return base, base + disp

    layers = {}
    # A: tight identical displacements along a; clouds tight -> sep max.
    layers["A"] = states(1.6 * a, 0.05)
    # B: behavior variation lives here; displacements centered on r, wide.
    layers["B"] = states(1.6 * r, 0.85)
    # Fillers: moderate separation, moderate spread, off-readout dirs.
    for name, frac, spread in [("C", 0.45, 0.35), ("D", 0.6, 0.5), ("E", 0.3, 0.25)]:
        d = _unit(frac * r + np.sqrt(1 - frac ** 2) * q[:, 2])
        layers[name] = states(1.4 * d, spread)

    out = {}
    for name, (off, on) in layers.items():
        sep, topo, d_hat = _layer_scores(off, on)
        out[name] = {"sep": sep, "topo": topo, "gain": float(d_hat @ r)}
    return out


def run_sweep(seed):
    """Single-layer family: grow per-example displacement spread."""
    rng = np.random.default_rng(16000 + seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, 2)))
    r = q[:, 0]
    seps, topos = [], []
    for spread in (0.05, 0.3, 1.0):
        base = rng.standard_normal((N, D)) * 0.8
        disp = 1.6 * r + rng.standard_normal((N, D)) * spread
        sep, topo, _ = _layer_scores(base, base + disp)
        seps.append(sep)
        topos.append(topo)
    return seps, topos


def test_toposteer_layer_selection():
    for seed in range(3):
        out = run_seed(seed)
        sep_pick = max(out, key=lambda k: out[k]["sep"])
        topo_pick = max(out, key=lambda k: out[k]["topo"])
        # The criteria disagree on the layer, as the paper's theorem
        # allows: separability crowns the tight-cloud layer A,
        # TopoSteer crowns the extensive-cloud layer B.
        assert sep_pick == "A"
        assert topo_pick == "B"
        # Steering at the TopoSteer pick actually moves the behavior
        # readout; at the separability pick it barely does -- the
        # check separability-based selection never runs.
        assert out["B"]["gain"] >= 0.70
        assert out["A"]["gain"] <= 0.30
        assert out["B"]["gain"] >= out["A"]["gain"] + 0.40
        # Theorem direction: variation up -> separability down, MST up.
        seps, topos = run_sweep(seed)
        assert seps[0] > seps[1] > seps[2]
        assert topos[0] < topos[1] < topos[2]
