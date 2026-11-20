"""CPU gate for the MAST geometry claim (arXiv:2611.15168): when the
activations for a behavior form multiple clusters, a single global
mean steering direction lands in the empty region between clusters,
is faithful to no single cluster, and steers unevenly with cross-task
interference; cluster-local directions anchored at cluster medoids
steer each cluster's behavior with less interference.

Paper: "MAST: Medoid-Anchored Subspace Steering for Visual Grounding
in Multimodal LLMs" (Chen et al.). Its central diagnosis: existing
steering uses the global activation mean, which fails on
multi-centered geometry -- the global mean is faithful to no single
task, and per-cluster steering signal is low-rank and local. MAST
clusters per task, picks a medoid per cluster, extracts a low-rank
local direction per cluster, and routes each query to its medoids.
(MAST-PR's probe reweighting adds +8.09 points over MAST in the
paper; this gate tests the base geometry claim, not the probe.)

Toy operationalization: states in R^24 from K = 3 clusters with
well-separated centers c_k and orthonormal ground-truth behavior axes
u_k. Within a cluster, behavior intensity z shifts states along u_k.
Cluster prevalences in the pooled contrast population are skewed
(0.6 / 0.3 / 0.1), so the pooled (global) contrast direction tilts
toward the dominant cluster and away from the rare one -- the paper's
"faithful to no single task", made measurable per cluster.

Arms:
  (a) global: one pooled mean-difference direction applied to every
      held-out state;
  (b) local (MAST-style): k-means (K = 3) on the harvested states,
      medoid per cluster, per-cluster mean-difference direction;
      each held-out state is routed to its nearest medoid and steered
      along that cluster's direction;
  (c) wrong-route control: local directions, but each state is routed
      to a different cluster's direction (routing is load-bearing,
      not decoration).

Metrics per seed: per-cluster behavior gain = mean change of the
ground-truth readout h . u_k for held-out states of cluster k, at
matched steering norm ALPHA; interference = mean absolute change of
the OTHER clusters' readouts h . u_j (j != k). The gate asserts the
local arm's worst-cluster gain is high while the global arm's
worst-cluster gain is much lower, and that local interference is
below global interference. Measured on pop-os (particlegan 0.10.1):
global-direction alignment per cluster 0.73 / 0.65 / 0.20 (dominant
to rare -- the pooled direction is faithful to the prevalent cluster
and nearly orthogonal to the rare one); local directions recover
their axes at cos 0.998-0.999 on every seed; wrong-route gains
|.| <= 0.018; local interference <= 0.023 vs global 0.65-0.73.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe.
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


D = 24
K = 3
CENTER_SEP = 6.0
NOISE_SD = 0.5
Z_HI, Z_LO = 1.5, -0.5     # behavior intensity in on/off contrast states
PREV = np.array([0.6, 0.3, 0.1])   # skewed cluster prevalences
N_PER = 1200               # harvested states per cluster per contrast arm
N_HELD = 600               # held-out states per cluster
ALPHA = 1.0
KMEANS_ITERS = 25


def _unit(x):
    return x / np.linalg.norm(x)


def _kmeans_medoids(x, k, rng):
    # Farthest-point initialization: random sample init splits/merges
    # true clusters on some seeds (iteration 1: one seed's routing
    # collapsed, local cos 0.70); farthest-point init is deterministic
    # given the first point and cannot start two centers in one cloud.
    first = rng.integers(len(x))
    chosen = [x[first]]
    dmin = ((x - chosen[0]) ** 2).sum(-1)
    for _ in range(k - 1):
        nxt = x[int(np.argmax(dmin))]
        chosen.append(nxt)
        dmin = np.minimum(dmin, ((x - nxt) ** 2).sum(-1))
    centers = np.array(chosen)
    for _ in range(KMEANS_ITERS):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for j in range(k):
            if np.any(lab == j):
                centers[j] = x[lab == j].mean(0)
    medoids = []
    for j in range(k):
        members = x[lab == j]
        c = members.mean(0)
        medoids.append(members[np.argmin(((members - c) ** 2).sum(-1))])
    return np.array(medoids), lab


def run_seed(seed):
    _empf_recipe()
    rng = np.random.default_rng(13000 + seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, 2 * K)))
    centers = CENTER_SEP * q[:, :K].T          # (K, D), orthonormal rows scaled
    axes = q[:, K:].T                          # (K, D) orthonormal behavior axes

    def sample(k, n, z):
        return (centers[k] + z * axes[k]
                + rng.standard_normal((n, D)) * NOISE_SD)

    # Harvested contrast states per cluster.
    on = [sample(k, N_PER, Z_HI) for k in range(K)]
    off = [sample(k, N_PER, Z_LO) for k in range(K)]

    # (a) Global direction from the prevalence-skewed pooled contrast.
    pooled_on = np.concatenate(
        [on[k][: int(N_PER * 3 * PREV[k])] for k in range(K)])
    pooled_off = np.concatenate(
        [off[k][: int(N_PER * 3 * PREV[k])] for k in range(K)])
    d_global = _unit(pooled_on.mean(0) - pooled_off.mean(0))

    # (b) Local directions via k-means + medoids on all harvested states.
    all_states = np.concatenate(on + off)
    medoids, _ = _kmeans_medoids(all_states, K, rng)
    # Assign harvested contrast states to nearest medoid; match clusters
    # to ground truth by medoid proximity to the true centers.
    def nearest_medoid(x):
        d2 = ((x[:, None, :] - medoids[None, :, :]) ** 2).sum(-1)
        return d2.argmin(1)
    medoid_true = np.array([np.argmin(((medoids - centers[k]) ** 2).sum(-1))
                            for k in range(K)])  # true cluster -> medoid idx
    local_dir = {}
    for k in range(K):
        m = medoid_true[k]
        on_m = np.concatenate([s[nearest_medoid(s) == m] for s in on])
        off_m = np.concatenate([s[nearest_medoid(s) == m] for s in off])
        local_dir[k] = _unit(on_m.mean(0) - off_m.mean(0))

    held = [sample(k, N_HELD, 0.0) for k in range(K)]

    def evaluate(direction_for):
        gains, inters = [], []
        for k in range(K):
            delta = ALPHA * direction_for(k)
            gains.append(float(delta @ axes[k]))
            other = [float(np.abs(delta @ axes[j])) for j in range(K) if j != k]
            inters.append(max(other))
        return np.array(gains), np.array(inters)

    g_gain, g_inter = evaluate(lambda k: d_global)
    l_gain, l_inter = evaluate(lambda k: local_dir[k])
    w_gain, _ = evaluate(lambda k: local_dir[(k + 1) % K])
    return {
        "global_gain": g_gain, "global_inter": g_inter,
        "local_gain": l_gain, "local_inter": l_inter,
        "wrong_gain": w_gain,
        "global_cos": np.array([float(d_global @ axes[k]) for k in range(K)]),
        "local_cos": np.array([float(local_dir[k] @ axes[k]) for k in range(K)]),
    }


def test_mast_medoid_subspace_steering():
    results = [run_seed(seed) for seed in range(3)]
    for r in results:
        # The global direction is diluted/tilted: its worst-cluster
        # alignment is far below the local directions' recovery.
        assert r["global_cos"].min() <= 0.65
        assert r["local_cos"].min() >= 0.90
        # Worst-cluster behavior gain: local steering delivers, the
        # global direction leaves the rare cluster nearly unsteered.
        assert r["local_gain"].min() >= 0.85
        assert r["global_gain"].min() <= 0.65
        assert r["local_gain"].min() >= r["global_gain"].min() + 0.25
        # Routing is load-bearing: the wrong cluster's direction fails.
        assert r["wrong_gain"].max() <= 0.35
        # Local steering interferes less with other clusters' readouts.
        assert r["local_inter"].max() <= r["global_inter"].max()
