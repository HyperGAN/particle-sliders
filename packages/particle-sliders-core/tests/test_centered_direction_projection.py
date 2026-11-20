"""CPU gate for the Centered Ablation claim (arXiv:2611.20148):
at the last residual stream, a behavior direction that overlaps the
states' shared common mode gates and ablates badly -- it fires on
benign inputs and wrecks a benign readout -- while projecting the
estimated common mode out of the direction (closed form, no training)
restores gate selectivity and benign preservation at a retained
target effect.

Paper: "Centered Ablation: Why Last-Layer Directional Ablation
Under-Fires or Collapses, and a Closed-Form Fix" (Dang & Ngo). Its
mechanism: directional ablation / steering along a behavior direction
d is gated by cos(h, d) measured from the ORIGIN, but last-layer
states share a large common mode far from the origin, so a direction
overlapping that mode fires on benign inputs and selects poorly on
its targets. The fix projects the common mode out of the direction:
d' = d - (d . chat) chat. Measured across 8 models: +29 and +46
points of effect where the standard recipe under-fires, collapse cut
from 50% to 1% on Gemma-2-9B, and 12-40 more benign points preserved
on high-overlap models. This gate asserts the MECHANISM (gate
selectivity + benign preservation at retained effect), not the
paper's exact point gains.

Toy operationalization: synthetic last-layer states in R^32 with
KNOWN structure,

    h = c + b * u + noise,

where c is one large shared common mode (|c| = 6, same for every
state), u is the ground-truth behavior axis (orthogonal to c), b in
[0.8, 1.6] on behavior-present (target) states and b = 0 on benign
states. Two ground-truth readouts: behavior s_beh(h) = h . u and
benign s_ben(h) = h . chat -- the benign readout genuinely loads on
the common mode, so any intervention that removes the common mode
damages it, exactly the paper's benign-preservation axis.

How the direction is built -- and how the overlap enters honestly:
a PLAIN mean difference mean(h|target) - mean(h|benign) cancels c
exactly in this construction, so it cannot exhibit the paper's
failure. The toy therefore uses the paper's actual setting: the
direction is the UNCENTERED mean of behavior-present states,
d = mean(h | target), the single-mean / carried-direction estimator
practitioners use when a direction is harvested in one context and
applied in another (contrast sets from a different context or model
do not share this population's benign mean, so c does not cancel).
That estimator carries cos(d, chat) ~= 0.98 by construction of the
world, and the test asserts that overlap is present -- without it
the toy tests nothing.

Arms (all unit-norm, so the naive arm is already the matched-norm
control):
  (a) naive:    d as above;
  (b) centered: d' = d - (d . chat_est) chat_est, renormalized, with
      chat_est the unit mean state over ALL states (targets+benign)
      -- the paper's closed-form fix, estimated, not oracle;
  (c) random:   a fixed random unit direction (control).

Metrics per seed: gate AUC of cos(h, .) separating target from
benign states; mean benign gate value (false-fire level); ablation
h' = h - (h . d) d applied to every state -- target effect = drop
in s_beh on target states, benign damage = drop in s_ben on benign
states; and the matched-damage target effect (naive direction
rescaled so its benign damage equals the centered arm's -- its
target effect collapses, which is the under-firing half of the
paper's claim in efficiency form).

Measured on pop-os (particlegan 0.10.1, seeds 0-2): overlap
cos(d, chat) 0.980-0.981; gate AUC naive 0.695-0.716 vs centered
0.941-0.946 vs random 0.522-0.602; benign gate mean 0.89 naive vs
~-0.09 centered; ablation target effect 0.59 centered (vs
1.19-1.21 naive at ~97x the benign damage); benign damage 5.77
naive vs 0.058-0.060 centered; matched-damage naive target effect
~0.01. Skips cleanly when the installed particlegan does not
provide the EMPF recipe.
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


D = 32            # state dimension
N_HALF = 2000     # target states and benign states, each, per seed
C_NORM = 6.0      # common-mode magnitude (>> behavior component)
B_LO, B_HI = 0.8, 1.6   # behavior loading on target states
NOISE_SD = 0.5    # per-state noise


def _unit(x):
    return x / np.linalg.norm(x)


def _auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_seed(seed):
    rng = np.random.default_rng(9000 + seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, 2)))
    chat, u = q[:, 0], q[:, 1]
    c = C_NORM * chat

    b = rng.uniform(B_LO, B_HI, N_HALF)
    h_t = c[None, :] + b[:, None] * u[None, :] \
        + rng.standard_normal((N_HALF, D)) * NOISE_SD
    h_b = c[None, :] + rng.standard_normal((N_HALF, D)) * NOISE_SD
    h = np.vstack([h_t, h_b])
    y = np.array([1] * N_HALF + [0] * N_HALF)

    chat_est = _unit(h.mean(0))
    d_naive = _unit(h_t.mean(0))
    d_cent = _unit(d_naive - (d_naive @ chat_est) * chat_est)
    d_rand = _unit(rng.standard_normal(D))

    def gate(d):
        return (h @ d) / np.linalg.norm(h, axis=1)

    def ablate(states, d):
        return states - (states @ d)[:, None] * d[None, :]

    def effects(d):
        tgt = float((h_t @ u).mean() - (ablate(h_t, d) @ u).mean())
        ben = float((h_b @ chat).mean() - (ablate(h_b, d) @ chat).mean())
        return tgt, ben

    tgt_n, ben_n = effects(d_naive)
    tgt_c, ben_c = effects(d_cent)
    tgt_r, ben_r = effects(d_rand)
    # Matched-damage: rescale the naive ablation so its benign damage
    # equals the centered arm's; read off what target effect survives.
    scale = ben_c / ben_n if ben_n > 0 else 0.0
    h_t_md = h_t - scale * (h_t @ d_naive)[:, None] * d_naive[None, :]
    tgt_matched = float((h_t @ u).mean() - (h_t_md @ u).mean())

    return dict(
        overlap=float(d_naive @ chat), overlap_cent=float(d_cent @ chat),
        cos_cent_u=float(d_cent @ u),
        auc_naive=_auc(gate(d_naive), y), auc_cent=_auc(gate(d_cent), y),
        auc_rand=_auc(gate(d_rand), y),
        benign_gate_naive=float(gate(d_naive)[y == 0].mean()),
        benign_gate_cent=float(gate(d_cent)[y == 0].mean()),
        tgt_naive=tgt_n, ben_naive=ben_n,
        tgt_cent=tgt_c, ben_cent=ben_c,
        tgt_rand=tgt_r, ben_rand=ben_r,
        tgt_matched=tgt_matched,
    )


def test_centered_projection_restores_gate_and_preserves_benign():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Trap guard: the naive direction must actually overlap the
        # common mode -- otherwise the toy tests nothing.
        assert r["overlap"] > 0.9, (
            f"seed {seed}: cos(d, chat) {r['overlap']:.3f} -- common-mode "
            "overlap absent, the paper's failure cannot occur")
        # Centering removes the overlap and recovers the behavior axis.
        assert abs(r["overlap_cent"]) < 0.2, (
            f"seed {seed}: centered cos(d', chat) {r['overlap_cent']:.3f}")
        assert r["cos_cent_u"] > 0.9, (
            f"seed {seed}: centered recovery cos(d', u) {r['cos_cent_u']:.3f}")
        # Gate selectivity: centered separates target from benign
        # strictly and clearly better than naive and than random.
        assert r["auc_cent"] > 0.9, (
            f"seed {seed}: centered AUC {r['auc_cent']:.3f}")
        assert r["auc_cent"] >= r["auc_naive"] + 0.15, (
            f"seed {seed}: AUC centered {r['auc_cent']:.3f} vs "
            f"naive {r['auc_naive']:.3f}")
        assert r["auc_cent"] >= r["auc_rand"] + 0.15, (
            f"seed {seed}: AUC centered {r['auc_cent']:.3f} vs "
            f"random {r['auc_rand']:.3f}")
        # False-fire: the naive gate sits near its ceiling on benign
        # states (they share the common mode); the centered gate does not.
        assert r["benign_gate_naive"] > 0.85, (
            f"seed {seed}: naive benign gate {r['benign_gate_naive']:.3f}")
        assert abs(r["benign_gate_cent"]) < 0.1, (
            f"seed {seed}: centered benign gate {r['benign_gate_cent']:.3f}")
        # Intervention: centered ablation retains a substantial target
        # effect while damaging the benign readout ~100x less.
        assert r["tgt_cent"] > 0.4, (
            f"seed {seed}: centered target effect {r['tgt_cent']:.3f} "
            f"(naive {r['tgt_naive']:.3f}, random {r['tgt_rand']:.3f})")
        assert r["ben_cent"] < 0.15 * r["ben_naive"], (
            f"seed {seed}: benign damage centered {r['ben_cent']:.3f} vs "
            f"naive {r['ben_naive']:.3f}")
        assert abs(r["ben_rand"]) < 0.15 * r["ben_naive"], (
            f"seed {seed}: random benign damage {r['ben_rand']:.3f} vs "
            f"naive {r['ben_naive']:.3f} -- damage is not direction-specific")
        # Under-firing, in efficiency form: at matched benign damage
        # the naive direction's target effect collapses far below
        # the centered arm's.
        assert r["tgt_matched"] < 0.25 * r["tgt_cent"], (
            f"seed {seed}: matched-damage naive target effect "
            f"{r['tgt_matched']:.3f} vs centered {r['tgt_cent']:.3f}")
