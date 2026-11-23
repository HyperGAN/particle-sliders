"""CPU gate for the interference-resolution claim of Yadav,
Tam, Choshen, Raffel & Bansal, "TIES-Merging: Resolving
Interference When Merging Models" (arXiv:2306.01708,
NeurIPS 2023), arbitrated in the harvested-delta class of
PR #282 (arXiv:2611.25000).

Paper claim: naive merging loses information through two
interference sources -- (a) redundant small parameter
changes and (b) sign disagreement across models on the same
parameter. TIES = TRIM (reset all but the top-k%
largest-magnitude changes per model), ELECT SIGN (per
parameter, take the sign with the larger total magnitude
mass), MERGE (average only the parameters aligned with the
elected sign, disjointly). It outperforms prior merging
methods across settings, and the paper highlights sign
interference as the dominant term.

Chain: PR #282 proved merging interference is predictable
by a first-order derivative ONLY when anchored at the
RECEIVER (adapted) model -- receiver-anchored cos
0.92/0.97/0.94, base-anchored near-orthogonal and worse
than predicting no change -- and its use-verdict was that
slider composition checks must anchor at the adapted
checkpoint. This test arbitrates the canonical
interference-REDUCTION rule (TIES) in exactly that class:
deltas harvested against a shared base from overlapping
state distributions, composition applied and read AT THE
RECEIVER.

Toy operationalization (the #282 MLP family, widened):
    z(x) = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3,
    input dim 16, hidden 96, output dim 4.
A shared base is first PRETRAINED on a general linear
teacher V0 read on a broad Gaussian (sigma = 1.0), so the
base is a competent feature extractor and fine-tuning
deltas are adaptations, not whole functions (the paper's
regime: pretrained backbone + per-task fine-tunes; a
random-base version was tried first and rejected by the
loop -- every arm collapses there, see the PR body).
The receiver A is a fine-tune of the base on teacher
V0 + dv_A read on blob A. Each incoming behavior is an
independent fine-tune of the SAME base on teacher
V0 + dv_t read on its own blob (blob means drawn at the
#282 scale, sigma = 0.5, so state distributions overlap
through the shared trunk); tau_t = theta_t - theta_0 is a
genuine harvested task vector. Per seed, 3 independent
groups of 3 incoming deltas are composed onto the
receiver, so each arm's readout averages 9 incoming
behaviors per seed.

Composition arms (all produce one composed delta from the
group's taus, applied as theta_A + delta):
  plain        sum of taus (the naive composition)
  ties         TRIM to top-20% per tensor, ELECT SIGN by
               total magnitude mass, disjoint MEAN of the
               sign-aligned values (the paper's rule)
  trim_only    TRIM, then plain sum (no election)
  sign_only    ELECT SIGN + disjoint mean, no trim
  rand_trim    random 20% of coordinates per tau (matched
               sparsity), then plain sum -- the control
               that separates "sparsity/shrinkage helps"
               from "TIES' chosen coordinates help"
  shuffled     TIES with the elected-sign map permuted per
               tensor -- the control that separates
               "election with the CORRECT signs" from
               "any disjoint-mean mosaic"
  avg          plain mean of taus (extra reference: the
               canonical merging baseline of the paper)

Readouts, all anchored at the receiver theta_A:
  lossret      per incoming behavior, the loss gain the
               composed delta realizes at the receiver on
               that behavior's probes, divided by the gain
               its solo delta realizes at the receiver;
               averaged over behaviors and groups
  proj         drift-projection retention: <D_comp, D_solo>
               / ||D_solo||^2 for the composed vs solo
               output drift at the receiver (#282's drift
               language)
  A_ret        receiver-behavior retention: the composed
               model's gain on A's probes over the base,
               divided by the receiver's own solo gain --
               the cross-behavior interference readout

Measured on pop-os (particlegan 0.10.1), 3 seeds, means
over the 3 groups (per-seed detail in the PR body):
  sign-conflict rate on shared coordinates: 0.71-0.76 --
  the interference regime is real, and plain summation is
  catastrophic in it: lossret -0.16 / -2.44 / -0.87,
  A_ret 0.01 / 0.09 / -0.64.
  TIES improves on plain summation on every seed
  (lossret +0.63 / +2.09 / +0.91 over plain; A_ret
  +0.74 / +0.64 / +1.18) -- but it does NOT beat the
  controls. Plain averaging beats TIES' lossret on every
  seed (0.59 / 0.19 / 0.45 vs 0.47 / -0.35 / 0.04), and
  the random-trim control matches or beats it too
  (0.47 / 0.10 / 0.45). Neither TIES step carries a
  separable gain: trim_only and sign_only sit within
  ~0.1 of full TIES in seed-mean lossret. Correct sign
  election DOES carry the composed drift's direction
  (proj 0.33-0.43 vs shuffled 0.07-0.11) -- it just does
  not convert into retained loss gain beyond what plain
  shrinkage (averaging, or randomly trimming then
  summing) already buys. At keep-fraction 50% the same
  ordering holds (TIES lossret still <= avg on all
  seeds; see PR body).

The verdict stated plainly: in this harvested-delta
receiver class, the operative interference lever is the
SCALE of the composed delta, not per-coordinate sign
arbitration. TIES' headline mechanism does not reproduce
here; assertions encode exactly that -- TIES over plain
sum, TIES not over avg / random-trim, ablations
inseparable from full TIES, and correct-sign election
visible only in the drift-projection readout.

What is NOT asserted: anything about large pretrained
checkpoints, classification accuracy (the paper's main
metric; this toy's readouts are regression gains at a
receiver), many-model merges (K > 3 incoming), or TIES
with a tuned global merge coefficient. Pure torch -- no
GAN machinery, because the claim is about how deltas
compose, not training dynamics. Skips cleanly when the
installed particlegan does not provide the EMPF recipe
(the house guard for this paper-test family).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
    # (any exception type: an ImportError from a failed inner
    # import, or a heavier raise -- both must skip, not error)
    pytest.skip("particlegan unavailable", allow_module_level=True)
try:
    from particlegan import get_recipe  # noqa: E402
except Exception:  # a particlegan without the recipe API at all
    get_recipe = None


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    if get_recipe is None:
        pytest.skip("installed particlegan has no recipe API")
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


DIN, HID, COUT = 16, 96, 4    # toy MLP shape
SIG = 0.5                     # per-blob input width
MEAN_SCALE = 2.5              # blob-mean draw scale
DV_SCALE = 1.5                # per-task teacher deviation
FT_STEPS = 40                 # per-task fine-tune steps
FT_LR = 0.005                 # fine-tune Adam step
PRE_STEPS = 600               # base pretraining steps
KEEP = 0.2                    # TIES trim keep-fraction
N_IN = 3                      # incoming deltas per group
N_GROUPS = 3                  # independent groups per seed
ARMS = ("plain", "ties", "trim_only", "sign_only",
        "rand_trim", "shuffled", "avg")


def _base(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return {
        "W1": torch.randn(HID, DIN, generator=g) / np.sqrt(DIN),
        "b1": torch.zeros(HID),
        "W2": torch.randn(HID, HID, generator=g) / np.sqrt(HID),
        "b2": torch.zeros(HID),
        "W3": torch.randn(COUT, HID, generator=g) / np.sqrt(HID),
        "b3": torch.zeros(COUT),
    }


def _fwd(p, x):
    h1 = torch.tanh(x @ p["W1"].T + p["b1"])
    h2 = torch.tanh(h1 @ p["W2"].T + p["b2"])
    return h2 @ p["W3"].T + p["b3"]


def _data(v, m, n, seed, sig=SIG):
    g = torch.Generator().manual_seed(seed)
    x = m + sig * torch.randn(n, DIN, generator=g)
    return x, x @ v.T


def _train(base, v, m, steps, seed, lr, sig=SIG, n=256):
    params = {k: t.clone().requires_grad_(True)
              for k, t in base.items()}
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    x, y = _data(v, m, n, seed, sig)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((_fwd(params, x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return {k: t.detach().clone() for k, t in params.items()}


def _loss(p, v, m, seed):
    x, y = _data(v, m, 128, seed)
    return float(((_fwd(p, x) - y) ** 2).mean())


def _trim(tau, keep_frac, rng=None, random=False):
    out = {}
    for k, t in tau.items():
        flat = t.flatten()
        keep = max(1, int(round(keep_frac * flat.numel())))
        if random:
            idx = torch.randperm(flat.numel(), generator=rng)[:keep]
        else:
            idx = torch.topk(flat.abs(), keep).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1
        out[k] = (flat * mask).reshape(t.shape)
    return out


def _elect(taus, rng=None, shuffle=False):
    """Disjoint mean under the elected sign (optionally shuffled)."""
    out = {}
    for k in taus[0]:
        stack = torch.stack([t[k] for t in taus])
        total = stack.sum(0)
        if shuffle:
            flat = total.flatten()
            perm = torch.randperm(flat.numel(), generator=rng)
            sign = torch.sign(flat[perm]).reshape(total.shape)
        else:
            sign = torch.sign(total)
        aligned = torch.where(stack * sign > 0, stack,
                              torch.zeros_like(stack))
        cnt = (aligned != 0).sum(0).clamp(min=1)
        out[k] = aligned.sum(0) / cnt
    return out


def _arms(taus, seed):
    rng = torch.Generator().manual_seed(999 + seed)
    trimmed = [_trim(t, KEEP) for t in taus]
    rtrim = [_trim(t, KEEP, rng=rng, random=True) for t in taus]
    return {
        "plain": {k: sum(t[k] for t in taus) for k in taus[0]},
        "ties": _elect(trimmed),
        "trim_only": {k: sum(t[k] for t in trimmed) for k in taus[0]},
        "sign_only": _elect(taus),
        "rand_trim": {k: sum(t[k] for t in rtrim) for k in taus[0]},
        "shuffled": _elect(trimmed, rng=rng, shuffle=True),
        "avg": {k: sum(t[k] for t in taus) / len(taus)
                for k in taus[0]},
    }


def run_seed(seed):
    """All arm readouts for one seed, averaged over groups.

    Returns per-arm mean lossret / proj / A_ret, the
    sign-conflict rate of the first group's taus, and the
    number of incoming behaviors whose solo gain at the
    receiver cleared the validity floor in every group.
    """
    g = torch.Generator().manual_seed(3000 + seed)
    v0 = torch.randn(COUT, DIN, generator=g) * (3.0 / np.sqrt(DIN))
    base = _train(_base(seed), v0, torch.zeros(DIN),
                  PRE_STEPS, 2000 + seed, lr=0.03, sig=1.0, n=512)

    def make_task(s):
        gg = torch.Generator().manual_seed(s)
        dv = torch.randn(COUT, DIN, generator=gg) * (DV_SCALE / np.sqrt(DIN))
        m = torch.randn(DIN, generator=gg) * (MEAN_SCALE / np.sqrt(DIN))
        return v0 + dv, m

    va, ma = make_task(5000 + seed * 100)
    th_a = _train(base, va, ma, FT_STEPS, 7000 + seed, lr=FT_LR)
    a_gain = _loss(base, va, ma, 9000 + seed) - _loss(th_a, va, ma, 9000 + seed)

    agg = {name: [] for name in ARMS}
    conflict = None
    n_valid = []
    for grp in range(N_GROUPS):
        tasks = [make_task(6000 + seed * 1000 + grp * 100 + i * 17)
                 for i in range(N_IN)]
        thetas = [_train(base, v, m, FT_STEPS,
                         7100 + seed * 10 + grp * 3 + i, lr=FT_LR)
                  for i, (v, m) in enumerate(tasks)]
        taus = [{k: th[k] - base[k] for k in base} for th in thetas]
        if grp == 0:
            stack = np.stack([np.concatenate(
                [taus[i][k].numpy().flatten() for k in base])
                for i in range(N_IN)])
            conflict = float(np.mean((stack > 0).any(0)
                                     & (stack < 0).any(0)))
        valid = 0
        for name, delta in _arms(taus, seed * 10 + grp).items():
            comp = {k: th_a[k] + delta[k] for k in base}
            lrets, prets = [], []
            for i, (v, m) in enumerate(tasks):
                solo = {k: th_a[k] + taus[i][k] for k in base}
                sg = (_loss(th_a, v, m, 9100 + seed)
                      - _loss(solo, v, m, 9100 + seed))
                cg = (_loss(th_a, v, m, 9100 + seed)
                      - _loss(comp, v, m, 9100 + seed))
                if sg > 0.02:
                    lrets.append(cg / sg)
                    if name == "plain":
                        valid += 1
                xp, _ = _data(v, m, 128, 9100 + seed)
                ds = (_fwd(solo, xp) - _fwd(th_a, xp)).flatten()
                dc = (_fwd(comp, xp) - _fwd(th_a, xp)).flatten()
                prets.append(float(dc @ ds / (ds @ ds)))
            a_ret = ((_loss(base, va, ma, 9000 + seed)
                      - _loss(comp, va, ma, 9000 + seed)) / a_gain)
            agg[name].append((float(np.mean(lrets)),
                              float(np.mean(prets)), a_ret))
        n_valid.append(valid)
    out = {name: tuple(float(np.mean([r[j] for r in rows]))
                       for j in range(3))
           for name, rows in agg.items()}
    return dict(arms=out, conflict=conflict, n_valid=n_valid,
                a_gain=a_gain)


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_interference_regime_at_receiver():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = _seed(seed)
        # The harvested deltas genuinely fight: a large share
        # of shared coordinates carry both signs across the
        # group (demonstrated 0.71-0.76), and every incoming
        # behavior's solo delta realizes a real gain at the
        # receiver (validity floor cleared by all 9).
        assert d["conflict"] >= 0.60, (
            f"seed {seed}: sign-conflict rate {d['conflict']:.3f}")
        assert d["n_valid"] == [N_IN] * N_GROUPS, (
            f"seed {seed}: valid solo gains {d['n_valid']}")
        # Plain summation is catastrophic in this regime:
        # incoming loss retention at/below zero and the
        # receiver's own behavior essentially destroyed
        # (demonstrated lossret -0.16/-2.44/-0.87,
        # A_ret 0.01/0.09/-0.64).
        plain = d["arms"]["plain"]
        assert plain[0] <= 0.20, (
            f"seed {seed}: plain lossret {plain[0]:.3f}")
        assert plain[2] <= 0.35, (
            f"seed {seed}: plain A_ret {plain[2]:.3f}")


def test_ties_beats_plain_sum_but_not_shrinkage_controls():
    _empf_recipe()
    ties_mean, avg_mean, rand_mean = [], [], []
    for seed in (0, 1, 2):
        d = _seed(seed)
        ties = d["arms"]["ties"]
        plain = d["arms"]["plain"]
        avg = d["arms"]["avg"]
        rand = d["arms"]["rand_trim"]
        # TIES does resolve part of the plain-sum damage, on
        # both readouts and every seed (demonstrated lossret
        # gaps +0.63/+2.09/+0.91, A_ret gaps +0.74/+0.64/+1.18).
        assert ties[0] >= plain[0] + 0.25, (
            f"seed {seed}: ties vs plain lossret "
            f"{ties[0]:.3f} vs {plain[0]:.3f}")
        assert ties[2] >= plain[2] + 0.30, (
            f"seed {seed}: ties vs plain A_ret "
            f"{ties[2]:.3f} vs {plain[2]:.3f}")
        # But it never beats plain averaging by any margin,
        # and never beats the matched-sparsity random-trim
        # control by any margin (demonstrated per-seed
        # avg-ties lossret gaps +0.12/+0.54/+0.41 and
        # rand-ties gaps +0.01/+0.45/+0.41).
        assert avg[0] >= ties[0] - 0.05, (
            f"seed {seed}: avg vs ties lossret "
            f"{avg[0]:.3f} vs {ties[0]:.3f}")
        assert rand[0] >= ties[0] - 0.05, (
            f"seed {seed}: rand_trim vs ties lossret "
            f"{rand[0]:.3f} vs {ties[0]:.3f}")
        ties_mean.append(ties[0])
        avg_mean.append(avg[0])
        rand_mean.append(rand[0])
    # Seed-mean: the shrinkage controls beat TIES outright
    # (demonstrated avg 0.41 vs ties 0.05; rand 0.34 vs 0.05).
    assert float(np.mean(avg_mean)) >= float(np.mean(ties_mean)) + 0.20
    assert float(np.mean(rand_mean)) >= float(np.mean(ties_mean)) + 0.10


def test_no_ties_step_carries_the_gain():
    _empf_recipe()
    ties_proj, shuf_proj = [], []
    ties_lr, trim_lr, sign_lr = [], [], []
    for seed in (0, 1, 2):
        d = _seed(seed)
        ties = d["arms"]["ties"]
        shuf = d["arms"]["shuffled"]
        # Correct sign election DOES carry the composed
        # drift's direction at the receiver: TIES' drift
        # projection clearly exceeds the shuffled-sign
        # control's (demonstrated proj gaps +0.25/+0.22/+0.32).
        assert ties[1] >= shuf[1] + 0.15, (
            f"seed {seed}: ties vs shuffled proj "
            f"{ties[1]:.3f} vs {shuf[1]:.3f}")
        ties_proj.append(ties[1])
        shuf_proj.append(shuf[1])
        ties_lr.append(ties[0])
        trim_lr.append(d["arms"]["trim_only"][0])
        sign_lr.append(d["arms"]["sign_only"][0])
    # But neither step converts into a separable loss-gain
    # advantage: trim-only and sign-only seed-means sit on
    # top of full TIES (demonstrated trim -0.01, sign 0.13,
    # ties 0.05 -- all within ~0.1 of each other, and all
    # below the shrinkage controls of the previous test).
    assert abs(float(np.mean(trim_lr)) - float(np.mean(ties_lr))) <= 0.25
    assert abs(float(np.mean(sign_lr)) - float(np.mean(ties_lr))) <= 0.25
