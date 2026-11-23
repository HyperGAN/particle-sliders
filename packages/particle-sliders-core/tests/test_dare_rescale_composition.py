"""CPU gate for the rescale claim of Yu, Zhang, Song,
Wang, Chen & Chen, "Language Models are Super Mario:
Absorbing Abilities from Homologous Models as a Free
Lunch" (DARE, arXiv:2311.03099), arbitrated in the
harvested-delta receiver class of PR #282 / PR #307.

Paper claims: (1) a single SFT delta is so redundant that
randomly dropping a fraction p of its parameters and
RESCALING the survivors by 1/(1-p) leaves its abilities
essentially intact -- the paper reports drops up to
p = 0.9, even 0.99; (2) sparsifying deltas this way
mitigates merge interference, so a DARE-merge should beat
naive composition. PR #307's random-trim control was DARE
WITHOUT the rescale; the 1/(1-p) term is the untested
piece, and it is a pure scale claim.

Chain: PR #282 (arXiv:2611.25000) proved merging
interference is predictable only anchored at the RECEIVER
(adapted checkpoint), so every readout here is anchored at
the receiver. PR #307 (TIES, arXiv:2306.01708) found, in
this same class, that plain summation is catastrophic,
TIES repairs part of it, but plain AVERAGING beats TIES on
every seed and random trim at matched sparsity matches it
-- the operative lever was composed-delta SCALE, not sign
arbitration. Field tension: in Altair-Nidhogg's regime
(arXiv:2611.27550, full fine-tune task vectors) TIES beat
averaging and DARE did NOT repair the full-vector sum.
This test asks what DARE's rescale does in OUR class, in
three places: a single delta, an averaged merge, and the
full sum (Altair's null configuration).

Toy: identical construction to PR #307 (calibration
below reproduces #307's reference numbers exactly).
Two-hidden-layer tanh MLP, input 16, hidden 96, output 4.
A shared base is PRETRAINED on a general linear teacher
V0 over a broad Gaussian (sigma = 1.0, 600 steps); the
receiver A is a fine-tune of the base on V0 + dv_A on
blob A; each incoming behavior is an independent
fine-tune of the same base on V0 + dv_t on its own blob
(blob sigma = 0.5, so distributions overlap through the
shared trunk); tau_t = theta_t - theta_0. Per seed:
3 groups x 3 incoming deltas.

Arms:
  single-delta (one tau alone at the receiver, 2 mask
    replicates per behavior): intact (= tau, ratio 1 by
    construction), dare{p} (drop p, rescale 1/(1-p)),
    drop{p} (drop p, NO rescale -- #307's trim family /
    the scale control), for p in {0.5, 0.9}. The SAME mask
    feeds the dare and drop variants (paired).
  composition: avg (plain mean, #307's champion) vs
    dare_avg{p} (drop+rescale each tau, then mean) vs
    drop_avg{p} (drop, no rescale, then mean); plus the
    #307 calibration arms plain / ties / rand_trim.
  full-sum: plain (sum) vs dare_sum{p} (drop+rescale
    each tau, then sum) vs drop_sum90 (drop, no rescale,
    then sum).

Readouts, all anchored at the receiver theta_A:
  lossret  per behavior, the loss gain a delta realizes
           at the receiver divided by the gain the intact
           solo delta realizes at the receiver
  proj     drift-projection retention <D, D_solo> /
           ||D_solo||^2 at the receiver (#282's drift
           language; the first-order scale readout)
  A_ret    receiver-behavior retention (cross-behavior
           interference readout)

Measured on pop-os (particlegan 0.10.1), 3 seeds:
  CALIBRATION: plain / avg / ties / rand_trim lossret
  reproduce PR #307 exactly (plain -0.16/-2.44/-0.87;
  avg 0.59/0.19/0.45; ties 0.47/-0.35/0.04; rand_trim
  0.47/0.10/0.45; sign-conflict 0.71-0.76).
  (a) SINGLE-DELTA: the rescale law holds in the drift /
  scale readout at BOTH drop rates -- dare proj
  0.99-1.01 (p=0.5) and 0.96-1.01 (p=0.9), while the
  no-rescale drop's proj sits at the kept fraction
  (0.54-0.55 at p=0.5; 0.11-0.12 at p=0.9). In LOSS gain
  the paper's "essentially intact" holds at p=0.5
  (dare50 lossret 0.89/0.89/0.96) but FAILS at p=0.9:
  dare90 lossret 0.62/-0.52/0.59 -- the rescaled sparse
  delta has the right expected drift, but a single mask
  draw's variance destroys the realized loss gain on one
  seed in three. Drop-without-rescale lossret is the
  stable shrinkage profile (0.80/0.88/0.80 at p=0.5;
  0.22/0.21/0.23 at p=0.9).
  (b) COMPOSITION: DARE-merge does NOT beat plain
  averaging. dare_avg50 tracks avg within 0.05 on every
  seed (0.55/0.22/0.42 vs 0.59/0.19/0.45) -- the rescale
  exactly cancels the drop in expectation, buying
  nothing over the mean -- and dare_avg90 falls behind
  (0.52/-0.33/0.27; seed-mean 0.15 vs avg 0.41).
  drop_avg90 is a pure shrinkage arm (lossret ~0.09,
  A_ret ~0.99: barely moves, barely disturbs).
  (c) FULL-SUM: DARE does NOT repair the sum here
  either (Altair's null reproduces): dare_sum50 stays
  catastrophic (-0.36/-1.93/-1.04, cf. plain
  -0.16/-2.44/-0.87) and dare_sum90 is far WORSE than
  plain on every seed (-0.82/-5.57/-2.43). The arm that
  repairs the sum is drop-without-rescale
  (drop_sum90 0.28/0.11/0.23) -- shrinkage, again.

Verdict: the 1/(1-p) rescale is an unbiased scale
correction (proven in the drift readout) and preserves
a lone delta's loss gain at p=0.5, but it is NOT a
merge-interference remedy in this class and does not
make p=0.9 single-delta loss retention reliable.
Assertions encode exactly these margins. Pure torch --
no GAN machinery. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
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


DIN, HID, COUT = 16, 96, 4    # toy MLP shape (PR #307)
SIG = 0.5                     # per-blob input width
MEAN_SCALE = 2.5              # blob-mean draw scale
DV_SCALE = 1.5                # per-task teacher deviation
FT_STEPS = 40                 # per-task fine-tune steps
FT_LR = 0.005                 # fine-tune Adam step
PRE_STEPS = 600               # base pretraining steps
KEEP = 0.2                    # TIES trim keep-fraction (#307)
N_IN = 3                      # incoming deltas per group
N_GROUPS = 3                  # independent groups per seed
PS = (0.5, 0.9)               # DARE drop rates
COMP_ARMS = ("plain", "avg", "ties", "rand_trim",
             "dare_avg50", "dare_avg90", "drop_avg50",
             "drop_avg90", "dare_sum50", "dare_sum90",
             "drop_sum90")
SINGLE_ARMS = ("dare50", "dare90", "drop50", "drop90")


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


def _elect(taus):
    out = {}
    for k in taus[0]:
        stack = torch.stack([t[k] for t in taus])
        sign = torch.sign(stack.sum(0))
        aligned = torch.where(stack * sign > 0, stack,
                              torch.zeros_like(stack))
        cnt = (aligned != 0).sum(0).clamp(min=1)
        out[k] = aligned.sum(0) / cnt
    return out


def _masked(taus, p, rng):
    """Paired (dare, drop) variants of each tau: one mask
    per tau, rescaled or not -- the ONLY difference between
    the two variants is the 1/(1-p) factor."""
    dare, drop = [], []
    for t in taus:
        masks = {k: (torch.rand(t[k].shape, generator=rng) >= p).float()
                 for k in t}
        dare.append({k: t[k] * masks[k] / (1 - p) for k in t})
        drop.append({k: t[k] * masks[k] for k in t})
    return dare, drop


def run_seed(seed):
    """All readouts for one seed, averaged over groups.

    Returns the composition arms' (lossret, A_ret), the
    single-delta arms' (lossret, proj) over 2 mask
    replicates per behavior, the sign-conflict rate, and
    the per-group count of behaviors with a valid solo
    gain at the receiver.
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

    comp_agg = {name: [] for name in COMP_ARMS}
    sing_agg = {name: [] for name in SINGLE_ARMS}
    sing_proj = {name: [] for name in SINGLE_ARMS}
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
        # Paired masks: replicate 1 drives composition and
        # single-delta; replicate 2 is single-delta only.
        reps = []
        for rseed in (4242, 7777):
            rng = torch.Generator().manual_seed(rseed + seed * 100 + grp)
            reps.append({p: _masked(taus, p, rng) for p in PS})
        rng = torch.Generator().manual_seed(999 + seed * 10 + grp)
        trimmed = [_trim(t, KEEP) for t in taus]
        rtrim = [_trim(t, KEEP, rng=rng, random=True) for t in taus]
        d50, n50 = reps[0][0.5]
        d90, n90 = reps[0][0.9]
        arms = {
            "plain": {k: sum(t[k] for t in taus) for k in taus[0]},
            "avg": {k: sum(t[k] for t in taus) / len(taus)
                    for k in taus[0]},
            "ties": _elect(trimmed),
            "rand_trim": {k: sum(t[k] for t in rtrim) for k in taus[0]},
            "dare_avg50": {k: sum(t[k] for t in d50) / len(d50)
                           for k in taus[0]},
            "dare_avg90": {k: sum(t[k] for t in d90) / len(d90)
                           for k in taus[0]},
            "drop_avg50": {k: sum(t[k] for t in n50) / len(n50)
                           for k in taus[0]},
            "drop_avg90": {k: sum(t[k] for t in n90) / len(n90)
                           for k in taus[0]},
            "dare_sum50": {k: sum(t[k] for t in d50) for k in taus[0]},
            "dare_sum90": {k: sum(t[k] for t in d90) for k in taus[0]},
            "drop_sum90": {k: sum(t[k] for t in n90) for k in taus[0]},
        }
        valid = 0
        for i, (v, m) in enumerate(tasks):
            solo = {k: th_a[k] + taus[i][k] for k in base}
            sg = (_loss(th_a, v, m, 9100 + seed)
                  - _loss(solo, v, m, 9100 + seed))
            if sg <= 0.02:
                continue
            valid += 1
            xp, _ = _data(v, m, 128, 9100 + seed)
            ds = (_fwd(solo, xp) - _fwd(th_a, xp)).flatten()
            for rep in reps:
                for p in PS:
                    tag = f"{int(p * 100)}"
                    for kind, src in (("dare", rep[p][0][i]),
                                      ("drop", rep[p][1][i])):
                        comp = {k: th_a[k] + src[k] for k in base}
                        cg = (_loss(th_a, v, m, 9100 + seed)
                              - _loss(comp, v, m, 9100 + seed))
                        dc = (_fwd(comp, xp) - _fwd(th_a, xp)).flatten()
                        sing_agg[f"{kind}{tag}"].append(cg / sg)
                        sing_proj[f"{kind}{tag}"].append(
                            float(dc @ ds / (ds @ ds)))
        n_valid.append(valid)
        for name, delta in arms.items():
            comp = {k: th_a[k] + delta[k] for k in base}
            lrets = []
            for i, (v, m) in enumerate(tasks):
                solo = {k: th_a[k] + taus[i][k] for k in base}
                sg = (_loss(th_a, v, m, 9100 + seed)
                      - _loss(solo, v, m, 9100 + seed))
                cg = (_loss(th_a, v, m, 9100 + seed)
                      - _loss(comp, v, m, 9100 + seed))
                if sg > 0.02:
                    lrets.append(cg / sg)
            a_ret = ((_loss(base, va, ma, 9000 + seed)
                      - _loss(comp, va, ma, 9000 + seed)) / a_gain)
            comp_agg[name].append((float(np.mean(lrets)), a_ret))
    comp_out = {name: tuple(float(np.mean([r[j] for r in rows]))
                            for j in range(2))
                for name, rows in comp_agg.items()}
    sing_out = {name: (float(np.mean(sing_agg[name])),
                       float(np.mean(sing_proj[name])))
                for name in SINGLE_ARMS}
    return dict(comp=comp_out, single=sing_out, conflict=conflict,
                n_valid=n_valid)


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_regime_and_calibration_vs_pr307():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = _seed(seed)
        # The #307 interference regime, reproduced: high
        # sign conflict, every solo gain valid, plain sum
        # catastrophic at the receiver.
        assert d["conflict"] >= 0.60, (
            f"seed {seed}: sign-conflict rate {d['conflict']:.3f}")
        assert d["n_valid"] == [N_IN] * N_GROUPS, (
            f"seed {seed}: valid solo gains {d['n_valid']}")
        plain = d["comp"]["plain"]
        assert plain[0] <= 0.20, (
            f"seed {seed}: plain lossret {plain[0]:.3f}")
        assert plain[1] <= 0.35, (
            f"seed {seed}: plain A_ret {plain[1]:.3f}")
        # Calibration ordering from #307: averaging is not
        # beaten by TIES in this class (demonstrated
        # avg-ties gaps +0.12/+0.54/+0.41).
        assert d["comp"]["avg"][0] >= d["comp"]["ties"][0] - 0.05, (
            f"seed {seed}: avg vs ties lossret "
            f"{d['comp']['avg'][0]:.3f} vs {d['comp']['ties'][0]:.3f}")


def test_single_delta_rescale_law():
    _empf_recipe()
    for seed in (0, 1, 2):
        s = _seed(seed)["single"]
        # The rescale law in the drift readout: with the
        # 1/(1-p) factor, the dropped delta's drift at the
        # receiver projects onto the intact drift at ~1.0
        # at BOTH drop rates (demonstrated dare50 proj
        # 0.99-1.01, dare90 proj 0.96-1.01).
        assert 0.85 <= s["dare50"][1] <= 1.15, (
            f"seed {seed}: dare50 proj {s['dare50'][1]:.3f}")
        assert 0.80 <= s["dare90"][1] <= 1.20, (
            f"seed {seed}: dare90 proj {s['dare90'][1]:.3f}")
        # Without the rescale, the projection sits at the
        # KEPT FRACTION (demonstrated drop50 0.54-0.55,
        # drop90 0.11-0.12): the drop alone is a pure
        # shrinkage, and the rescale is exactly what
        # cancels it in expectation.
        assert 0.40 <= s["drop50"][1] <= 0.70, (
            f"seed {seed}: drop50 proj {s['drop50'][1]:.3f}")
        assert 0.03 <= s["drop90"][1] <= 0.25, (
            f"seed {seed}: drop90 proj {s['drop90'][1]:.3f}")
        assert s["dare50"][1] >= s["drop50"][1] + 0.30
        assert s["dare90"][1] >= s["drop90"][1] + 0.50
        # In loss gain, "essentially intact" holds at
        # p=0.5 (demonstrated dare50 lossret
        # 0.97/0.89/0.96) ...
        assert s["dare50"][0] >= 0.75, (
            f"seed {seed}: dare50 lossret {s['dare50'][0]:.3f}")
        assert s["drop50"][0] >= 0.65, (
            f"seed {seed}: drop50 lossret {s['drop50'][0]:.3f}")
        # ... but FAILS at p=0.9: dare90's realized loss
        # gain falls well short of dare50's on every seed
        # (demonstrated 0.62/-0.52/0.59 vs 0.97/0.89/0.96)
        # -- the rescale fixes the expected scale, not the
        # single-draw variance. The no-rescale drop at
        # p=0.9 is the stable small profile (0.21-0.23).
        assert s["dare90"][0] <= s["dare50"][0] - 0.20, (
            f"seed {seed}: dare90 vs dare50 lossret "
            f"{s['dare90'][0]:.3f} vs {s['dare50'][0]:.3f}")
        assert 0.05 <= s["drop90"][0] <= 0.35, (
            f"seed {seed}: drop90 lossret {s['drop90'][0]:.3f}")


def test_dare_merge_does_not_beat_averaging():
    _empf_recipe()
    avg_mean, dare90_mean = [], []
    for seed in (0, 1, 2):
        c = _seed(seed)["comp"]
        avg, d50, d90 = c["avg"], c["dare_avg50"], c["dare_avg90"]
        # DARE-merge at p=0.5 never beats plain averaging
        # by any margin and tracks it closely (demonstrated
        # gaps dare-avg -0.04/+0.03/-0.03): the rescale
        # cancels the drop in expectation and buys nothing
        # over the mean.
        assert d50[0] <= avg[0] + 0.05, (
            f"seed {seed}: dare_avg50 vs avg lossret "
            f"{d50[0]:.3f} vs {avg[0]:.3f}")
        assert abs(d50[0] - avg[0]) <= 0.10, (
            f"seed {seed}: |dare_avg50 - avg| "
            f"{abs(d50[0] - avg[0]):.3f}")
        # At p=0.9 the merge degrades on every seed
        # (demonstrated dare_avg90 0.52/-0.33/0.27 vs
        # dare_avg50 0.55/0.22/0.42).
        assert d90[0] <= d50[0] + 0.02, (
            f"seed {seed}: dare_avg90 vs dare_avg50 lossret "
            f"{d90[0]:.3f} vs {d50[0]:.3f}")
        # The no-rescale averaged drop at p=0.9 is pure
        # shrinkage: tiny incoming gain, receiver almost
        # undisturbed (demonstrated lossret ~0.09,
        # A_ret 0.99-1.00).
        assert c["drop_avg90"][0] <= 0.20, (
            f"seed {seed}: drop_avg90 lossret "
            f"{c['drop_avg90'][0]:.3f}")
        assert c["drop_avg90"][1] >= 0.85, (
            f"seed {seed}: drop_avg90 A_ret "
            f"{c['drop_avg90'][1]:.3f}")
        avg_mean.append(avg[0])
        dare90_mean.append(d90[0])
    # Seed-mean: dare_avg90 falls clearly behind plain
    # averaging (demonstrated 0.15 vs 0.41).
    assert float(np.mean(dare90_mean)) <= float(np.mean(avg_mean)) - 0.10


def test_dare_does_not_repair_full_sum():
    _empf_recipe()
    for seed in (0, 1, 2):
        c = _seed(seed)["comp"]
        plain = c["plain"]
        # Altair's null configuration reproduces in this
        # class: DARE-then-sum does not repair the sum.
        # At p=0.5 it stays catastrophic (demonstrated
        # -0.36/-1.93/-1.04) ...
        assert c["dare_sum50"][0] <= 0.20, (
            f"seed {seed}: dare_sum50 lossret "
            f"{c['dare_sum50'][0]:.3f}")
        # ... and at p=0.9 it is far WORSE than the plain
        # sum on every seed (demonstrated gaps
        # -0.65/-3.13/-1.56): rescaling survivors by 10x
        # amplifies exactly the overshoot that destroys
        # the plain sum.
        assert c["dare_sum90"][0] <= plain[0] - 0.25, (
            f"seed {seed}: dare_sum90 vs plain lossret "
            f"{c['dare_sum90'][0]:.3f} vs {plain[0]:.3f}")
        # The arm that repairs the sum is the one WITHOUT
        # the rescale (demonstrated drop_sum90
        # 0.28/0.11/0.23, positive on every seed) -- the
        # operative lever is shrinkage of the composed
        # delta, per #307's scale conclusion.
        assert c["drop_sum90"][0] >= 0.05, (
            f"seed {seed}: drop_sum90 lossret "
            f"{c['drop_sum90'][0]:.3f}")
        assert c["drop_sum90"][0] >= plain[0] + 0.25, (
            f"seed {seed}: drop_sum90 vs plain lossret "
            f"{c['drop_sum90'][0]:.3f} vs {plain[0]:.3f}")
        assert c["dare_sum90"][0] <= c["drop_sum90"][0] - 0.50, (
            f"seed {seed}: dare_sum90 vs drop_sum90 lossret "
            f"{c['dare_sum90'][0]:.3f} vs {c['drop_sum90'][0]:.3f}")
