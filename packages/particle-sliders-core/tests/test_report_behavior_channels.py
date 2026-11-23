"""CPU gate for the self-report / behavior coherence claim of
Kocielnik et al., "Representational Control over Self-Report
& Behavior Coherence in LLM Risk-Taking" (arXiv:2610.04125):
across nine steering methods, two behavioral tasks, two
psychometric instruments, and four open-weight LLMs,
directions built from DESCRIPTIONS move self-report but
leave behavior at chance; directions built from BEHAVIOR do
the reverse; the two channels are steered by near-orthogonal
directions; composing one of each moves both together; and
flipping one sign sets the channels in opposition on
69-89% of flipped compositions. (The paper also reports
that diagnosis leaves task-specific directives only ~32% of
behavioral effect after surface confounders are removed;
that sub-claim is not gated here.) Provenance: cycle 16 of
the Particle Sliders paper watch, brief #3 -- the
channel-level version of the readout-split family
(#278/#255/#271): the readout you harvest from decides the
channel you move. Our adaptation: channel-matched harvest
and evaluation -- a direction harvested from
verbalized/descriptive material is a report-channel object
until proven behavioral, and a behavioral eval of a
report-harvested direction reads ~zero even when the
direction is real (the #273 lesson in channel form).

Toy operationalization -- the gift, stated plainly: this
toy BUILDS IN the channel separation the paper DISCOVERS.
Inputs x in R^12 carry (0) a description-side trait
component, (1) a behavior-side trait component, and (2-11)
nuisance components. A frozen random map W (per seed) makes
the shared hidden representation

    h(x) = W x / sqrt(12) + eps,   eps ~ N(0, 0.3^2 I),

in R^128, and both readout heads see phi = tanh(h):

  * report head: ridge fit of phi against the verbalized
    trait (component 0) on 4000 fresh items;
  * behavior head: ridge fit of phi against the enacted
    trait (component 1); the action choice is
    Bernoulli(sigmoid(1.5 * behavior logit)) and the
    behavior channel is read as the choice probability.

What the toy actually tests is the harvest/steering
CONSEQUENCES of that structure: which contrasts recover
which direction, and what steering each recovered direction
does to each channel. It tests nothing about whether real
LLM representations separate this cleanly -- that is the
paper's empirical finding, taken as the toy's premise.

Harvest (the fit_direction construction, paired contrasts):
description pairs share nuisance, hold the behavior trait
at 0, and set the described trait to +1 vs -1;
d_desc = mean difference in h over 256 pairs. Behavior
pairs likewise for component 1 -> d_beh. Steering:
h' = h + d (alpha = 1: one harvested direction is exactly
the -1 -> +1 trait swing it was harvested from), applied to
neutral items (both traits 0). Each channel's shift is
normalized by that channel's unsteered trait effect size
(the readout difference between trait +1 and -1 items).

Controls, at natural scale: (a) a random direction with
norm matched to ||d_desc||; (b) a shuffled-pairing harvest
-- the description pool's +/- labels permuted before the
mean difference, so the pairing that defines the contrast
is destroyed.

Measured on pop-os (particlegan 0.10.1), 3 seeds --
normalized (report, behavior) shifts:
  d_desc:        rep +0.95/+0.92/+0.94, beh +0.01/+0.00/-0.01
  d_beh:         rep -0.01/-0.00/-0.01, beh +0.69/+0.69/+0.68
  cos(d_desc, d_beh): -0.07/-0.01/-0.12
  d_desc + d_beh: rep +0.89/+0.86/+0.89, beh +0.68/+0.67/+0.66
  d_desc - d_beh: rep +0.88/+0.88/+0.83, beh -0.68/-0.68/-0.67
  opposition fraction (flip, per item): 1.00/1.00/1.00
  random control:  |rep| <= 0.17, |beh| <= 0.16
  shuffled control: |rep| <= 0.01, |beh| <= 0.01
Unsteered trait effects: report 2.03/2.09/2.06 readout
units, behavior 0.64/0.64/0.65 choice-probability units.

The toy's opposition fraction is 1.00, above the paper's
69-89%: in this toy the channels are cleanly separated and
every neutral item sits in the behavior head's responsive
region, so the flipped composition opposes on every item.
The paper's sub-100% range is across heterogeneous real
models, tasks, and instruments; the gate asserts the toy's
demonstrated fraction with headroom (>= 0.90), not the
paper's number. Assertions below encode exactly the
demonstrated margins, no more.

What is NOT asserted: anything about real LLM
self-report/behavior coherence; the paper's ~32%
diagnosis-residual sub-claim; steering at other strengths
(alpha is fixed at the harvested scale); and channel
separation under a NON-paired harvest (the shuffled
control shows the pairing is load-bearing for recovery,
but a systematic unpaired-harvest study is not gated).
Pure numpy -- no GAN machinery, because the claim is about
contrast harvest and readout channels, not training
dynamics. Skips cleanly when the installed particlegan
does not provide the EMPF recipe (the house guard for
this paper-test family; PR #211 is what enables this test
in CI).
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


D = 128              # hidden representation dimension
KIN = 12             # input components: 0 desc, 1 beh, 2-11 nuisance
N_FIT = 4000         # head-fitting items per seed
N_PAIRS = 256        # contrast pairs per harvest
N_EVAL = 400         # neutral / trait-effect items per seed
RIDGE = 1.0          # head-fit ridge penalty
BEH_GAIN = 1.5       # behavior logit -> choice probability gain
SIGMA_H = 0.3        # representation observation noise


def _represent(W, x, rng):
    """Shared hidden representation h(x) (frozen map + noise)."""
    return (x @ W.T) / np.sqrt(KIN) \
        + SIGMA_H * rng.standard_normal((x.shape[0], D))


def _readout(phi, w):
    return np.hstack([phi, np.ones((phi.shape[0], 1))]) @ w


def _choice_prob(phi, w):
    return 1.0 / (1.0 + np.exp(-BEH_GAIN * _readout(phi, w)))


def _fit_heads(W, seed):
    """Ridge-fit the report and behavior heads on fresh items."""
    rng = np.random.default_rng(42000 + seed)
    x = rng.standard_normal((N_FIT, KIN))
    phi = np.tanh(_represent(W, x, rng))
    Phi = np.hstack([phi, np.ones((N_FIT, 1))])
    A = Phi.T @ Phi + RIDGE * np.eye(D + 1)
    w_rep = np.linalg.solve(A, Phi.T @ x[:, 0])
    w_beh = np.linalg.solve(A, Phi.T @ x[:, 1])
    return w_rep, w_beh


def _contrast_pool(W, seed, trait, stage):
    """Paired contrast items: trait +1 vs -1, other trait 0,
    nuisance shared within each pair."""
    rng = np.random.default_rng(stage + seed)
    base = rng.standard_normal((N_PAIRS, KIN))
    base[:, 0:2] = 0.0
    xp = base.copy()
    xp[:, trait] = 1.0
    xm = base.copy()
    xm[:, trait] = -1.0
    return _represent(W, xp, rng), _represent(W, xm, rng)


def _harvest(W, seed, trait, stage):
    """Mean-difference direction from paired contrasts
    (the fit_direction construction)."""
    hp, hm = _contrast_pool(W, seed, trait, stage)
    return np.mean(hp - hm, axis=0)


def _shuffled_harvest(W, seed):
    """Shuffled-pairing control: permute the description
    pool's +/- labels, then take the same mean difference."""
    hp, hm = _contrast_pool(W, seed, 0, 43000)
    h = np.vstack([hp, hm])
    labels = np.array([1.0] * N_PAIRS + [-1.0] * N_PAIRS)
    rng = np.random.default_rng(46000 + seed)
    labels = labels[rng.permutation(2 * N_PAIRS)]
    return np.mean(h[labels > 0], axis=0) \
        - np.mean(h[labels < 0], axis=0)


def run_seed(seed):
    rng_w = np.random.default_rng(41000 + seed)
    W = rng_w.standard_normal((D, KIN))
    w_rep, w_beh = _fit_heads(W, seed)
    d_desc = _harvest(W, seed, 0, 43000)
    d_beh = _harvest(W, seed, 1, 44000)
    cos = float(d_desc @ d_beh
                / (np.linalg.norm(d_desc) * np.linalg.norm(d_beh)))

    # Unsteered trait effect sizes (paired +1 vs -1 items).
    rng_e = np.random.default_rng(45000 + seed)
    base = rng_e.standard_normal((N_EVAL, KIN))
    base[:, 0:2] = 0.0
    eff = {}
    for trait, name in ((0, "rep"), (1, "beh")):
        xp = base.copy()
        xp[:, trait] = 1.0
        xm = base.copy()
        xm[:, trait] = -1.0
        hp = np.tanh(_represent(W, xp, rng_e))
        hm = np.tanh(_represent(W, xm, rng_e))
        if name == "rep":
            eff[name] = float(np.mean(_readout(hp, w_rep)
                                      - _readout(hm, w_rep)))
        else:
            eff[name] = float(np.mean(_choice_prob(hp, w_beh)
                                      - _choice_prob(hm, w_beh)))

    # Steering on neutral items (both traits 0).
    base0 = rng_e.standard_normal((N_EVAL, KIN))
    base0[:, 0:2] = 0.0
    h0 = _represent(W, base0, rng_e)
    rep0 = _readout(np.tanh(h0), w_rep)
    beh0 = _choice_prob(np.tanh(h0), w_beh)

    def shifts(d):
        phi = np.tanh(h0 + d)
        rs = _readout(phi, w_rep) - rep0
        bs = _choice_prob(phi, w_beh) - beh0
        return rs, bs

    def normalized(d):
        rs, bs = shifts(d)
        return float(np.mean(rs) / eff["rep"]), \
            float(np.mean(bs) / eff["beh"]), rs, bs

    arms = {}
    for name, d in (("desc", d_desc), ("beh", d_beh),
                    ("comp", d_desc + d_beh),
                    ("flip", d_desc - d_beh)):
        rep_n, beh_n, _, _ = normalized(d)
        arms[name] = (rep_n, beh_n)
    _, _, flip_rs, flip_bs = normalized(d_desc - d_beh)
    opp_frac = float(np.mean((flip_rs > 0) & (flip_bs < 0)))

    rng_c = np.random.default_rng(47000 + seed)
    d_rand = rng_c.standard_normal(D)
    d_rand *= np.linalg.norm(d_desc) / np.linalg.norm(d_rand)
    rand_rep, rand_beh, _, _ = normalized(d_rand)
    shuf_rep, shuf_beh, _, _ = normalized(_shuffled_harvest(W, seed))

    return dict(cos=cos, eff=eff, arms=arms, opp_frac=opp_frac,
                rand=(rand_rep, rand_beh),
                shuf=(shuf_rep, shuf_beh))


def test_channel_harvests_are_channel_selective():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Normalization is meaningful: both channels have a
        # substantial unsteered trait effect.
        assert r["eff"]["rep"] > 1.0, (
            f"seed {seed}: report trait effect "
            f"{r['eff']['rep']:.3f}")
        assert r["eff"]["beh"] > 0.4, (
            f"seed {seed}: behavior trait effect "
            f"{r['eff']['beh']:.3f}")
        desc_rep, desc_beh = r["arms"]["desc"]
        beh_rep, beh_beh = r["arms"]["beh"]
        # (1) The description-harvested direction steers
        # the report channel materially (demonstrated
        # +0.92-0.95) and the behavior channel ~not
        # (demonstrated |.| <= 0.01, i.e. <= 0.25x its own
        # effect by a wide margin).
        assert desc_rep >= 0.75, (
            f"seed {seed}: d_desc report shift {desc_rep:+.3f}")
        assert abs(desc_beh) <= 0.10, (
            f"seed {seed}: d_desc behavior shift {desc_beh:+.3f}")
        assert abs(desc_beh) <= 0.25 * desc_rep, (
            f"seed {seed}: d_desc cross/own "
            f"{abs(desc_beh) / desc_rep:.3f}")
        # (2) The behavior-harvested direction does the
        # reverse (demonstrated beh +0.68-0.69, rep |.| <=
        # 0.01).
        assert beh_beh >= 0.55, (
            f"seed {seed}: d_beh behavior shift {beh_beh:+.3f}")
        assert abs(beh_rep) <= 0.10, (
            f"seed {seed}: d_beh report shift {beh_rep:+.3f}")
        assert abs(beh_rep) <= 0.25 * beh_beh, (
            f"seed {seed}: d_beh cross/own "
            f"{abs(beh_rep) / beh_beh:.3f}")


def test_directions_near_orthogonal_and_controls_inert():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # (3) Near-orthogonal channels (demonstrated |cos|
        # <= 0.12).
        assert abs(r["cos"]) <= 0.20, (
            f"seed {seed}: cos(d_desc, d_beh) {r['cos']:+.3f}")
        # Random-direction control, norm-matched: moves
        # neither channel materially (demonstrated |rep|
        # <= 0.17, |beh| <= 0.16, vs own effects >= 0.68).
        assert abs(r["rand"][0]) <= 0.25, (
            f"seed {seed}: random report shift {r['rand'][0]:+.3f}")
        assert abs(r["rand"][1]) <= 0.25, (
            f"seed {seed}: random behavior shift {r['rand'][1]:+.3f}")
        # Shuffled-pairing harvest control: destroying the
        # contrast pairing leaves ~no direction and ~no
        # steering (demonstrated |rep| <= 0.01, |beh| <=
        # 0.01).
        assert abs(r["shuf"][0]) <= 0.15, (
            f"seed {seed}: shuffled report shift {r['shuf'][0]:+.3f}")
        assert abs(r["shuf"][1]) <= 0.15, (
            f"seed {seed}: shuffled behavior shift "
            f"{r['shuf'][1]:+.3f}")


def test_composition_moves_both_and_flip_opposes():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        comp_rep, comp_beh = r["arms"]["comp"]
        flip_rep, flip_beh = r["arms"]["flip"]
        # (4) Composition moves BOTH channels materially
        # (demonstrated rep +0.86-0.89, beh +0.66-0.68).
        assert comp_rep >= 0.70, (
            f"seed {seed}: composition report shift "
            f"{comp_rep:+.3f}")
        assert comp_beh >= 0.55, (
            f"seed {seed}: composition behavior shift "
            f"{comp_beh:+.3f}")
        # (5) Sign-flipped composition sets the channels in
        # opposition: report follows d_desc (demonstrated
        # +0.83-0.88) while behavior reverses (demonstrated
        # -0.68 to -0.67), on a substantial fraction of items
        # (demonstrated 1.00; the paper's 69-89% is across
        # real models/tasks -- the gate asserts the toy's
        # fraction with headroom, directionally > 1/2).
        assert flip_rep >= 0.70, (
            f"seed {seed}: flipped report shift {flip_rep:+.3f}")
        assert flip_beh <= -0.55, (
            f"seed {seed}: flipped behavior shift {flip_beh:+.3f}")
        assert r["opp_frac"] >= 0.90, (
            f"seed {seed}: opposition fraction "
            f"{r['opp_frac']:.3f}")
