"""CPU gate for the two-timescales-of-forgetting claim
(arXiv:2611.33302): "Two Timescales of Forgetting in Fine-tuning
of Language Models: Fast Behavioral Suppression and Slow
Representational Drift" -- SFT on a restricted subset of
generative rules suppresses generation of the deleted rules on
a FAST timescale, with the suppression timescale inversely
proportional to the magnitude of the token-latent correlation
(stronger correlation -> faster suppression), while the hidden
representations supporting the deleted rules stay intact at
that point and are forgotten only on a MUCH longer timescale,
via slow drift of the network parameters. The authors measure
this on transformers pretrained on a controlled hierarchical
grammar (RHM) and echo it in natural language.

Provenance and anchors inside this watch:

  * PR #286 (iterated lineages): across iterated distillation
    generations the behavioral SCREEN decays ~4x faster than
    the internal PROBE -- the same behavior/representation
    split with a different driver (iterated screening there, a
    single SFT run here). This paper supplies the mechanism
    underneath #286's split.
  * The cycle-28 brief note on arXiv:2611.29285: behavior
    cannot separate suppressed from removed; relearning speed
    is the conservative readout. This toy is that warning with
    a clock attached: at the step where behavior is already at
    its floor, the representation probe is still near its
    pretrained value, so a behavioral audit at that step would
    certify a removal that has not happened.
  * PR #334 (CPSD reversed ACSD): a removal claim must cross
    BELOW the plain floor, not merely reach it. Here the
    deleted rules' behavior does cross to ~0 -- and the claim
    still fails at the representation seat, which is exactly
    #334's point priced per-readout.
  * PR #288 (the balloon effect): bans relocate, only
    writer-side cuts remove. SFT relabeling IS a writer-side
    cut on the output head -- and this toy shows even a
    writer-side cut has two timescales: the head moves fast,
    the representation drifts slow.

Toy operationalization. Five latent rules: A (kept), C (kept,
the untargeted control), and three deleted rules B_low,
B_med, B_high whose token-latent correlation rho_z =
P(rule token | rule) in the pretraining mixture is planted at
0.25 / 0.40 / 0.62 (A and C sit at 0.80). Contexts are
continuous: x = prototype_z + Gaussian noise (prototypes are
fixed random vectors of norm 2 in R^20). The model is a small
generative MLP, 20 -> 48 (tanh) -> 7 tokens (the five rule
tokens plus two generic tokens shared by all rules), genuinely
pretrained by full-batch SGD on the mixture until every rule's
behavior and the representation probe are installed.

SFT (the restricted subset): the context stream still visits
every latent, but targets are generated ONLY by the kept
rules -- a deleted rule's contexts are relabeled to the A
token (the deleted production is replaced, never emitted),
while A and C targets are drawn from their own rules as
before. All parameters train, one plain SGD learning rate;
nothing is frozen, so the timescale split is measured, not
built in. The suppression pressure on a deleted rule is the
relabeling gradient, whose size scales with the model's own
P(rule token | rule context) -- hence with rho_z, which is
the paper's pricing in this toy's terms.

Readouts, tracked at checkpoints over 5000 SFT steps:
  * BEHAVIOR_z: mean P(rule token | rule contexts) on a
    held-out eval set -- the generation probability of the
    rule's token.
  * REPRESENTATION (two seats): (a) the FIXED-PROBE MARGIN --
    a linear probe trained once on the FROZEN pretrained
    hidden activations (5-way rule classification) and never
    retrained; per deleted rule, its own-class logit minus
    the best other-class logit on hidden activations h(t);
    (b) CENTROID COSINE between mean h_B(t) and mean h_B(0).
  * CONTROLS: the untargeted kept rule C (behavior +
    representation must stay ~intact throughout), and a
    SHUFFLED-LABEL probe trained on the same frozen
    activations (must sit at the 5-way chance level, 0.20).

Timescales. Behavior half-life: first crossing of the
midpoint between the step-0 behavior and its long-run floor
(the horizon-end value, ~0), linearly interpolated. The
rho-ordering is asserted on the EARLY-PHASE SUPPRESSION
TIMESCALE tau_z = 1 / (mean logit decay rate over steps
0..6): for a softmax generator the log-odds decay rate is
proportional to the model's own token probability, so this
is the rate constant the paper's law prices, read off before
the self-limiting tail (where every integrated half-life
washes the ordering out -- see the iteration note in the PR
body: probability half-lives order low-vs-high on 5/5 seeds
but med/high compress, and logit half-lives invert on 2/5
seeds because their midpoint sits in the floor-noise tail).
Representation half-life: the pooled fixed-probe margin
against its no-information floor of 0 -- it never halves
inside the 5000-step horizon, so the behavior-vs-
representation half-life ratio is reported as a lower bound
(horizon / slowest behavior timescale).

Modeling gifts, stated plainly:
  * rho is planted by construction as P(rule token | rule)
    in the pretraining mixture; pretrained behavior is
    therefore ordered by rho (asserted, not assumed).
  * SFT contexts still visit the deleted latents -- the
    context stream is the grammar's state stream; only the
    TARGET productions are restricted to the kept subset.
  * The probe is trained on frozen pretrained activations
    and frozen thereafter. A RETRAINED probe was tried in
    iteration and saturates at accuracy 1.0 for the whole
    horizon in this toy (the latents stay linearly separable
    long after the fixed probe's margin has measurably
    drifted), so the fixed probe's continuous margin is the
    sensitive representation readout; retrained accuracy is
    not asserted.
  * Full-batch SGD throughout: trajectories are deterministic
    given the seed, and "timescale" means gradient steps.

Measured on pop-os (particlegan 0.10.1), 5 seeds -- per seed:
pretrained behavior (low/med/high), early-phase suppression
timescales tau (low/med/high), probability half-lives
(low/med/high), margin retention + pooled cosine at the
behavior-floor step, pooled margin at horizon end, control C
behavior start->end, shuffled-probe accuracy:
  seed 0: 0.244/0.370/0.591; tau 18.2/8.6/5.9;
          HL 12.4/8.7/7.7; floor 60; 0.863; 0.959;
          7.54/11.30; 403x; 0.758->0.787; 0.193
  seed 1: 0.236/0.382/0.557; tau 23.9/9.4/7.8;
          HL 13.6/9.4/9.4; floor 60; 0.905; 0.965;
          9.08/10.83; 368x; 0.747->0.787; 0.192
  seed 2: 0.217/0.405/0.604; tau 23.7/10.8/5.1;
          HL 12.8/10.5/7.2; floor 60; 0.918; 0.959;
          7.84/10.63; 389x; 0.757->0.805; 0.219
  seed 3: 0.209/0.358/0.578; tau 17.9/11.0/4.3;
          HL 12.2/10.2/5.7; floor 60; 0.888; 0.963;
          6.89/10.02; 409x; 0.777->0.777; 0.206
  seed 4: 0.200/0.366/0.582; tau 14.8/9.5/4.8;
          HL 11.4/9.5/6.0; floor 60; 0.852; 0.953;
          6.64/10.23; 439x; 0.732->0.812; 0.221
The behavior floor step is 60 on every seed; the pooled
margin half-life is infinite (never halves) on every seed,
so the ratio column is the lower bound 5000 / (slowest
behavior half-life); the probe accuracy is 0.999-1.000 on
every seed.

What is NOT asserted: the paper's exact constants, anything
about the RHM's hierarchy as such, and anything about real
LM fine-tuning (contexts here are prototype-plus-noise, the
model is a 2-layer MLP, and rho is a single planted number
per rule, where a real token-latent correlation is a property
of a corpus). The assertions encode exactly the demonstrated
margins -- the rho ordering on the early-phase timescale,
the probability half-life ordering low-vs-high, the
representation's intactness at the behavior floor, and the
half-life ratio lower bound -- with headroom, no more.
Pure torch toy on CPU, single-threaded, seeded. Skips
cleanly when the installed particlegan does not provide the
EMPF recipe (the house guard for this paper-test family,
PR #273's guarded-import pattern).
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

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


# ------------------------------------------------------------------ world
RULES = ("A", "Blow", "Bmed", "Bhigh", "C")
DELETED = ("Blow", "Bmed", "Bhigh")
RHO = {"A": 0.80, "Blow": 0.25, "Bmed": 0.40, "Bhigh": 0.62, "C": 0.80}
TOK = {"A": 0, "Blow": 1, "Bmed": 2, "Bhigh": 3, "C": 4}
V = 7            # 5 rule tokens + 2 generic tokens (5, 6)
D = 20           # context dimension
H = 48           # hidden width
NOISE = 0.3
PROTO_NORM = 2.0
N_TRAIN = 800    # contexts per rule, train and SFT corpora
N_EVAL = 400     # contexts per rule, held-out eval set
PRE_STEPS = 300
PRE_LR = 0.05
SFT_STEPS = 5000
SFT_LR = 0.10
EARLY_WINDOW = 10  # steps over which the suppression rate is read
CHECKPOINTS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 34, 44,
               60, 80, 120, 200, 300, 500, 800, 1200, 1800, 2400,
               3000, 4000, 5000)
SEEDS = (0, 1, 2, 3, 4)


def _make_data(rng, n_per, protos, sft=False):
    xs, ys, zs = [], [], []
    for zi, z in enumerate(RULES):
        for _ in range(n_per):
            xs.append(protos[zi] + rng.normal(0, NOISE, D))
            zs.append(zi)
            if sft and z in DELETED:
                y = TOK["A"]  # deleted production replaced by kept A
            elif rng.random() < RHO[z]:
                y = TOK[z]
            else:
                y = 5 + int(rng.integers(2))
            ys.append(y)
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.long),
            np.array(zs))


class _Net(torch.nn.Module):
    """20 -> 48 tanh -> 7 logits; every parameter trains in both
    phases (the timescale split is measured, not built in)."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.nn.Parameter(
            torch.randn(H, D, generator=g) * (2.0 / math.sqrt(D)))
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(
            torch.randn(V, H, generator=g) * (0.9 / math.sqrt(H)))
        self.b2 = torch.nn.Parameter(torch.zeros(V))

    def hidden(self, x):
        return torch.tanh(x @ self.W1.T + self.b1)

    def forward(self, x):
        return self.hidden(x) @ self.W2.T + self.b2


def _train_probe(hf, labels, seed, shuffled=False, steps=300):
    """Linear probe on FROZEN activations (half train / half eval).
    Returns (W, b, eval accuracy)."""
    rng = np.random.default_rng(seed)
    lab = labels.copy()
    if shuffled:
        rng.shuffle(lab)
    idx = rng.permutation(len(labels))
    tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]
    W = torch.zeros(H, len(RULES), requires_grad=True)
    b = torch.zeros(len(RULES), requires_grad=True)
    xt = torch.tensor(hf[tr], dtype=torch.float32)
    yt = torch.tensor(lab[tr])
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(xt @ W + b, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.tensor(hf[te], dtype=torch.float32)
                @ W.detach() + b.detach()).argmax(1).numpy()
    return W.detach(), b.detach(), float((pred == lab[te]).mean())


def _half_life(steps, vals, floor):
    """First crossing of (vals[0] + floor) / 2, linearly
    interpolated; inf if never crossed."""
    mid = (vals[0] + floor) / 2.0
    for i in range(1, len(vals)):
        if vals[i] <= mid:
            t0, t1 = steps[i - 1], steps[i]
            v0, v1 = vals[i - 1], vals[i]
            return t0 + (t1 - t0) * (v0 - mid) / max(v0 - v1, 1e-12)
    return float("inf")


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


_CACHE = {}


def run_seed(seed):
    if seed in _CACHE:
        return _CACHE[seed]
    rngp = np.random.default_rng(1000 + seed)
    protos = rngp.normal(0, 1, (len(RULES), D))
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True) \
        * PROTO_NORM
    model = _Net(seed)
    # --- pretraining on the full mixture of rules
    X, y, _ = _make_data(np.random.default_rng(2000 + seed),
                         N_TRAIN, protos)
    opt = torch.optim.SGD(model.parameters(), lr=PRE_LR)
    for _ in range(PRE_STEPS):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # --- held-out eval set + frozen pretrained probe
    Xe, _, Ze = _make_data(np.random.default_rng(3000 + seed),
                           N_EVAL, protos)
    with torch.no_grad():
        h_pre = model.hidden(Xe).numpy()
    centroids0 = {zi: h_pre[Ze == zi].mean(0) for zi in range(len(RULES))}
    Wp, bp, probe_acc = _train_probe(h_pre, Ze, seed)
    _, _, shuffled_acc = _train_probe(h_pre, Ze, seed, shuffled=True)

    def behavior(m):
        with torch.no_grad():
            p = F.softmax(m(Xe), dim=1).numpy()
        return {z: float(p[Ze == zi][:, TOK[z]].mean())
                for zi, z in enumerate(RULES)}

    def fixed_margins(m):
        with torch.no_grad():
            lg = m.hidden(Xe) @ Wp + bp
        out = {}
        for z in DELETED:
            zi = RULES.index(z)
            msk = torch.tensor(Ze == zi)
            own = lg[msk][:, zi]
            other = lg[msk].clone()
            other[:, zi] = -1e9
            out[z] = float((own - other.max(1).values).mean())
        return out

    def cosines(m):
        with torch.no_grad():
            hf = m.hidden(Xe).numpy()
        out = {}
        for zi, z in enumerate(RULES):
            c = hf[Ze == zi].mean(0)
            c0 = centroids0[zi]
            out[z] = float(c @ c0 / np.linalg.norm(c)
                           / np.linalg.norm(c0))
        return out

    steps = [0]
    b0 = behavior(model)
    beh = {z: [b0[z]] for z in RULES}
    m0 = fixed_margins(model)
    mar = {z: [m0[z]] for z in DELETED}
    cos = {z: [1.0] for z in RULES}
    # --- SFT on the restricted subset (kept-rule targets only)
    Xs, ys, _ = _make_data(np.random.default_rng(4000 + seed),
                           N_TRAIN, protos, sft=True)
    opt = torch.optim.SGD(model.parameters(), lr=SFT_LR)
    check = set(CHECKPOINTS)
    for s in range(1, SFT_STEPS + 1):
        loss = F.cross_entropy(model(Xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in check:
            steps.append(s)
            bb = behavior(model)
            mm = fixed_margins(model)
            cc = cosines(model)
            for z in RULES:
                beh[z].append(bb[z])
                cos[z].append(cc[z])
            for z in DELETED:
                mar[z].append(mm[z])
    out = {"steps": steps, "beh": beh, "mar": mar, "cos": cos,
           "probe_acc": probe_acc, "shuffled_acc": shuffled_acc}
    # derived: probability half-lives (floor = horizon-end value)
    out["prob_hl"] = {z: _half_life(steps, beh[z], beh[z][-1])
                      for z in DELETED}
    # derived: early-phase suppression timescale tau = 1 / rate,
    # rate = mean logit decay over steps 0..EARLY_WINDOW
    i_w = steps.index(EARLY_WINDOW)
    out["tau"] = {}
    for z in DELETED:
        rate = (_logit(beh[z][0]) - _logit(beh[z][i_w])) / EARLY_WINDOW
        out["tau"][z] = 1.0 / rate
    # derived: behavior-floor step = first checkpoint where every
    # deleted rule's behavior is <= 0.02, and the representation
    # readouts at that checkpoint
    floor_idx = None
    for i, s in enumerate(steps):
        if all(beh[z][i] <= 0.02 for z in DELETED):
            floor_idx = i
            break
    out["floor_step"] = steps[floor_idx] if floor_idx is not None else None
    if floor_idx is not None:
        pooled_mar0 = float(np.mean([mar[z][0] for z in DELETED]))
        pooled_mar_f = float(np.mean([mar[z][floor_idx] for z in DELETED]))
        out["margin_retention_at_floor"] = pooled_mar_f / pooled_mar0
        out["cos_at_floor"] = float(np.mean([cos[z][floor_idx]
                                             for z in DELETED]))
    # derived: pooled margin half-life vs the no-information
    # floor 0 (inf if it never halves inside the horizon)
    pooled_mar = [float(np.mean([mar[z][i] for z in DELETED]))
                  for i in range(len(steps))]
    out["pooled_margin"] = pooled_mar
    out["margin_hl"] = _half_life(steps, pooled_mar, 0.0)
    _CACHE[seed] = out
    return out


# ------------------------------------------------------------------- tests
def test_pretraining_installs_rules_probe_and_controls():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        b = r["beh"]
        # rho planted by construction -> pretrained behavior is
        # ordered by rho and well above the generic-token level.
        assert b["Bhigh"][0] >= b["Bmed"][0] + 0.10, (
            f"seed {seed}: pretrained Bhigh {b['Bhigh'][0]:.3f} vs "
            f"Bmed {b['Bmed'][0]:.3f}")
        assert b["Bmed"][0] >= b["Blow"][0] + 0.08, (
            f"seed {seed}: pretrained Bmed {b['Bmed'][0]:.3f} vs "
            f"Blow {b['Blow'][0]:.3f}")
        assert b["Blow"][0] >= 0.15, (
            f"seed {seed}: pretrained Blow {b['Blow'][0]:.3f}")
        assert b["A"][0] >= 0.60 and b["C"][0] >= 0.60, (
            f"seed {seed}: pretrained A/C {b['A'][0]:.3f}/"
            f"{b['C'][0]:.3f}")
        # The representation probe reads the latents back off
        # frozen hidden activations; the shuffled-label probe
        # sits at the 5-way chance level (0.20).
        assert r["probe_acc"] >= 0.95, (
            f"seed {seed}: probe accuracy {r['probe_acc']:.3f}")
        assert 0.10 <= r["shuffled_acc"] <= 0.32, (
            f"seed {seed}: shuffled probe {r['shuffled_acc']:.3f}")


def test_behavior_suppression_is_fast_and_rho_ordered():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        tau, hl, beh = r["tau"], r["prob_hl"], r["beh"]
        # The paper's law: stronger token-latent correlation ->
        # FASTER behavioral suppression (smaller timescale).
        assert tau["Bhigh"] <= 0.90 * tau["Bmed"], (
            f"seed {seed}: tau high {tau['Bhigh']:.1f} vs med "
            f"{tau['Bmed']:.1f}")
        assert tau["Bmed"] <= 0.75 * tau["Blow"], (
            f"seed {seed}: tau med {tau['Bmed']:.1f} vs low "
            f"{tau['Blow']:.1f}")
        # Probability half-lives order at the extremes on every
        # seed (med/high compress in the self-limiting tail --
        # see docstring -- so the strict per-seed ordering is
        # asserted on tau, the extremes on half-lives).
        assert hl["Bhigh"] <= 0.75 * hl["Blow"], (
            f"seed {seed}: prob HL high {hl['Bhigh']:.1f} vs low "
            f"{hl['Blow']:.1f}")
        # Fast in absolute terms: every deleted rule is at its
        # floor while SFT has barely started, and stays there.
        i80 = r["steps"].index(80)
        for z in DELETED:
            assert beh[z][i80] <= 0.03, (
                f"seed {seed}: {z} behavior at step 80 "
                f"{beh[z][i80]:.3f}")
            assert beh[z][-1] <= 0.005, (
                f"seed {seed}: {z} behavior at horizon end "
                f"{beh[z][-1]:.4f}")


def test_representation_intact_at_behavior_floor_and_c_control():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        assert r["floor_step"] is not None and r["floor_step"] <= 120, (
            f"seed {seed}: behavior floor step {r['floor_step']}")
        # Claim (ii), first half: at the step where behavior has
        # already reached its floor, the representation is still
        # near its pretrained value -- a behavioral audit here
        # would certify a removal that has not happened.
        assert r["margin_retention_at_floor"] >= 0.80, (
            f"seed {seed}: margin retention at floor "
            f"{r['margin_retention_at_floor']:.3f}")
        assert r["cos_at_floor"] >= 0.93, (
            f"seed {seed}: pooled cosine at floor "
            f"{r['cos_at_floor']:.3f}")
        # Untargeted control: kept rule C's behavior and
        # representation stay ~intact through the whole run.
        assert r["beh"]["C"][-1] >= 0.90 * r["beh"]["C"][0], (
            f"seed {seed}: C behavior {r['beh']['C'][0]:.3f} -> "
            f"{r['beh']['C'][-1]:.3f}")
        assert r["cos"]["C"][-1] >= 0.88, (
            f"seed {seed}: C cosine at end {r['cos']['C'][-1]:.3f}")


def test_representation_half_life_far_longer_than_behavior():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # Claim (ii), second half: the representation is
        # forgotten only on a much longer timescale. The pooled
        # fixed-probe margin never halves inside the 5000-step
        # horizon (floor 0 = no information), so its half-life
        # exceeds the horizon and the ratio is a lower bound.
        assert r["margin_hl"] == float("inf") or \
            r["margin_hl"] >= 25 * max(r["prob_hl"].values()), (
                f"seed {seed}: margin HL {r['margin_hl']} vs prob "
                f"HLs {r['prob_hl']}")
        slowest_beh = max(r["prob_hl"].values())
        ratio_lb = SFT_STEPS / slowest_beh
        assert ratio_lb >= 100, (
            f"seed {seed}: half-life ratio lower bound "
            f"{ratio_lb:.0f}x")
        # And the representation is still plainly there at the
        # horizon end -- drifted, not gone.
        retention = r["pooled_margin"][-1] / r["pooled_margin"][0]
        assert retention >= 0.55, (
            f"seed {seed}: pooled margin retention at end "
            f"{retention:.3f}")
