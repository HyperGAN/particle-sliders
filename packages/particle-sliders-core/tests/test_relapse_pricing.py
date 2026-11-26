"""CPU gate for the relapse-pricing chain claim (chain test:
PR #344 x PR #334): after a trait has been behaviorally
"removed", does its representation residue price its RELEARNING
speed? #344 (arXiv:2611.33302, two timescales) proved behavioral
suppression is fast and correlation-priced while the
representation persists >=368x longer, and concluded removal
claims need a probe/relearning readout with a timescale. #334
(CPSD, arXiv:2611.31932) proved reversed-direction distillation
ON-POLICY removes an installed trait (behavior crosses below
the plain floor) while the same target distilled OFF-POLICY
lands short. Neither priced what happens NEXT. This test chains
them: install a trait, remove it three ways, then RELEARN it
and count steps.

Claim: a behaviorally removed trait relearns far faster than a
fresh trait installs, and relearning speed tracks the retained
representation -- behavioral removal is not erasure, and
relearning steps are the readout that prices the residue. If a
removed trait relearned like a fresh one, the claim would fail
and removal in this class would really erase; the assertions
below encode the demonstrated margins only.

Toy operationalization (the #344 world, one target trait).
Contexts are continuous: x = center + Gaussian noise in R^20.
Three latents: A (kept sibling), C (untargeted control), and
the target trait B, whose contexts are drawn from TWO
sub-centers symmetric about A's prototype (cos(A, B_lobe) =
0.85), so B's identity is nonlinear in the base representation:
no linear readout of the base hidden separates B well (base
fixed-probe margin 1.5-3.4 vs C's 10.9-12.2). The model is #344's
generative MLP, 20 -> 48 (tanh) -> 5 tokens (three rule tokens
plus two generic tokens shared by all rules).

Phases:
  * BASE (the plain model): full-batch SGD on a mixture where
    A and B contexts are BOTH labeled with the A token at rate
    0.80 (B is merged into A; the B token is never a base
    target) and C contexts with the C token at 0.80. The plain
    floor -- the base model's own B behavior -- is therefore
    ~0.007-0.009: this plain model never expresses the trait.
  * INSTALL: continue training on a mixture where B contexts
    carry the B token at rate 0.65. Install must build the
    nonlinear B feature, not just attach a head: a fresh
    install takes 226-482 gradient steps to reach the
    reinstall threshold (B behavior >= 0.35), and the
    installed model sits at B behavior 0.42-0.51.
  * REMOVAL arms, each from a clone of the installed model:
      (b) SUPPRESSION (#344's restricted SFT): the base
          labeling (B contexts relabeled to the A token),
          full-batch CE, run to B behavior <= 0.01 -- behavior
          back at the plain floor.
      (a) ON-POLICY removal (#334's CPSD): KL distillation
          toward the counterfactual target cf(x) = 2*base(x) -
          installed(x) in logit space (the privileged-vs-plain
          direction, reversed), on states drawn fresh each
          round with B contexts sampled in proportion to the
          CURRENT student's own P(B|x) -- the visitation
          analog of #334's on-policy rollouts in this
          trajectory-free world. Run to the same endpoint,
          B behavior <= 0.01.
      (d) OFF-POLICY removal (#334's ablation): the same CF
          target and loss, the same number of gradient steps
          as (a) used, but the states are ONE fixed pool of
          60 contexts per rule drawn once under the BASE
          model's sampling weights (the small static corpus).
          It lands short: B behavior 0.014-0.023 at the budget
          where (a) reached <= 0.01.
  * RELEARN: the install procedure again, from each
    post-removal clone; steps until B behavior >= 0.35 (the
    same fixed threshold the fresh install is priced against).
  * FRESH control (c): the install steps from the base model,
    measured during the install phase itself.

Readouts:
  * BEHAVIOR: mean P(rule token | rule contexts) on a held-out
    eval set, per phase.
  * REPRESENTATION at the moment relearning starts, two fixed
    seats, both from #344: (i) the FIXED-PROBE MARGIN -- a
    linear probe trained once on the FROZEN base hidden
    activations (3-way latent classification) and never
    retrained; retention = post-removal B margin / installed
    B margin; (ii) CENTROID COSINE between the mean hidden
    activation on B contexts post-removal and at install.
  * STEPS-TO-REINSTALL at the fixed threshold 0.35.
  * CONTROLS: untargeted rule C (behavior + probe margin must
    survive every phase); a SHUFFLED-LABEL probe on the same
    frozen base activations (must sit at the 3-way chance
    level, 1/3).

What the toy demonstrates (5 seeds): removal by either
endpoint-matched arm returns behavior to the plain floor, and
the off-policy arm lands short at matched budget -- yet every
removed trait relearns in 18-31 steps against 226-482 for a
fresh install (ratio <= 0.11), with the on-policy and
suppression arms within 1.3x of each other and the
off-policy arm fastest or tied (its behavior never fully
left). The two representation seats SPLIT, which is part of
the result: the fixed-probe margin collapses under removal
(retention -0.70..+0.58 -- removal optimizes directly against
any fixed linear readout of the trait), while the centroid
geometry is essentially untouched (cosine 0.978-0.990 on every
arm and seed). Relearning sides with the geometry: an audit
that reads only behavior, or only a fixed probe's margin,
would certify an erasure that a 20-step relearn refutes.

Modeling gifts, stated plainly:
  * B is PLANTED two-sided about A so that install carries a
    representation cost; with an orthogonal, linearly
    separable trait (the #344 world as-is) install is nearly
    pure head attachment and fresh-vs-relearn is unmeasurable
    (prototyped: fresh 5 steps, relearn 10). The claim is
    priced for traits whose expression needs a built feature.
  * The base labeling merges B into A, so the plain floor is
    ~0.008 and #334's below-the-floor crossing is not
    available in this world; the removal endpoint is "at the
    floor" (<= 0.01), stated as such, not dressed as a
    crossing.
  * The CF direction is computed exactly from the two frozen
    networks (base and installed) in logit space -- the same
    gift as #334's scripted teacher (there the contrast was
    exact by construction; here it is exact because both
    endpoints of the install are in hand).
  * On-policy state sampling is expression-weighted fresh
    draws, not literal trajectories: the #344 world has no
    trajectory structure, so #334's sticky-tail visitation is
    represented as sampling weight proportional to the
    student's current trait expression. The off-policy pool
    is thin by size (60/rule, reused), the static-corpus
    analog of #334's thin tail coverage.
  * Full-batch or fixed-batch SGD throughout; trajectories
    are deterministic given the seed, and "steps" means
    gradient steps (batch sizes differ by arm and are noted
    in the constants).

Measured on pop-os (particlegan 0.10.1), 5 seeds -- per seed:
plain floor / fresh steps / installed behavior; suppression
(behavior, margin retention, cosine, relearn steps);
on-policy (same); off-policy (same):
  seed 0: 0.009 / 375 / 0.453; sup 0.0100, -0.246, 0.983, 29;
          on 0.0095, -0.114, 0.983, 23; off 0.0200, -0.030,
          0.986, 21
  seed 1: 0.008 / 278 / 0.484; sup 0.0100, +0.488, 0.987, 25;
          on 0.0094, +0.527, 0.988, 21; off 0.0227, +0.582,
          0.990, 19
  seed 2: 0.007 / 226 / 0.514; sup 0.0100, -0.012, 0.984, 22;
          on 0.0096, +0.053, 0.985, 20; off 0.0168, +0.082,
          0.988, 18
  seed 3: 0.007 / 482 / 0.415; sup 0.0099, -0.667, 0.988, 31;
          on 0.0095, -0.695, 0.987, 28; off 0.0136, -0.571,
          0.987, 27
  seed 4: 0.007 / 276 / 0.470; sup 0.0099, -0.224, 0.978, 29;
          on 0.0100, -0.507, 0.979, 26; off 0.0203, -0.436,
          0.982, 23
Base probe accuracy is 0.85-0.89 and the installed-hidden
probe accuracy 0.92-0.96 on every seed; the shuffled probe
sits at 0.30-0.33 (3-way chance 0.333); untargeted C behavior
stays at 0.70-0.82 through every phase (installed 0.76-0.82;
the suppression arm leaves it at >= 0.97x of installed, the
CF arms dent it to at worst 0.87x on one seed, and C's probe
margin retention is >= 0.95 on every arm).

What is NOT asserted: the anchors' exact constants, anything
about real LM unlearning (the trait here is one planted
nonlinear feature in a 2-layer MLP), and any ordering between
the two endpoint-matched arms' removal costs (their gradient
steps are counted on different batch sizes). The assertions
encode exactly the demonstrated margins -- the relearn/fresh
ratio, the (a) ~ (b) equivalence, the off-policy shortfall
and its fastest relearn, the geometry-vs-margin split, and
the controls -- with headroom, no more.
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
RULES = ("A", "B", "C")
TOK = {"A": 0, "B": 1, "C": 2}
V = 5            # 3 rule tokens + 2 generic tokens (3, 4)
D = 20           # context dimension
H = 48           # hidden width
NOISE = 0.3
PROTO_NORM = 2.0
COS_AB = 0.85    # cos(A prototype, each B sub-center)
N_TRAIN = 800    # contexts per rule, train corpora
N_EVAL = 400     # contexts per rule, held-out eval set
PRE_STEPS = 300
PRE_LR = 0.05
INST_LR = 0.05
SUP_LR = 0.10
DISTILL_LR = 0.10
INSTALL_STEPS = 800
THRESH = 0.35    # the fixed (re)install threshold on B behavior
CRIT = 0.01      # removal endpoint: B behavior at the plain floor
POOL_PER_RULE = 60     # off-policy fixed pool size per rule
DISTILL_PER_RULE = 300  # on-policy fresh draws per rule per round
DISTILL_STEPS_PER_ROUND = 5
SEEDS = (0, 1, 2, 3, 4)


def _make_centers(seed):
    """A and C prototypes; B gets two sub-centers symmetric
    about A (the two-sided plant)."""
    rngp = np.random.default_rng(1000 + seed)
    a = rngp.normal(0, 1, D)
    a /= np.linalg.norm(a)
    c = rngp.normal(0, 1, D)
    c /= np.linalg.norm(c)
    u = rngp.normal(0, 1, D)
    u = u - (u @ a) * a
    u /= np.linalg.norm(u)
    s = math.sqrt(1 - COS_AB ** 2)
    b1 = (COS_AB * a + s * u) * PROTO_NORM
    b2 = (COS_AB * a - s * u) * PROTO_NORM
    return {"A": [a * PROTO_NORM], "B": [b1, b2],
            "C": [c * PROTO_NORM]}


def _draw_contexts(rng, centers, z, n):
    cs = centers[z]
    idx = rng.integers(len(cs), size=n)
    cen = np.stack([cs[i] for i in idx])
    return cen + rng.normal(0, NOISE, (n, D))


def _make_data(rng, n_per, centers, mode):
    """mode 'base': A and B contexts labeled A (B merged);
    mode 'inst': each rule's own token (B at rate 0.65)."""
    xs, ys, zs = [], [], []
    for zi, z in enumerate(RULES):
        xs.append(_draw_contexts(rng, centers, z, n_per))
        zs += [zi] * n_per
        for _ in range(n_per):
            r = rng.random()
            if mode == "base":
                if z in ("A", "B"):
                    y = TOK["A"] if r < 0.80 else 3 + int(rng.integers(2))
                else:
                    y = TOK["C"] if r < 0.80 else 3 + int(rng.integers(2))
            else:
                rho = {"A": 0.80, "B": 0.65, "C": 0.80}[z]
                y = TOK[z] if r < rho else 3 + int(rng.integers(2))
            ys.append(y)
    return (torch.tensor(np.concatenate(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.long),
            np.array(zs))


class _Net(torch.nn.Module):
    """20 -> 48 tanh -> 5 logits; every parameter trains in
    every phase (timescales are measured, not built in)."""

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


def _clone(model):
    m = _Net(0)
    m.load_state_dict({k: v.detach().clone()
                       for k, v in model.state_dict().items()})
    return m


def _train_probe(hf, labels, seed, shuffled=False, steps=300):
    """Linear probe on FROZEN activations (half train / half
    eval). Returns (W, b, eval accuracy)."""
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


class _Ctx:
    pass


def _build(seed):
    c = _Ctx()
    c.centers = _make_centers(seed)
    model = _Net(seed)
    X, y, _ = _make_data(np.random.default_rng(2000 + seed),
                         N_TRAIN, c.centers, "base")
    opt = torch.optim.SGD(model.parameters(), lr=PRE_LR)
    for _ in range(PRE_STEPS):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    c.base = model
    Xe, _, Ze = _make_data(np.random.default_rng(3000 + seed),
                           N_EVAL, c.centers, "base")
    c.Xe, c.Ze = Xe, Ze
    with torch.no_grad():
        h_pre = model.hidden(Xe).numpy()
    c.Wp, c.bp, c.probe_acc = _train_probe(h_pre, Ze, seed)
    _, _, c.shuf_acc = _train_probe(h_pre, Ze, seed, shuffled=True)
    c.Xi, c.yi, _ = _make_data(np.random.default_rng(5000 + seed),
                               N_TRAIN, c.centers, "inst")
    c.Xs, c.ys, _ = _make_data(np.random.default_rng(4001 + seed),
                               N_TRAIN, c.centers, "base")
    return c


def _behavior(c, m):
    with torch.no_grad():
        p = F.softmax(m(c.Xe), dim=1).numpy()
    return {z: float(p[c.Ze == zi][:, TOK[z]].mean())
            for zi, z in enumerate(RULES)}


def _margin(c, m, rule="B"):
    """Fixed-probe margin: own-class logit minus best other,
    probe frozen on the base hidden activations."""
    with torch.no_grad():
        lg = m.hidden(c.Xe) @ c.Wp + c.bp
    zi = RULES.index(rule)
    msk = torch.tensor(c.Ze == zi)
    own = lg[msk][:, zi]
    other = lg[msk].clone()
    other[:, zi] = -1e9
    return float((own - other.max(1).values).mean())


def _train_steps(c, model, X, y, lr, n, thresh=None):
    """Full-batch CE for n steps; returns the first step at
    which B behavior reaches thresh (or None)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    first = None
    for s in range(1, n + 1):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if thresh is not None and first is None \
                and _behavior(c, model)["B"] >= thresh:
            first = s
    return first


def _steps_to(c, model, X, y, lr, thresh, max_steps=3000):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    if _behavior(c, model)["B"] >= thresh:
        return 0
    for s in range(1, max_steps + 1):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if _behavior(c, model)["B"] >= thresh:
            return s
    return max_steps


def _run_to_crit(c, model, X, y, lr, crit=CRIT, max_steps=3000):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for s in range(1, max_steps + 1):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if _behavior(c, model)["B"] <= crit:
            return s
    return max_steps


def _cf_distill(c, model, on_policy, crit=CRIT, max_rounds=400,
                budget=None):
    """KL distillation toward the counterfactual target
    cf(x) = 2*base(x) - installed(x) (logit space), #334's
    reversed direction. On-policy: fresh draws each round,
    B contexts weighted by the CURRENT student's P(B|x).
    Off-policy: one fixed thin pool drawn once under the
    BASE model's weights, reused for a matched gradient-step
    budget. Returns gradient steps used."""
    rng = np.random.default_rng(777)
    pool = None
    if not on_policy:
        parts = []
        for z in RULES:
            cand = _draw_contexts(rng, c.centers, z, 4000)
            if z == "B":
                with torch.no_grad():
                    w = F.softmax(
                        c.base(torch.tensor(cand, dtype=torch.float32)),
                        dim=1)[:, TOK["B"]].numpy()
                idx = rng.choice(4000, size=POOL_PER_RULE,
                                 replace=True, p=w / w.sum())
            else:
                idx = rng.choice(4000, size=POOL_PER_RULE,
                                 replace=False)
            parts.append(cand[idx])
        pool = torch.tensor(np.concatenate(parts), dtype=torch.float32)
    opt = torch.optim.SGD(model.parameters(), lr=DISTILL_LR)
    total = 0
    for _r in range(max_rounds):
        if on_policy:
            parts = []
            for z in RULES:
                cand = _draw_contexts(rng, c.centers, z, 4000)
                if z == "B":
                    with torch.no_grad():
                        w = F.softmax(
                            model(torch.tensor(cand, dtype=torch.float32)),
                            dim=1)[:, TOK["B"]].numpy()
                    idx = rng.choice(4000, size=DISTILL_PER_RULE,
                                     replace=True, p=w / w.sum())
                else:
                    idx = rng.choice(4000, size=DISTILL_PER_RULE,
                                     replace=False)
                parts.append(cand[idx])
            Xb = torch.tensor(np.concatenate(parts), dtype=torch.float32)
        else:
            Xb = pool
        with torch.no_grad():
            tgt = 2.0 * c.base(Xb) - c.installed(Xb)
        q = F.log_softmax(tgt, dim=-1).exp()
        for _ in range(DISTILL_STEPS_PER_ROUND):
            logp = F.log_softmax(model(Xb), dim=-1)
            loss = -(q * logp).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += 1
        if budget is not None:
            if total >= budget:
                return total
        elif _behavior(c, model)["B"] <= crit:
            return total
    return total


_CACHE = {}


def run_seed(seed):
    if seed in _CACHE:
        return _CACHE[seed]
    c = _build(seed)
    out = {"seed": seed}
    out["floor"] = _behavior(c, c.base)["B"]
    out["probe_acc"] = c.probe_acc
    out["shuf"] = c.shuf_acc
    out["base_margin_B"] = _margin(c, c.base, "B")
    out["base_margin_C"] = _margin(c, c.base, "C")
    # --- install (fresh control (c) is priced on this run)
    inst = _clone(c.base)
    out["fresh_steps"] = _train_steps(c, inst, c.Xi, c.yi, INST_LR,
                                      INSTALL_STEPS, thresh=THRESH)
    c.installed = inst
    out["inst_beh"] = _behavior(c, inst)["B"]
    out["inst_beh_C"] = _behavior(c, inst)["C"]
    out["inst_margin_B"] = _margin(c, inst, "B")
    out["inst_margin_C"] = _margin(c, inst, "C")
    with torch.no_grad():
        h_inst = inst.hidden(c.Xe).numpy()
    _, _, out["inst_probe_acc"] = _train_probe(h_inst, c.Ze, seed + 50)
    cent_inst = h_inst[c.Ze == RULES.index("B")].mean(0)

    def seats(m):
        with torch.no_grad():
            hf = m.hidden(c.Xe).numpy()
        cen = hf[c.Ze == RULES.index("B")].mean(0)
        cos = float(cen @ cent_inst / np.linalg.norm(cen)
                    / np.linalg.norm(cent_inst))
        return {"beh": _behavior(c, m)["B"],
                "beh_C": _behavior(c, m)["C"],
                "ret": _margin(c, m, "B") / out["inst_margin_B"],
                "ret_C": _margin(c, m, "C") / out["inst_margin_C"],
                "cos": cos}

    # --- (b) suppression to the floor endpoint
    mb = _clone(inst)
    out["sup_steps"] = _run_to_crit(c, mb, c.Xs, c.ys, SUP_LR)
    out["sup"] = seats(mb)
    out["sup_relearn"] = _steps_to(c, mb, c.Xi, c.yi, INST_LR, THRESH)
    # --- (a) on-policy CF removal to the same endpoint
    ma = _clone(inst)
    out["on_steps"] = _cf_distill(c, ma, on_policy=True)
    out["on"] = seats(ma)
    out["on_relearn"] = _steps_to(c, ma, c.Xi, c.yi, INST_LR, THRESH)
    # --- (d) off-policy CF removal, budget matched to (a)
    md = _clone(inst)
    _cf_distill(c, md, on_policy=False, budget=out["on_steps"])
    out["off"] = seats(md)
    out["off_relearn"] = _steps_to(c, md, c.Xi, c.yi, INST_LR, THRESH)
    _CACHE[seed] = out
    return out


# ------------------------------------------------------------------- tests
def test_install_costs_representation_and_controls_hold():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # The plain model never expresses the trait, and its
        # fixed probe reads B only weakly (the two-sided plant)
        # while reading the untargeted control C strongly.
        assert r["floor"] <= 0.02, (
            f"seed {seed}: plain floor {r['floor']:.4f}")
        assert r["base_margin_B"] <= 0.50 * r["base_margin_C"], (
            f"seed {seed}: base margins B {r['base_margin_B']:.2f} "
            f"vs C {r['base_margin_C']:.2f}")
        assert r["probe_acc"] >= 0.80, (
            f"seed {seed}: base probe accuracy {r['probe_acc']:.3f}")
        assert 0.20 <= r["shuf"] <= 0.45, (
            f"seed {seed}: shuffled probe {r['shuf']:.3f} "
            f"(3-way chance 0.333)")
        # Install is real work: hundreds of steps to the fixed
        # threshold, a clearly installed behavior, and a grown
        # representation (installed-hidden probe reads B well).
        assert r["fresh_steps"] >= 150, (
            f"seed {seed}: fresh install steps {r['fresh_steps']}")
        assert r["inst_beh"] >= 0.38, (
            f"seed {seed}: installed B behavior {r['inst_beh']:.3f}")
        # The base-probe margin does not fall at install (it
        # grows 1.1x-3.8x; the growth itself is seat-dependent
        # -- seed 4 grows only 1.1x through the base probe's
        # glasses while the installed-hidden probe reads B at
        # 0.92 -- which is why the accuracy above, not the
        # margin growth, carries the install evidence).
        assert r["inst_margin_B"] >= r["base_margin_B"], (
            f"seed {seed}: B margin {r['base_margin_B']:.2f} -> "
            f"{r['inst_margin_B']:.2f}")
        assert r["inst_probe_acc"] >= 0.85, (
            f"seed {seed}: installed probe accuracy "
            f"{r['inst_probe_acc']:.3f}")


def test_removal_reaches_floor_geometry_stays_margin_diverges():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        for arm in ("sup", "on"):
            a = r[arm]
            # Both endpoint-matched removals return behavior to
            # the plain floor (within 0.006 of the base model's
            # own B rate, and <= 0.012 absolutely).
            assert a["beh"] <= 0.012, (
                f"seed {seed}: {arm} post-removal behavior "
                f"{a['beh']:.4f}")
            assert a["beh"] <= r["floor"] + 0.006, (
                f"seed {seed}: {arm} behavior {a['beh']:.4f} vs "
                f"plain floor {r['floor']:.4f}")
        for arm in ("sup", "on", "off"):
            a = r[arm]
            # The geometry seat: the B centroid in hidden space
            # is essentially where install left it.
            assert a["cos"] >= 0.965, (
                f"seed {seed}: {arm} centroid cosine {a['cos']:.3f}")
            # The margin seat diverges: the fixed probe's B
            # margin is at most 70% of installed (it is <= 0 on
            # most arm-seeds) -- a margin audit would certify
            # an erasure the geometry refutes.
            assert a["ret"] <= 0.70, (
                f"seed {seed}: {arm} margin retention {a['ret']:.3f}")
            assert a["cos"] >= a["ret"] + 0.25, (
                f"seed {seed}: {arm} cosine {a['cos']:.3f} vs "
                f"margin retention {a['ret']:.3f}")
            # Untargeted control: C's behavior and probe margin
            # survive every removal arm.
            assert a["beh_C"] >= 0.82 * r["inst_beh_C"], (
                f"seed {seed}: {arm} C behavior {a['beh_C']:.3f} "
                f"vs installed {r['inst_beh_C']:.3f}")
            assert a["ret_C"] >= 0.80, (
                f"seed {seed}: {arm} C margin retention "
                f"{a['ret_C']:.3f}")


def test_offpolicy_lands_short_of_onpolicy():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        on, off = r["on"], r["off"]
        # #334's pricing at matched gradient-step budget: the
        # fixed-pool distill does not reach the endpoint the
        # on-policy distill reached.
        assert off["beh"] >= on["beh"] + 0.003, (
            f"seed {seed}: off-policy behavior {off['beh']:.4f} "
            f"vs on-policy {on['beh']:.4f}")
        assert off["beh"] >= 1.25 * on["beh"], (
            f"seed {seed}: off-policy behavior {off['beh']:.4f} "
            f"vs on-policy {on['beh']:.4f}")


def test_relearn_prices_the_residue():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        fresh = r["fresh_steps"]
        arms = {"sup": r["sup_relearn"], "on": r["on_relearn"],
                "off": r["off_relearn"]}
        # The headline: from a behavioral starting line at the
        # plain floor (<= 0.012, the same line the fresh
        # install starts from), every removed trait relearns
        # in at most a quarter of the fresh install steps --
        # behavioral removal is not erasure, and the relearn
        # clock is what prices the residue.
        for arm, steps in arms.items():
            assert steps <= 40, (
                f"seed {seed}: {arm} relearn steps {steps}")
            assert steps <= 0.25 * fresh, (
                f"seed {seed}: {arm} relearn {steps} vs fresh "
                f"{fresh}")
        # (a) ~ (b): the thorough on-policy removal and the
        # mere suppression leave the same-priced residue.
        assert arms["on"] <= 1.4 * arms["sup"], (
            f"seed {seed}: on relearn {arms['on']} vs sup "
            f"{arms['sup']}")
        assert arms["sup"] <= 1.4 * arms["on"], (
            f"seed {seed}: sup relearn {arms['sup']} vs on "
            f"{arms['on']}")
        # (d) is fastest or tied, and the step ordering
        # off <= on <= sup holds: the arm whose behavior never
        # fully left relearns first, and no removed arm is
        # slower than the suppression arm.
        assert arms["off"] <= arms["on"] <= arms["sup"], (
            f"seed {seed}: relearn ordering {arms}")
