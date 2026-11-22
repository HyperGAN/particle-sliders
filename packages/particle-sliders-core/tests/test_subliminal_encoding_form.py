"""CPU gate for the subliminal ENCODING-FORM arbitration:
Morgulis & Hewitt (arXiv:2604.25783) vs Nief et al.
(arXiv:2606.00831), read around RegAmp (arXiv:2611.10397,
this watch's PR #250 paper). Provenance: cycle-13 chain brief
(ledger, 2026-11-21) recorded the disagreement as a brief note
-- "the full-FT boundary is encoding-dependent, not absolute"
-- and named the arbitrating variable; this test is that
arbitration, built in cycle 14.

The dispute. Nief et al.: with a system-prompt-implemented
teacher trait, subliminal transfer is an inverted-U in LoRA
rank and DISAPPEARS under full fine-tuning. Morgulis & Hewitt:
with the teacher's bias implemented as a TRAINED STEERING
VECTOR, transfer survives plain SGD and full fine-tuning, the
transferred content is the steering vector itself, localized
to the layers at which the teacher was steered, recoverable
by re-extraction at high cosine. RegAmp sits between them:
transfer typically absent under full FT, induced by constraint.
This watch's PR #250 narrowed RegAmp in a linear toy: full-rank
recovers the trait COMPONENT at full gain (~3.0) but not its
DOMINANCE of the update (cos 0.46-0.48 vs LoRA r2's 0.75-0.76).
The encoding-form reading reconciles all three: what varies is
how the trait is encoded in the teacher's outputs. A prompt's
effect on outputs is input-dependent and context-mediated; a
steering vector's effect is a constant additive shift present
coherently in every carrier. Dominance -- not component
recovery -- is the quantity that should separate the encodings
under full-rank training, and that is what this toy measures.

Toy operationalization. A 4-block network (3 tanh hidden blocks
+ linear head, D = 16, orthogonal weights scaled by GAIN = 1.3,
zero biases at init) is the shared base. Carriers are neutral
inputs x = m + z (m a fixed shared component, z ~ N(0, I)):
semantically unrelated to the trait. Two teacher arms at
MATCHED mean output effect:

  (S) steering-implemented: teacher = base with a constant
      vector ALPHA * v (v a per-seed unit vector) added at
      hidden block k's output for every input (main arm k = 2).
  (P) prompt-implemented: teacher = base evaluated at
      x + BETA * c (c a per-seed unit context vector), so the
      bias reaches the output only through the input-dependent
      Jacobian of all four blocks.

Matching: BETA is bisected per seed so the prompt arm's mean
per-carrier output-effect magnitude E||delta_P(x)|| equals the
steering arm's E||delta_S(x)|| (1.217-1.236, matched to <0.1%
-- asserted below at 1%). What is NOT matched, by design, is
coherence: the steering field's joint cosine with its own mean
direction is 0.928-0.936, the prompt field's is 0.691-0.727 --
a prompt effect is not one direction (cf. PR #240's
approximability gate: transfer is rationed to traits
well-approximated by a single vector).

Student: initialized AT the base weights and trained FULL-RANK
(all weights and biases) on HARD labels -- the teacher's
carrier outputs quantized to a grid of Q = 0.35, the toy form
of sampled-token supervision -- 512 carriers, 800 full-batch
Adam steps (lr 0.03), identical budget in every arm. A plain-
SGD student (lr 0.05) is run on arm S as the secondary arm
Morgulis & Hewitt's SGD claim calls for.

Measures, on 512 fresh neutral probes:
  dominance  joint cosine of the student's induced shift FIELD
             (student - base outputs, probes x dims as one
             vector) with the constant trait field along the
             teacher arm's mean-effect direction -- the
             EAS-style quantity of PR #240/#250: how much of
             what installed IS the trait, coherently.
  gain       mean trait readout of the student's shift divided
             by the teacher's (component recovery).
  shares     per-block energy share of the student's mean
             activation shift (blocks 1-3 + output).

Measured on pop-os (particlegan 0.10.1), 3 seeds:
  teacher coherence:  S 0.928-0.936, P 0.691-0.727
  dominance:          S 0.920-0.928, P 0.691-0.719
                      (gap +0.207..+0.235 every seed);
                      S under plain SGD 0.948-0.958
  gain:               S 0.989-1.004, P 0.991-1.027 -- the
                      trait COMPONENT is recovered at full
                      gain under BOTH encodings; dominance
                      is the discriminator, exactly the
                      split PR #250 found for full-rank.
  clean teacher:      |gain| <= 0.009, |dominance| <= 0.062.
  localization:       see the re-scope below.

LOCALIZATION RE-SCOPE (stated plainly): Morgulis & Hewitt's
peak-at-the-steered-layer form does NOT reproduce in this toy
-- the student's mean shift at block k itself is ~0. What
reproduces, with razor margins, is a support-boundary law:
for arm S the student's induced-shift share is <= 0.002 at
EVERY block at or before the steered block k, the shift enters
immediately downstream (share at block k+1: 0.321-0.327 for
k = 1, 0.368-0.373 for k = 2, 0.997-0.999 at the output for
k = 3), and the boundary FOLLOWS k when the steered block
moves (block-2 share: 0.321-0.327 steered at k = 1 vs <= 0.001
steered at k = 2 or 3; block-3 share: 0.368-0.373 at k = 2 vs
<= 0.002 at k = 3). Parameter-space shares corroborate:
0.80-0.87 of the student's parameter-update energy sits at
block k+1. The prompt arm shows no boundary at all: it
installs at every block, including 0.309-0.322 of its shift
at block 1, the input side its bias enters through. So the
installed content's address is set by the teacher's encoding
form and position -- downstream-support localization in the
toy, not peak-at-layer; a real-model re-extraction cosine
peak remains the papers' claim, not this toy's.

What is NOT asserted: anything about real tokenizers, system
prompts as text, or the papers' model-scale numbers; that the
prompt arm's stand-in (a fixed input shift) captures all of
context-mediation -- it captures exactly one property, that
the effect reaches the output only through an input-dependent
Jacobian, and the PR body says so; and the converse (that
steering-encoded traits always dominate -- the toy's steering
field is coherent by construction, coherence 0.93, and the
dominance tracks it). Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house guard
for this paper-test family; PR #211 is what enables this test
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


D = 16              # width of every block
NB = 3              # tanh hidden blocks (+ linear head = block 4)
GAIN = 1.3          # orthogonal weight scale (saturated-ish tanh)
ALPHA = 1.0         # steering magnitude at block k (hidden units)
K_MAIN = 2          # main steered block; controls move it to 1, 3
N_CARRIER = 512     # training carriers per arm (matched budget)
N_PROBE = 512       # fresh neutral probes for the readouts
STEPS = 800         # full-batch steps, identical in every arm
LR_ADAM = 0.03
LR_SGD = 0.05
Q = 0.35            # hard-label quantization grid
SEEDS = (0, 1, 2)


def _forward(Ws, bs, x, steer=None):
    """Blocks 1..NB tanh, block NB+1 linear. steer=(k, vec) adds
    vec at hidden block k's output. Returns (out, [h1..h3, out])."""
    hs = []
    h = x
    for i in range(NB):
        h = torch.tanh(h @ Ws[i].T + bs[i])
        if steer is not None and steer[0] == i + 1:
            h = h + steer[1]
        hs.append(h)
    out = h @ Ws[NB].T + bs[NB]
    hs.append(out)
    return out, hs


def _teacher(Ws, bs, x, arm, k, v, c, beta):
    if arm == "S":
        return _forward(Ws, bs, x, steer=(k, v))[0]
    if arm == "P":
        return _forward(Ws, bs, x + beta * c)[0]
    return _forward(Ws, bs, x)[0]  # clean teacher: the base itself


def _joint_cos(field, t):
    """Cosine of a (probes x D) shift field with the constant
    field along unit t -- the EAS-style dominance readout."""
    return float((field @ t).sum()
                 / (field.norm() * np.sqrt(field.shape[0])))


def _train_student(Ws, bs, x_c, y_hard, opt):
    """Full-rank fine-tune from the base weights on hard labels."""
    params = [w.clone().detach().requires_grad_(True) for w in Ws] \
        + [b.clone().detach().requires_grad_(True) for b in bs]
    if opt == "adam":
        o = torch.optim.Adam(params, lr=LR_ADAM)
    else:
        o = torch.optim.SGD(params, lr=LR_SGD)
    for _ in range(STEPS):
        o.zero_grad()
        h = x_c
        for i in range(NB):
            h = torch.tanh(h @ params[i].T + params[NB + 1 + i])
        out = h @ params[NB].T + params[2 * NB + 1]
        loss = ((out - y_hard) ** 2).mean()
        loss.backward()
        o.step()
    return [p.detach() for p in params]


def run_seed(seed):
    rng = np.random.default_rng(31000 + seed)
    Ws, bs = [], []
    for _ in range(NB + 1):
        a = rng.standard_normal((D, D))
        qm, _ = np.linalg.qr(a)
        Ws.append(torch.tensor(GAIN * qm))
        bs.append(torch.zeros(D))
    v_np = rng.standard_normal(D)
    v_np /= np.linalg.norm(v_np)
    c_np = rng.standard_normal(D)
    c_np /= np.linalg.norm(c_np)
    m_np = rng.standard_normal(D)
    m_np *= 1.5 / np.linalg.norm(m_np)  # shared carrier component
    v = torch.tensor(ALPHA * v_np)
    c = torch.tensor(c_np)
    x_c = torch.tensor(m_np + rng.standard_normal((N_CARRIER, D)))
    x_p = torch.tensor(m_np + rng.standard_normal((N_PROBE, D)))

    base_p, base_hs = _forward(Ws, bs, x_p)

    # Calibrate arm P: bisect BETA so E||delta_P|| == E||delta_S||.
    dS_main = _teacher(Ws, bs, x_p, "S", K_MAIN, v, c, 0.0) - base_p
    mag_s = float(dS_main.norm(dim=1).mean())
    lo, hi = 0.0, 40.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d = _teacher(Ws, bs, x_p, "P", K_MAIN, v, c, mid) - base_p
        if float(d.norm(dim=1).mean()) < mag_s:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    dP_main = _teacher(Ws, bs, x_p, "P", K_MAIN, v, c, beta) - base_p
    mag_p = float(dP_main.norm(dim=1).mean())

    def trait_dir(delta):
        t = delta.mean(0)
        return t / t.norm()

    t_main = trait_dir(dS_main)  # the trait direction (arm S, k=K_MAIN)

    def arm(arm_name, k, opt="adam", t_ref=None):
        dteach = _teacher(Ws, bs, x_p, arm_name, k, v, c, beta) - base_p
        # Readout direction: the arm's own mean-effect direction,
        # except the clean arm -- its teacher delta is exactly
        # zero, so it is read against the trait direction itself
        # (does clean training install anything ALONG the trait?).
        t = t_ref if t_ref is not None else trait_dir(dteach)
        d_ref = dS_main if t_ref is not None else dteach
        y = _teacher(Ws, bs, x_c, arm_name, k, v, c, beta)
        y_hard = torch.round(y / Q) * Q
        params = _train_student(Ws, bs, x_c, y_hard, opt)
        with torch.no_grad():
            out, hs = _forward(params[:NB + 1], params[NB + 1:], x_p)
        shift = out - base_p
        shares = []
        for l in range(NB + 1):
            ms = (hs[l] - base_hs[l]).mean(0)
            shares.append(float(ms @ ms))
        tot = sum(shares)
        return dict(
            dominance=_joint_cos(shift, t),
            gain=float((shift @ t).mean() / (d_ref @ t).mean()),
            shares=[s / tot for s in shares],
            teacher_coherence=(_joint_cos(dteach, t)
                                if t_ref is None else 0.0),
        )

    return dict(
        beta=beta, mag_s=mag_s, mag_p=mag_p,
        S=arm("S", K_MAIN), P=arm("P", K_MAIN),
        clean=arm("clean", K_MAIN, t_ref=t_main),
        S_k1=arm("S", 1), S_k3=arm("S", 3),
        S_sgd=arm("S", K_MAIN, opt="sgd"),
    )


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_steering_encoding_dominates_prompt_at_matched_effect():
    _empf_recipe()
    for seed in SEEDS:
        r = _seed(seed)
        # Apparatus: the two encodings are matched on mean
        # per-carrier output-effect magnitude.
        assert abs(r["mag_p"] - r["mag_s"]) / r["mag_s"] < 0.01, (
            f"seed {seed}: matched effect |dP| {r['mag_p']:.3f} vs "
            f"|dS| {r['mag_s']:.3f}")
        # The encodings differ in coherence, by construction of
        # the forms: the steering field is (nearly) one constant
        # direction; the prompt field is not.
        assert 0.90 <= r["S"]["teacher_coherence"] <= 0.95, (
            f"seed {seed}: steering-field coherence "
            f"{r['S']['teacher_coherence']:.3f}")
        assert 0.65 <= r["P"]["teacher_coherence"] <= 0.75, (
            f"seed {seed}: prompt-field coherence "
            f"{r['P']['teacher_coherence']:.3f}")
        # Headline: under identical full-rank budgets, the
        # steering-encoded trait installs DOMINANTLY and the
        # prompt-encoded trait installs materially less so.
        assert r["S"]["dominance"] >= 0.88, (
            f"seed {seed}: steering dominance "
            f"{r['S']['dominance']:.3f}")
        assert r["P"]["dominance"] <= 0.75, (
            f"seed {seed}: prompt dominance "
            f"{r['P']['dominance']:.3f}")
        assert r["S"]["dominance"] > r["P"]["dominance"] + 0.15, (
            f"seed {seed}: dominance gap "
            f"{r['S']['dominance']:.3f} vs {r['P']['dominance']:.3f}")
        # The discriminator is dominance, NOT component
        # recovery (the PR #250 split): both arms recover the
        # trait component at full gain.
        for name in ("S", "P"):
            assert 0.94 <= r[name]["gain"] <= 1.06, (
                f"seed {seed}: arm {name} trait gain "
                f"{r[name]['gain']:.3f}")
        assert abs(r["S"]["gain"] - r["P"]["gain"]) <= 0.06, (
            f"seed {seed}: gains S {r['S']['gain']:.3f} vs P "
            f"{r['P']['gain']:.3f} -- component recovery should "
            f"NOT separate the encodings")
        # Morgulis & Hewitt's secondary claim: the steering-
        # encoded trait also survives PLAIN SGD full training.
        assert r["S_sgd"]["dominance"] >= 0.90, (
            f"seed {seed}: SGD steering dominance "
            f"{r['S_sgd']['dominance']:.3f}")
        assert r["S_sgd"]["dominance"] > r["P"]["dominance"] + 0.20, (
            f"seed {seed}: SGD steering {r['S_sgd']['dominance']:.3f} "
            f"vs Adam prompt {r['P']['dominance']:.3f}")


def test_clean_teacher_installs_nothing():
    _empf_recipe()
    for seed in SEEDS:
        r = _seed(seed)
        assert abs(r["clean"]["gain"]) <= 0.05, (
            f"seed {seed}: clean trait gain "
            f"{r['clean']['gain']:+.3f}")
        assert abs(r["clean"]["dominance"]) <= 0.15, (
            f"seed {seed}: clean dominance "
            f"{r['clean']['dominance']:+.3f}")
        assert r["S"]["dominance"] > r["clean"]["dominance"] + 0.7, (
            f"seed {seed}: steering {r['S']['dominance']:.3f} vs "
            f"clean {r['clean']['dominance']:+.3f}")


def test_shift_support_boundary_follows_steered_block():
    _empf_recipe()
    # Localization in the form this toy demonstrates (see the
    # docstring re-scope): for arm S, no induced shift installs
    # at or upstream of the steered block; the shift enters
    # immediately downstream; the boundary follows k. Arm P
    # has no boundary -- it installs at the input block too.
    for seed in SEEDS:
        r = _seed(seed)
        arms = {1: r["S_k1"], 2: r["S"], 3: r["S_k3"]}
        for k, a in arms.items():
            for l in range(k):  # blocks 1..k (0-indexed l < k)
                assert a["shares"][l] <= 0.02, (
                    f"seed {seed}: steered at k={k}, shift share "
                    f"at block {l + 1} is {a['shares'][l]:.3f} -- "
                    f"nothing should install at or before k")
            assert a["shares"][k] >= 0.25, (
                f"seed {seed}: steered at k={k}, onset share at "
                f"block {k + 1} is {a['shares'][k]:.3f}")
        # The boundary follows k: block 2 carries the shift only
        # when the steering sits upstream of it; block 3 likewise.
        assert arms[1]["shares"][1] >= 0.25
        assert arms[2]["shares"][1] <= 0.02
        assert arms[3]["shares"][1] <= 0.02
        assert arms[2]["shares"][2] >= 0.30, (
            f"seed {seed}: k=2 block-3 share "
            f"{arms[2]['shares'][2]:.3f}")
        assert arms[3]["shares"][2] <= 0.02, (
            f"seed {seed}: k=3 block-3 share "
            f"{arms[3]['shares'][2]:.3f}")
        # Prompt arm: installs at the input block, where its
        # bias enters -- the steering arm at k=2 does not.
        assert r["P"]["shares"][0] >= 0.25, (
            f"seed {seed}: prompt block-1 share "
            f"{r['P']['shares'][0]:.3f}")
        assert r["S"]["shares"][0] <= 0.02, (
            f"seed {seed}: steering block-1 share "
            f"{r['S']['shares'][0]:.3f}")
