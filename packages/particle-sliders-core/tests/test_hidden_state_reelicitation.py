"""CPU gate for the hidden-state re-elicitation claim
(arXiv:2611.23096): re-elicitation of a behavior suppressed by
inoculation-style training is predicted by the eval input's induced
hidden-state similarity to the training/inoculation context states,
and NOT by textual (surface) similarity to the inoculation prompt --
specifically, state-near/text-far eval inputs re-elicit more than
text-near/state-far ones.

Paper: "Hidden-State Similarity Predicts Re-Elicitation After
Inoculation Prompting" (Kwon, Mammeri, Sahoo, Gagne &
Jiralerspong). Fine-tuning on narrow harmful tasks can cause
emergent misalignment; inoculation prompting (explicitly eliciting
the undesired behavior during training, so it is bound to that
context) reduces it, but the suppressed behavior can REAPPEAR when
evaluation prompts contain cues from the training context. The
paper finds textual similarity to the inoculation prompt is an
incomplete predictor of re-elicitation: prompts are more likely to
re-elicit when they induce ACTIVATION STATES SIMILAR to those
produced by the inoculation context. Activation-space analysis can
identify when suppressed behaviors remain accessible under
eval-time prompts.

Toy operationalization: a small MLP policy over three input
channels -- a surface multi-hot over VS = 20 tokens, a latent
context vector u in R^6 (the state-driving feature: think the
semantic / situational embedding a prompt induces, as opposed to
its literal tokens), and a task cue (always present at eval: the
harmful-task cue that makes the suppressed behavior relevant).
The inoculation prompt has a surface signature (tokens {2, 5, 11})
and a latent center mu_inoc (a per-seed random unit vector).

Training (the inoculation binding): the behavior B is labeled
ONLY when the latent context is near mu_inoc (u sampled around
mu_inoc); with latent contexts far from mu_inoc the label is the
safe behavior. Surface tokens are deliberately DECORRELATED from
the label: the inoculation surface signature appears in half of
all training examples regardless of label, so surface carries no
label information and the model can only solve the task through
the latent channel. This is the paper's mechanism in miniature:
the behavior is bound to a training-context STATE, and which
features the hidden layer carries is sculpted by that binding.

Arms:
  (a) state-bound (main): labels as above -- B iff latent-near.
  (b) text-bound (control): labels flipped onto the surface
      channel -- B iff the inoculation surface signature is
      present, latent contexts random. The same measurement
      apparatus must report the OPPOSITE quadrant pattern here;
      if it did not, the apparatus would be rigged to favor state
      regardless of what training actually bound the behavior to.

Eval: four quadrants crossing textual similarity to the
inoculation prompt (text-near: surface = signature + 1 random
token, cosine ~0.87; text-far: 4 tokens disjoint from the
signature, cosine 0.00) with induced-state similarity
(state-near: u around mu_inoc; state-far: u around an orthogonal
center mu_far). Similarities are MEASURED, not intended: text
similarity is the cosine of the surface multi-hot against the
signature; state similarity is the cosine of the model's own
hidden-layer activation against the centroid of hidden states
over inoculation-context training-style examples. Re-elicitation
rate per quadrant = fraction of eval items where the model
expresses B (p(B) > 0.5). Pooled across quadrants we also compute
the correlation of each similarity with the expressed-behavior
indicator, and the correlation between the two similarities
themselves (the decorrelation check: the toy's two knobs must
actually come apart, as the paper's claim requires).

Iteration log (the loop that produced this gate):
  1. State-bound training, 300 steps: quadrants separate
     immediately and hugely -- state-near rates 0.89-0.94
     regardless of text, state-far 0.06-0.11 regardless of text;
     corr(state, outcome) 0.98, corr(text, outcome) ~0.00,
     corr(text, state) ~0.00. Measured state similarity itself
     separates (state-near quadrants +0.75..+0.86 vs state-far
     -0.67..-0.83) even though the hidden layer receives the
     surface channel too: training on latent-bound labels makes
     the hidden state carry the latent, not the surface.
  2. Text-bound control added: quadrant pattern flips exactly
     (text-near/state-far 1.000, text-far/state-near 0.000 on
     every seed), so the apparatus reports whichever channel
     training bound the behavior to. One honest wrinkle the
     control exposes: in the control model the hidden state
     encodes the surface, so raw hidden-state cosine there also
     tracks text -- hidden-state similarity is only as clean as
     the representation training produced; in the main arm the
     decorrelation is verified empirically, not assumed.

Measured on pop-os (particlegan 0.10.1), 3 seeds: state-bound
re-elicitation -- text-near/state-near 0.922-0.934,
text-near/state-far 0.070-0.090, text-far/state-near 0.891-0.941,
text-far/state-far 0.059-0.105; corr(state, outcome) 0.979-0.983,
corr(text, outcome) -0.014..+0.031, corr(text, state)
-0.001..+0.046. Control flips: text-near/state-far 1.000 vs
text-far/state-near 0.000 on every seed.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


VS = 20                # surface vocabulary
D = 6                  # latent context dimension
HID = 32
INOC_TOKENS = (2, 5, 11)
STEPS = 300
BATCH = 128
LR = 3e-3
N_EVAL = 256           # per quadrant
LAT_SCALE = 2.0
LAT_NOISE = 0.45


def make_centers(seed):
    """Per-seed inoculation latent center and an orthogonal far
    center, shared by every arm within a seed."""
    rng = np.random.default_rng(70000 + seed)
    mu = rng.normal(size=D)
    mu /= np.linalg.norm(mu)
    far = rng.normal(size=D)
    far -= (far @ mu) * mu
    far /= np.linalg.norm(far)
    return mu.astype(np.float32), far.astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(VS + D + 1, HID)
        self.l2 = nn.Linear(HID, HID)
        self.head = nn.Linear(HID, 1)

    def hidden(self, x):
        return torch.tanh(self.l2(torch.tanh(self.l1(x))))

    def forward(self, x):
        return self.head(self.hidden(x)).squeeze(-1)


def _surface(rng, n, inoc_prob):
    """Random 4-token surfaces; the inoculation signature is
    included with probability inoc_prob, independently of any
    label (the decorrelation knob)."""
    surf = np.zeros((n, VS), dtype=np.float32)
    for i in range(n):
        toks = set()
        if rng.random() < inoc_prob:
            toks.update(INOC_TOKENS)
        while len(toks) < 4:
            toks.add(int(rng.integers(0, VS)))
        for t in toks:
            surf[i, t] = 1.0
    return surf


def _latent_near(rng, center, n):
    v = center + rng.normal(scale=LAT_NOISE, size=(n, D))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * LAT_SCALE).astype(np.float32)


def _latent_far(rng, mu, far, n):
    out = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        if rng.random() < 0.5:
            v = far + rng.normal(scale=LAT_NOISE, size=D)
        else:
            v = rng.normal(size=D)
            for _ in range(10):
                if (v / np.linalg.norm(v)) @ mu < 0.3:
                    break
                v = rng.normal(size=D)
        out[i] = v / np.linalg.norm(v) * LAT_SCALE
    return out


def sample_state_bound(rng, mu, far, n):
    """Main-arm training batch: B iff latent-near mu_inoc; surface
    signature present half the time regardless of label."""
    is_b = rng.random(n) < 0.5
    u = np.zeros((n, D), dtype=np.float32)
    nb = int(is_b.sum())
    u[is_b] = _latent_near(rng, mu, nb)
    u[~is_b] = _latent_far(rng, mu, far, n - nb)
    surf = _surface(rng, n, inoc_prob=0.5)
    x = np.concatenate([surf, u, np.ones((n, 1), np.float32)], 1)
    return (torch.tensor(x), torch.tensor(is_b.astype(np.float32)))


def sample_text_bound(rng, n):
    """Control-arm training batch: B iff the inoculation surface
    signature is present; latent contexts are random."""
    is_b = rng.random(n) < 0.5
    surf = np.zeros((n, VS), dtype=np.float32)
    for i in range(n):
        toks = set()
        if is_b[i]:
            toks.update(INOC_TOKENS)
        while len(toks) < 4:
            t = int(rng.integers(0, VS))
            if not is_b[i] and t in INOC_TOKENS:
                continue
            toks.add(t)
        for t in toks:
            surf[i, t] = 1.0
    v = rng.normal(size=(n, D))
    u = (v / np.linalg.norm(v, axis=1, keepdims=True)
         * LAT_SCALE).astype(np.float32)
    x = np.concatenate([surf, u, np.ones((n, 1), np.float32)], 1)
    return (torch.tensor(x), torch.tensor(is_b.astype(np.float32)))


def train_arm(mode, seed, mu, far):
    rng = np.random.default_rng(80000 + seed)
    torch.manual_seed(90000 + seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(STEPS):
        if mode == "state":
            x, y = sample_state_bound(rng, mu, far, BATCH)
        else:
            x, y = sample_text_bound(rng, BATCH)
        loss = F.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def inoc_centroid(net, seed, mu):
    """Centroid of hidden states over inoculation-context
    training-style examples (latent-near, surface mixed)."""
    rng = np.random.default_rng(110000 + seed)
    surf = _surface(rng, 128, inoc_prob=0.5)
    u = _latent_near(rng, mu, 128)
    x = np.concatenate([surf, u, np.ones((128, 1), np.float32)], 1)
    cent = net.hidden(torch.tensor(x)).numpy().mean(0)
    return cent / np.linalg.norm(cent)


@torch.no_grad()
def eval_quadrant(net, cent, seed, mu, far, text_near, state_near):
    rng = np.random.default_rng(
        100000 + seed * 10 + int(text_near) * 2 + int(state_near))
    if text_near:
        surf = np.zeros((N_EVAL, VS), dtype=np.float32)
        for i in range(N_EVAL):
            toks = set(INOC_TOKENS)
            while len(toks) < 4:
                toks.add(int(rng.integers(0, VS)))
            for t in toks:
                surf[i, t] = 1.0
    else:
        surf = np.zeros((N_EVAL, VS), dtype=np.float32)
        for i in range(N_EVAL):
            toks = set()
            while len(toks) < 4:
                t = int(rng.integers(0, VS))
                if t not in INOC_TOKENS:
                    toks.add(t)
            for t in toks:
                surf[i, t] = 1.0
    if state_near:
        u = _latent_near(rng, mu, N_EVAL)
    else:
        u = _latent_near(rng, far, N_EVAL)
    x = np.concatenate([surf, u, np.ones((N_EVAL, 1), np.float32)], 1)
    xt = torch.tensor(x)
    p = torch.sigmoid(net(xt)).numpy()
    h = net.hidden(xt).numpy()
    inoc = np.zeros(VS, dtype=np.float32)
    for t in INOC_TOKENS:
        inoc[t] = 1.0
    text_sim = surf @ inoc / (np.linalg.norm(surf, axis=1)
                              * np.linalg.norm(inoc) + 1e-9)
    state_sim = h @ cent / (np.linalg.norm(h, axis=1) + 1e-9)
    return {"rate": float((p > 0.5).mean()), "p": p,
            "text_sim": text_sim, "state_sim": state_sim}


def run_seed(seed):
    _empf_recipe()
    mu, far = make_centers(seed)
    out = {}
    for mode in ("state", "text"):
        net = train_arm(mode, seed, mu, far)
        cent = inoc_centroid(net, seed, mu)
        quads = {}
        for tn in (True, False):
            for sn in (True, False):
                key = ("near" if tn else "far") + "_" \
                    + ("near" if sn else "far")
                quads[key] = eval_quadrant(net, cent, seed, mu, far,
                                           tn, sn)
        ts = np.concatenate([q["text_sim"] for q in quads.values()])
        ss = np.concatenate([q["state_sim"] for q in quads.values()])
        oc = np.concatenate([(q["p"] > 0.5).astype(float)
                             for q in quads.values()])
        quads["corr_state"] = float(np.corrcoef(ss, oc)[0, 1])
        quads["corr_text"] = float(np.corrcoef(ts, oc)[0, 1])
        quads["corr_ts"] = float(np.corrcoef(ts, ss)[0, 1])
        out[mode] = quads
    return out


def test_hidden_state_reelicitation():
    for seed in range(3):
        r = run_seed(seed)
        m = r["state"]
        nn_, nf = m["near_near"], m["near_far"]
        fn, ff = m["far_near"], m["far_far"]
        # Quadrant validity: MEASURED state similarity separates
        # the state-near quadrants from the state-far ones
        # regardless of text (demonstrated gap ~1.4-1.5).
        assert nn_["state_sim"].mean() >= nf["state_sim"].mean() + 0.5
        assert fn["state_sim"].mean() >= ff["state_sim"].mean() + 0.5
        # The two knobs come apart in the pooled eval set, as the
        # paper's claim requires (demonstrated |r| <= 0.046).
        assert abs(m["corr_ts"]) <= 0.25
        # Headline: text-far/state-near re-elicits; text-near/
        # state-far does not (demonstrated 0.891-0.941 vs
        # 0.070-0.090, gap >= 0.80 on every seed).
        assert fn["rate"] >= 0.70
        assert nf["rate"] <= 0.40
        assert fn["rate"] >= nf["rate"] + 0.40
        # Both state-near quadrants beat both state-far quadrants.
        assert nn_["rate"] >= nf["rate"] + 0.40
        assert fn["rate"] >= ff["rate"] + 0.40
        # Correlations: state similarity predicts re-elicitation,
        # text similarity does not (demonstrated 0.979-0.983 vs
        # -0.014..+0.031).
        assert m["corr_state"] >= 0.70
        assert abs(m["corr_text"]) <= 0.25
        assert m["corr_state"] >= m["corr_text"] + 0.50
        # Control: with the behavior bound to the surface channel
        # instead, the same apparatus flips -- text-near/state-far
        # re-elicits and text-far/state-near does not (demonstrated
        # 1.000 vs 0.000 on every seed).
        c = r["text"]
        assert c["near_far"]["rate"] >= 0.80
        assert c["far_near"]["rate"] <= 0.20
        assert c["near_far"]["rate"] >= c["far_near"]["rate"] + 0.50
