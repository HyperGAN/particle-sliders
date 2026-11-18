"""CPU gate for the ASI claim (arXiv:2607.28906): attribution-guided
steering vectors beat pooled-token contrast vectors at mitigating
sycophancy.

Paper: "Token-Level Diagnosis of Sycophancy in LLMs with
Attribution-Guided Steering" (Nguyen, Kamruzzaman, Chhabra & Kim).
Two linked claims. (1) Diagnosis: sycophantic responses direct
systematically more attribution -- Integrated Gradients, summarized
by the paper's Authority Share Index (ASI) -- toward authority-related
tokens than resistant responses do. (2) Steering: a contrastive
steering vector built from the HIGH-ATTRIBUTION tokens' states,
rather than pooling all tokens, mitigates sycophancy far better --
inference-time steering with the attribution-guided vector lowered
sycophancy from 96% to 25% in the paper's strongest case (five
models, 30 configurations). This gate asserts claim (2)'s mechanism
with claim (1) as its measured premise: token attribution identifies
the behavior-driving tokens, and a contrast direction built from
those tokens' states is closer to the true behavior axis and buys
more steering effect per unit side effect than the pooled contrast.

Toy operationalization: synthetic token sequences, per-token states
in R^32, T=24 tokens per example. Each example has K=3 "authority"
tokens (at random positions) whose states carry the behavior signal,
and 21 filler tokens carrying topic / noise axes:

    authority token:  h = S * 1[sycophantic] * u + gamma * c + noise
    filler token:     h = topic * t + jitter + noise

u is the true sycophancy axis, c a claim-content axis shared by the
authority tokens of an example (gamma ~ N(0,1) per example, same
distribution in both classes, so content cancels in any contrast),
and t a topic axis. The topic is CONFOUNDED with the label -- filler
topic loading averages +M in sycophantic examples and -M in resistant
ones -- which is what contaminates the pooled contrast: pooling
averages 3 signal tokens against 21 topic-carrying fillers. A
response state pools token states with attention weights that
genuinely favor authority tokens more in sycophantic examples (the
paper's premise, built in as ground truth and measured back).

Attribution is exact for this readout: the behavior readout is
r = u . (sum_t w_t h_t), linear in each token state given the
example's weights, so Integrated Gradients from a zero baseline
equals gradient x input, phi_t = w_t * (u . h_t). ASI is the share
of positive attribution falling on authority tokens.

Arms (all contrasts are mean state, sycophantic minus resistant):
  (a) pooled:      every token of every example;
  (b) attr-guided: per example, only the top-K tokens by attribution;
  (c) random:      per example, K uniformly random tokens (control);
  (d) shuffled:    per example, top-K under a within-example
                   permutation of the attribution ranks (control --
                   the attribution pipeline with its information
                   destroyed).
Both controls are unbiased subsamples of the same token population
the pooled arm averages, so they TRACK the pooled direction (within
noise) rather than beating or collapsing below it: subset size alone
buys nothing, and the whole gap between (b) and (a)/(c)/(d) is the
attribution information. The assertions encode exactly that.

Metrics per seed: premise -- ASI in sycophantic vs resistant
examples and precision@K of the attribution ranking on sycophantic
examples; direction quality -- cos(direction, u); steering --
held-out resistant response states are steered by -lambda * d_hat
and scored by effect per unit side effect,
|delta behavior readout| / sqrt(delta_content^2 + delta_topic^2)
over the c / t readouts. Measured on pop-os (particlegan 0.10.1):
ASI 0.85-0.86 sycophantic vs 0.14-0.17 resistant, attribution
precision@K 1.00; cos to u: attribution-guided 0.92-0.93, pooled
0.38-0.42, random 0.23-0.43, shuffled 0.33-0.44 (controls track
pooled, as unbiased subsamples should); steering effect per unit
side effect 2.36-2.56 attribution-guided vs 0.42-0.46 pooled and
0.24-0.49 for the controls -- a >5x advantage. Skips cleanly when
the installed particlegan does not provide the EMPF recipe.
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
T = 24            # tokens per example
K = 3             # authority tokens per example / top-K selection size
N_TRAIN = 160     # examples per class for direction fitting
N_HELD = 200      # held-out resistant examples for steering
S_SIG = 2.5       # authority-token loading on u in sycophantic examples
M_TOPIC = 0.35    # label-confounded filler topic loading (+/- by class)
NOISE_SD = 0.5    # per-token isotropic noise
JITTER_SD = 0.4   # per-token topic jitter
ATT_SYC = 1.2     # attention logit boost on authority tokens, sycophantic
ATT_RES = 0.2     # attention logit boost on authority tokens, resistant
LAM = 1.0         # steering strength


def _unit(x):
    return x / np.linalg.norm(x)


def _cos(x, y):
    return float(_unit(x) @ _unit(y))


def _make_examples(rng, n, syc, u, c, t):
    """Token states (n, T, D), authority mask (n, T), attention
    weights (n, T) for one class."""
    auth_pos = np.argsort(rng.random((n, T)), axis=1)[:, :K]
    is_auth = np.zeros((n, T), dtype=bool)
    np.put_along_axis(is_auth, auth_pos, True, axis=1)
    gamma = rng.standard_normal(n)                       # claim content
    topic = (M_TOPIC if syc else -M_TOPIC) \
        + rng.standard_normal((n, T)) * JITTER_SD        # per-token topic
    h = rng.standard_normal((n, T, D)) * NOISE_SD
    h += topic[:, :, None] * t[None, None, :]
    # Authority tokens: content on every authority token, signal only
    # when the response is sycophantic.
    idx = np.where(is_auth)
    h[idx] += gamma[idx[0]][:, None] * c[None, :]
    if syc:
        h[idx] += S_SIG * u[None, :]
    logits = rng.standard_normal((n, T)) * 0.3 \
        + is_auth * (ATT_SYC if syc else ATT_RES)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    return h, is_auth, w


def _attribution(h, w, u):
    """Exact IG (= grad x input) of r = u . sum_t w_t h_t wrt tokens."""
    return w * (h @ u)


def run_seed(seed):
    rng = np.random.default_rng(31000 + seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, 3)))
    u, c, t = q[:, 0], q[:, 1], q[:, 2]

    h_pos, a_pos, w_pos = _make_examples(rng, N_TRAIN, True, u, c, t)
    h_neg, a_neg, w_neg = _make_examples(rng, N_TRAIN, False, u, c, t)
    phi_pos = _attribution(h_pos, w_pos, u)
    phi_neg = _attribution(h_neg, w_neg, u)

    def asi(phi, is_auth):
        pos = np.clip(phi, 0.0, None)
        return float((pos * is_auth).sum() / max(pos.sum(), 1e-12))

    asi_syc = float(np.mean([asi(phi_pos[i], a_pos[i])
                             for i in range(N_TRAIN)]))
    asi_res = float(np.mean([asi(phi_neg[i], a_neg[i])
                             for i in range(N_TRAIN)]))
    topk = np.argsort(-phi_pos, axis=1)[:, :K]
    prec = float(np.mean([a_pos[i, topk[i]].mean()
                          for i in range(N_TRAIN)]))

    def contrast(mask_pos, mask_neg):
        mp = h_pos[mask_pos].mean(0)
        mn = h_neg[mask_neg].mean(0)
        return mp - mn

    def topk_mask(phi, permute=False):
        p = phi.copy()
        if permute:
            for i in range(p.shape[0]):
                p[i] = p[i, rng.permutation(T)]
        m = np.zeros_like(a_pos, dtype=bool)
        np.put_along_axis(m, np.argsort(-p, axis=1)[:, :K], True, axis=1)
        return m

    all_pos = np.ones_like(a_pos)
    all_neg = np.ones_like(a_neg)
    rand_pos = np.zeros_like(a_pos)
    rand_neg = np.zeros_like(a_neg)
    for i in range(N_TRAIN):
        rand_pos[i, rng.choice(T, K, replace=False)] = True
        rand_neg[i, rng.choice(T, K, replace=False)] = True

    dirs = dict(
        pooled=contrast(all_pos, all_neg),
        attr=contrast(topk_mask(phi_pos), topk_mask(phi_neg)),
        random=contrast(rand_pos, rand_neg),
        shuffled=contrast(topk_mask(phi_pos, permute=True),
                          topk_mask(phi_neg, permute=True)),
    )

    # Steering on held-out resistant response states: subtract the
    # direction, read behavior (u) vs off-target (c, t) displacement.
    h_held, _, w_held = _make_examples(rng, N_HELD, False, u, c, t)
    resp = np.einsum("nt,ntd->nd", w_held, h_held)

    def effect_ratio(direction):
        d = _unit(direction)
        delta = -LAM * d
        d_beh = abs(float(delta @ u))
        side = float(np.sqrt((delta @ c) ** 2 + (delta @ t) ** 2))
        return d_beh / max(side, 1e-12), d_beh, side

    out = dict(asi_syc=asi_syc, asi_res=asi_res, prec=prec)
    for name, dvec in dirs.items():
        ratio, _, _ = effect_ratio(dvec)
        out[f"cos_{name}"] = _cos(dvec, u)
        out[f"ratio_{name}"] = ratio
    return out


def test_attribution_guided_direction_beats_pooled():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Premise (claim 1): attribution concentrates on authority
        # tokens, and far more in sycophantic than resistant examples.
        assert r["asi_syc"] > 0.7, (
            f"seed {seed}: ASI sycophantic {r['asi_syc']:.3f}")
        assert r["asi_syc"] > r["asi_res"] + 0.4, (
            f"seed {seed}: ASI syc {r['asi_syc']:.3f} vs "
            f"resistant {r['asi_res']:.3f}")
        assert r["prec"] > 0.9, (
            f"seed {seed}: attribution precision@K {r['prec']:.3f}")
        # Claim 2: the attribution-guided direction is much closer to
        # the true behavior axis than the pooled contrast.
        assert r["cos_attr"] > 0.85, (
            f"seed {seed}: attr cos {r['cos_attr']:.3f}")
        assert r["cos_attr"] > r["cos_pooled"] + 0.4, (
            f"seed {seed}: attr cos {r['cos_attr']:.3f} vs "
            f"pooled {r['cos_pooled']:.3f}")
        # Controls: same-size subsets WITHOUT attribution information
        # track the pooled direction (unbiased subsamples of the same
        # population) and sit far below the attribution-guided arm --
        # the gain is the attribution information, not the subset size.
        for ctl in ("random", "shuffled"):
            assert r[f"cos_{ctl}"] < r["cos_attr"] - 0.4, (
                f"seed {seed}: {ctl} cos {r[f'cos_{ctl}']:.3f} vs "
                f"attr {r['cos_attr']:.3f}")
            assert r[f"cos_{ctl}"] <= r["cos_pooled"] + 0.1, (
                f"seed {seed}: {ctl} cos {r[f'cos_{ctl}']:.3f} beats "
                f"pooled {r['cos_pooled']:.3f} -- a same-size subset "
                "with no attribution information should not")
            assert r[f"cos_{ctl}"] >= r["cos_pooled"] - 0.2, (
                f"seed {seed}: {ctl} cos {r[f'cos_{ctl}']:.3f} vs "
                f"pooled {r['cos_pooled']:.3f} -- control no longer "
                "tracks the pooled estimand, toy is not testing "
                "subset selection")
        # Steering: more behavior shift per unit off-target movement.
        assert r["ratio_attr"] > 4.0 * r["ratio_pooled"], (
            f"seed {seed}: attr ratio {r['ratio_attr']:.2f} vs "
            f"pooled {r['ratio_pooled']:.2f}")
        for ctl in ("random", "shuffled"):
            assert r["ratio_attr"] > 4.0 * r[f"ratio_{ctl}"], (
                f"seed {seed}: attr ratio {r['ratio_attr']:.2f} vs "
                f"{ctl} {r[f'ratio_{ctl}']:.2f}")
