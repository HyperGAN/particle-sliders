"""CPU gate for the Jev-audit claims (arXiv:2611.12521): bundling
multiple questions into one single-pass decision request induces
cross-question interference that flips individual answers, and the
model's reported confidence -- a closed-form dispersion transform of
the output distribution, not an accuracy estimate -- does not flag
the flips, so confidence-based triage cannot filter them out.

Paper: "Can the Confidence and Decisions of System 1 Models Be
Trusted? An Explainability and Perturbation Audit of Jev-Style
Decision Models" (Delaunay). System 1 decision models (TypeSafe
AI's Jev family) replace generation with one forward pass returning
typed distributions. Two of its five findings are gated here:
(2) bundling multiple questions in one request induces directional
cross-question interference -- 11.7% to 22.4% binary flips on
Tev1-0.8B vs 0.1% to 0.2% on Tev1-4B, i.e. the effect is a property
of the shared forward pass, worst on small models; and (1) provider
confidence is exactly 1 - H(p)/ln K, a dispersion transform of the
output distribution rather than empirical accuracy.

Toy operationalization: a minimal System-1-style scorer. Each
question has features x in R^16 and a solo score s = w . x; the solo
decision is sign(s) and the solo distribution is
softmax([s, -s] / TAU). In the bundled arm, B = 4 questions share
one forward pass whose trunk leaks across questions:
h_i = x_i + BETA * mean_{j != i} x_j, and the decision is
sign(w . h_i). The leak is the toy's stand-in for the shared
representation a real bundled request computes; the masked arm runs
the identical bundled code path with the leak removed (BETA = 0),
which is what per-question scoring guarantees.

Metrics per seed, over N_Q questions: flip rate = fraction of
questions whose bundled decision differs from their solo decision;
confidence = 1 - H(p)/ln 2 from the bundled distribution. Finding
(1) is gated in its operational form: confidence is a function of
the output distribution alone, so a flipped answer and a kept answer
with the same |score| receive the same confidence to within
matching tolerance -- at fixed dispersion, confidence carries no
information about whether bundling flipped this answer -- and flips
persist (1-3%) even among answers with confidence >= 0.8, where the
masked arm has none. (Iteration 1 asserted the stronger claim that
confidence-based triage cannot reduce flips at all; the toy refuted
it -- flips strike low-margin questions first, so triage does cut
the flip rate, from ~0.18 to ~0.045 in the top half. The paper does
not claim otherwise; the gate now asserts exactly what the paper
claims and the toy demonstrates.)

Measured on pop-os (particlegan 0.10.1): flip rate 0.164-0.184
(paper: 0.117-0.224 on Tev1-0.8B); masked arm 0.0000; flip rate at
confidence >= 0.8 is 0.012-0.029; 130-146 matched |score| pairs per
seed with confidence differing by <= 0.0044. Skips cleanly when the
installed particlegan does not provide the EMPF recipe.
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


D = 16
N_Q = 800          # questions per seed
B = 4              # bundle size
BETA = 1.1         # shared-trunk leak strength
TAU = 0.7          # softmax temperature for the output distribution


def _confidence(scores):
    logits = np.stack([scores, -scores], -1) / TAU
    logits = logits - logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(-1, keepdims=True)
    h = -(p * np.log(p + 1e-12)).sum(-1)
    return 1.0 - h / np.log(2.0)


def run_seed(seed, beta):
    _empf_recipe()
    rng = np.random.default_rng(17000 + seed)
    w = rng.standard_normal(D)
    w = w / np.linalg.norm(w)
    x = rng.standard_normal((N_Q, D))
    solo_scores = x @ w
    solo_dec = np.sign(solo_scores)
    flips = np.zeros(N_Q, dtype=bool)
    bundled_scores = np.zeros(N_Q)
    for start in range(0, N_Q, B):
        xb = x[start:start + B]
        others_mean = (xb.sum(0)[None, :] - xb) / (B - 1)
        h = xb + beta * others_mean
        s = h @ w
        bundled_scores[start:start + B] = s
        flips[start:start + B] = np.sign(s) != solo_dec[start:start + B]
    conf = _confidence(bundled_scores)
    return {
        "flip_rate": float(flips.mean()),
        "conf": conf, "flips": flips, "scores": bundled_scores,
    }


def test_jev_bundled_interference():
    for seed in range(3):
        r = run_seed(seed, BETA)
        m = run_seed(seed, 0.0)
        # Bundling flips a substantial share of answers (paper:
        # 11.7-22.4% binary flips on the small model).
        assert 0.08 <= r["flip_rate"] <= 0.32
        # The identical bundled path with the cross-question leak
        # removed flips essentially nothing: the flips are the shared
        # forward pass, not the questions.
        assert m["flip_rate"] <= 0.01
        conf, flips, scores = r["conf"], r["flips"], r["scores"]
        # Flips persist among very confident answers, where the
        # masked arm has none at any confidence.
        hi = conf >= 0.8
        assert float(flips[hi].mean()) >= 0.008
        # At fixed dispersion, confidence is blind to correctness:
        # many flipped answers have a kept answer with |score|
        # matched to 0.01 whose confidence differs by <= 0.01 --
        # confidence is a function of the distribution alone and
        # cannot encode whether bundling flipped this answer.
        kept = np.where(~flips)[0]
        pairs, max_diff = 0, 0.0
        for i in np.where(flips)[0]:
            d = np.abs(np.abs(scores[kept]) - abs(scores[i]))
            k = int(np.argmin(d))
            if d[k] < 0.01:
                pairs += 1
                max_diff = max(max_diff, abs(conf[kept[k]] - conf[i]))
        assert pairs >= 50
        assert max_diff <= 0.01
