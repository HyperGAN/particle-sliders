"""CPU gate for the ExSD claims (arXiv:2611.28638): "Make Position
Uninformative: Exchangeable Self-Distillation Flattens the
Lost-in-the-Middle Curve" (Ma, Wang, Liu & King).

Paper claim (from the abstract): retrieval-augmented models answer
well when the relevant document comes first or last and poorly in
the middle, and training contexts teach that positional bias.
Exchangeable Self-Distillation (ExSD) trains a small adapter on
SHUFFLED contexts to reproduce the model's own answer when it
reads the answer document ALONE. It needs no gold answers, and in
the paper it matches or beats gold-answer fine-tuning on both
average and worst-position accuracy.

Chain anchor (in-house): PR #218 (Shuffled OPSD) established the
curation rule -- teacher-side frozen, student-side permutation is
safe ONLY for order-invariant axes (student-shuffle 1.000 vs clean
0.932, teacher-shuffle 0.540). ExSD is the constructive dual of
that rule: where an axis SHOULD be order-invariant, permutation
training is how the invariance gets INSTALLED. This toy tests the
installation claim in the document-grounded slider seat: the
position of the evidence inside the context is a nuisance
ordering the behaviour must not depend on.

Toy operationalization (an adaptation of the abstract's
construction, stated precisely because the toy is ours, not the
paper's): a context is K = 5 documents in R^D (D = 12). Exactly
one document is DECISIVE: it is a class prototype mu_y (C = 4
classes, orthonormal prototypes) plus noise, and the correct
answer is the function y of the decisive document's CONTENT only.
The other four documents are content-free noise. Contexts are
generated in RETRIEVER ORDER -- every document gets a noisy
relevance score, the decisive document's score is highest in
expectation, documents are sorted by score -- so in the natural
presentation position is INFORMATIVE (the decisive document sits
at the front ~60% of the time). That correlation is the paper's
bias-teaching mechanism, reproduced honestly.

Student: an attention-lite reader. Logits are
    logits = (sum_p softmax(alpha)_p * doc_p) @ (MU + Delta)^T,
a position-weighted pool followed by a prototype readout. The
base reader KNOWS the content map (MU is exact, Delta init 0),
so reading the decisive document alone answers ~correctly. The
trainable adapter has two parts -- the position logits alpha
(K = 5 numbers) and the readout delta Delta (C x D, init 0) --
the toy form of the paper's small adapter: alpha is the position
pathway, Delta the content pathway, and wrong targets have a
content pathway they can corrupt (with a fixed readout they
could not, and the target-side controls would test nothing).
The
adapter is INITIALISED with a primacy tilt,
alpha0 = [1.2, 0.6, 0, -0.6, -1.2] (pool weights ~[0.48, 0.26,
0.14, 0.08, 0.04]): the honest bias source, standing in for the
position-informative training history a real model arrives with.
Measured before any training (arm 1), accuracy falls from the
front placement to the back placement -- the positional gap this
test exists to close.

TARGET CONSTRUCTION (ExSD's shape): the target for a context is
the FROZEN initial student's own output distribution when it
reads the decisive document ALONE (single-document context, where
the pool weight is 1 whatever alpha says, so the target is
position-free by construction). No gold labels are used anywhere
in arms 2, 3, 5 or 6.

ARMS (same init, same step count, same optimiser):
  (1) NO TRAINING -- bias baseline.
  (2) UNSHUFFLED SELF-DISTILLATION -- retriever-order contexts,
      doc-alone targets. Controls for self-distillation itself:
      position stays informative, so the bias should survive.
  (3) ExSD -- contexts re-shuffled uniformly at every step,
      doc-alone targets. Position is made uninformative; the
      optimal adapter ignores it.
  (4) GOLD FINE-TUNING -- retriever-order contexts, true one-hot
      labels. The ceiling ExSD is claimed to match.
  (5) TARGET-PERMUTED CONTROL -- shuffled contexts, but each
      context is trained toward ANOTHER context's doc-alone
      target (teacher-side permutation, cf. #218's teacher-
      shuffle harm). Content signal destroyed; must NOT install
      anything useful.
  (6) FIXED-TARGET CONTROL -- shuffled contexts, every target is
      one fixed doc-alone output regardless of content. Must
      collapse toward a constant answer and fail accuracy.

READOUTS (paired evaluation: the same 300 content sets with the
decisive document placed at each of the 5 positions):
  positional gap = accuracy(decisive at front) - accuracy(at
  back) -- the headline; accuracy overall and per placement;
  worst-position accuracy; invariance = fraction of content sets
  whose argmax answer is identical across all 5 placements.

RESULTS (pop-os, 5 seeds, torch threads = 1; means with
[seed min, max] -- the assertions below encode exactly these
margins, no more):
  baseline        gap 0.491 [0.450, 0.543], mean acc 0.489,
                  worst-position 0.286, invariance 0.378;
                  accuracy by position [0.777, 0.603, 0.447,
                  0.332, 0.286] -- the primacy curve.
  unshuffled SD   gap 0.529 [0.480, 0.580], mean 0.463,
                  worst 0.255, invariance 0.313 -- self-
                  distillation on informative order PRESERVES
                  (slightly sharpens) the bias.
  ExSD            gap 0.007 [-0.033, 0.043], mean 0.586,
                  worst 0.553, invariance 0.900; accuracy by
                  position [0.591, 0.582, 0.573, 0.598, 0.584]
                  -- the curve is FLAT, worst position nearly
                  doubles vs baseline, no gold labels anywhere.
  gold FT         gap 0.499 [0.427, 0.597], mean 0.466,
                  worst 0.285, invariance 0.349 -- true labels
                  on the same informative order do NOT close the
                  gap in this toy; ExSD beats the gold arm's gap
                  by >= 0.25 on every seed (the paper's "matches
                  or beats gold-answer fine-tuning", here "beats",
                  because the gold arm inherits its training
                  order's bias -- the shuffle, not the label, is
                  the active ingredient).
  target-permuted gap -0.005, mean acc 0.243 [0.200, 0.281] --
                  chance: teacher-side permutation destroys the
                  content signal (its flat gap / 0.834 invariance
                  are vacuous -- agreement on garbage; accuracy
                  is the discriminating readout).
  fixed-target    gap 0.002, mean acc 0.236 [0.218, 0.253] --
                  chance, as a content-blind target must give.

Iteration log (10-iteration loop, one refinement per round):
  R1: alpha-only adapter. ExSD closed the gap (-0.005) and
      unshuffled SD preserved it (0.563), BUT the target-side
      controls also landed at gap ~0 / acc ~0.58: with a fixed
      perfect readout, accuracy is a function of alpha alone and
      any target pushes alpha toward uniform (variance
      shrinkage) -- the controls tested nothing. Gold collapsed
      to front-only reading (acc by position [0.815, 0.249,
      0.248, 0.248, 0.248]).
  R2: adapter gains a content pathway (readout delta, init 0).
      Controls now fail at chance (0.243 / 0.236), gold behaves
      (no collapse; gap 0.499 ~ baseline), ExSD flat at 0.007.
      All margins clear on all 5 seeds.
  R3: assertions tightened to the measured margins (incl. the
      ExSD-beats-gold gap margin >= 0.25/seed and the gold-
      preserves-bias control); no mechanism change. PROVEN.

What is NOT asserted: anything about real LLMs, tokenisation, or
the paper's 1.5B-32B numbers; a U-shaped (middle-worst) curve --
this toy isolates a primacy tilt, one half of lost-in-the-middle,
and the front-back gap is its readout; that permutation training
helps for axes that are NOT order-invariant (#218 says the
opposite, and arm 5 is its echo here).
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
    from particlegan import get_recipe  # noqa: E402,F401
except Exception:  # a particlegan without the recipe API at all
    get_recipe = None


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ---------------------------------------------------------------- toy
D = 12          # document dimension
C = 4           # answer classes
K = 5           # documents per context
DEC_NOISE = 0.50
DIS_NOISE = 0.35
ALPHA0 = torch.tensor([1.2, 0.6, 0.0, -0.6, -1.2])
N_TRAIN = 1200
N_EVAL_SETS = 300
STEPS = 400
BATCH = 200
LR = 0.08
SEEDS = [0, 1, 2, 3, 4]

_PROTO_RNG = np.random.default_rng(999)
_Q, _ = np.linalg.qr(_PROTO_RNG.standard_normal((D, D)))
MU = torch.tensor(_Q[:C], dtype=torch.float32)  # (C, D) orthonormal rows


def _gen_sets(rng, n):
    """Content sets: docs stored decisive-first, plus gold labels."""
    y = torch.tensor(rng.integers(0, C, size=n))
    dec = MU[y] + DEC_NOISE * torch.tensor(
        rng.standard_normal((n, D)), dtype=torch.float32)
    dis = DIS_NOISE * torch.tensor(
        rng.standard_normal((n, K - 1, D)), dtype=torch.float32)
    docs = torch.cat([dec.unsqueeze(1), dis], dim=1)  # (n, K, D)
    return docs, y


def _retriever_order(rng, n):
    """Permutation sorting docs by noisy relevance, decisive (slot
    0 in storage) highest in expectation. Returns perm such that
    docs[i, perm[i]] is the presented context."""
    dec_score = 1.0 + 0.5 * rng.standard_normal(n)
    dis_score = 0.5 * rng.standard_normal((n, K - 1))
    scores = np.concatenate([dec_score[:, None], dis_score], axis=1)
    return torch.tensor(np.argsort(-scores, axis=1))


def _teacher_targets(docs):
    """Frozen initial student's output on the decisive doc ALONE."""
    with torch.no_grad():
        logits = docs[:, 0] @ MU.T
        return torch.softmax(logits, dim=-1)


def _forward(alpha, delta, docs):
    w = torch.softmax(alpha, dim=0)
    pooled = (docs * w.view(1, K, 1)).sum(dim=1)
    return pooled @ (MU + delta).T


def _train(alpha_init, docs, targets, order_mode, seed):
    """order_mode: 'retriever' (fixed order) or 'shuffle' (fresh
    uniform permutation of every presented context at every step).
    Trains the adapter (alpha, Delta) from (alpha_init, 0)."""
    rng = np.random.default_rng(10_000 + seed)
    alpha = alpha_init.clone().requires_grad_(True)
    delta = torch.zeros(C, D, requires_grad=True)
    opt = torch.optim.Adam([alpha, delta], lr=LR)
    n = docs.shape[0]
    base_perm = _retriever_order(np.random.default_rng(777 + seed), n)
    for _ in range(STEPS):
        idx = torch.tensor(rng.integers(0, n, size=BATCH))
        batch = docs[idx]
        if order_mode == "retriever":
            perm = base_perm[idx]
        else:
            perm = torch.tensor(np.argsort(rng.random((BATCH, K)), axis=1))
        batch = torch.gather(
            batch, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        logits = _forward(alpha, delta, batch)
        loss = -(targets[idx] * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return alpha.detach(), delta.detach()


@torch.no_grad()
def _evaluate(adapter, eval_docs, eval_y):
    """eval_docs: (n_sets, K positions, K, D) paired placements."""
    alpha, delta = adapter
    accs = []
    for p in range(K):
        pred = _forward(alpha, delta, eval_docs[:, p]).argmax(-1)
        accs.append((pred == eval_y).float().mean().item())
    preds = torch.stack([
        _forward(alpha, delta, eval_docs[:, p]).argmax(-1) for p in range(K)
    ], dim=1)
    agree_all = (preds == preds[:, :1]).all(dim=1).float().mean().item()
    return {
        "acc_by_pos": accs,
        "gap": accs[0] - accs[-1],
        "mean_acc": float(np.mean(accs)),
        "worst_acc": float(np.min(accs)),
        "invariance": agree_all,
    }


def run_seed(seed):
    rng = np.random.default_rng(seed)
    docs, y = _gen_sets(rng, N_TRAIN)
    targets = _teacher_targets(docs)
    # deranged targets for arm 5 (teacher-side permutation)
    der = torch.tensor(np.random.default_rng(555 + seed).permutation(N_TRAIN))
    if (der == torch.arange(N_TRAIN)).any():
        der = torch.roll(der, 1)
    fixed_target = targets[0:1].expand(N_TRAIN, C).contiguous()
    gold = torch.zeros(N_TRAIN, C).scatter_(1, y.unsqueeze(1), 1.0)

    # paired eval sets: same content, decisive at each position
    edocs, ey = _gen_sets(np.random.default_rng(20_000 + seed), N_EVAL_SETS)
    eval_docs = torch.zeros(N_EVAL_SETS, K, K, D)
    for p in range(K):
        placed = edocs.clone()
        dec = edocs[:, 0].clone()
        rest = edocs[:, 1:]
        placed[:, p] = dec
        others = [i for i in range(K) if i != p]
        placed[:, others] = rest
        eval_docs[:, p] = placed

    zero_delta = torch.zeros(C, D)
    arms = {}
    arms["baseline"] = _evaluate((ALPHA0, zero_delta), eval_docs, ey)
    arms["unshuffled_sd"] = _evaluate(
        _train(ALPHA0, docs, targets, "retriever", seed), eval_docs, ey)
    arms["exsd"] = _evaluate(
        _train(ALPHA0, docs, targets, "shuffle", seed), eval_docs, ey)
    arms["gold"] = _evaluate(
        _train(ALPHA0, docs, gold, "retriever", seed), eval_docs, ey)
    arms["target_permuted"] = _evaluate(
        _train(ALPHA0, docs, targets[der], "shuffle", seed), eval_docs, ey)
    arms["fixed_target"] = _evaluate(
        _train(ALPHA0, docs, fixed_target, "shuffle", seed), eval_docs, ey)
    return arms


_RESULTS = None


def results():
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = {s: run_seed(s) for s in SEEDS}
    return _RESULTS


def _agg(arm, key):
    return [results()[s][arm][key] for s in SEEDS]


# ---------------------------------------------------------------- gate
def test_bias_exists_before_training():
    for s in SEEDS:
        assert results()[s]["baseline"]["gap"] >= 0.25, s
        assert results()[s]["baseline"]["acc_by_pos"][0] >= 0.70, s


def test_exsd_closes_gap_to_gold_level_without_gold_labels():
    for s in SEEDS:
        r = results()[s]
        assert r["exsd"]["gap"] <= 0.10, (s, r["exsd"]["gap"])
        # measured: ExSD does not merely match the gold arm, it
        # beats its gap by a wide margin on every seed -- gold
        # labels on position-informative order inherit the bias
        assert r["exsd"]["gap"] + 0.25 <= r["gold"]["gap"], s
        assert r["exsd"]["mean_acc"] >= r["gold"]["mean_acc"] - 0.06, s
        assert r["exsd"]["worst_acc"] >= r["baseline"]["worst_acc"] + 0.12, s


def test_gold_labels_on_informative_order_also_preserve_bias():
    for s in SEEDS:
        r = results()[s]
        assert r["gold"]["gap"] >= 0.5 * r["baseline"]["gap"], s


def test_unshuffled_self_distillation_preserves_bias():
    for s in SEEDS:
        r = results()[s]
        assert r["unshuffled_sd"]["gap"] >= 0.5 * r["baseline"]["gap"], s
        assert r["unshuffled_sd"]["gap"] >= r["exsd"]["gap"] + 0.12, s


def test_permuted_or_fixed_targets_install_nothing():
    for s in SEEDS:
        r = results()[s]
        assert r["target_permuted"]["mean_acc"] <= 0.40, s
        assert r["fixed_target"]["mean_acc"] <= 0.40, s
        assert r["exsd"]["mean_acc"] >= r["target_permuted"]["mean_acc"] + 0.20, s


def test_exsd_installs_permutation_invariance():
    for s in SEEDS:
        r = results()[s]
        assert r["exsd"]["invariance"] >= 0.75, s
        assert r["exsd"]["invariance"] >= r["baseline"]["invariance"] + 0.15, s


if __name__ == "__main__":
    res = results()
    keys = ["gap", "mean_acc", "worst_acc", "invariance"]
    for arm in ["baseline", "unshuffled_sd", "exsd", "gold",
                "target_permuted", "fixed_target"]:
        line = f"{arm:16s}"
        for k in keys:
            vals = [res[s][arm][k] for s in SEEDS]
            line += f" {k}={np.mean(vals):.3f}[{np.min(vals):.3f},{np.max(vals):.3f}]"
        print(line)
    for arm in ["baseline", "unshuffled_sd", "exsd", "gold"]:
        pos = np.mean([res[s][arm]["acc_by_pos"] for s in SEEDS], axis=0)
        print(f"{arm:16s} acc_by_pos={np.round(pos, 3)}")
