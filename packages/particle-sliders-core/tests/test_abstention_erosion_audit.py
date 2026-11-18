"""CPU gate for the Guardrail Erosion claim (arXiv:2611.08780):
fine-tuning on domain data that contains no abstention examples erodes
knowledge-centric abstention, a paired before/after audit on
knowledge-limit probes detects the drop, and the size of the drop is
heterogeneous across fine-tuning configurations.

Paper: "Guardrail Erosion in Domain Fine-Tuned Large Language Models:
A Paired Audit of Knowledge-Centric Abstention in Biomedical LLMs"
(Peng et al.). Its paired audit of 25 biomedical fine-tuned models
against 11 paired base checkpoints on MedAbstain found appropriate
abstention was already low in base models (median 8.3%) and fell
further after fine-tuning (median 4.2%), decreasing in 21 of 25 pairs,
with the magnitude and direction of change heterogeneous across pairs.
The paper claims erosion and heterogeneity; it does NOT claim a
mitigation, and neither does this gate (the preservation arm below is
an exploratory diagnostic, kept separate from the paper's claims).

Toy operationalization: a tiny MLP readout over synthetic
document-grounded states -- the Tiny Aya contract in miniature. An
input is [entity one-hot (32), evidence vector (8)]; the label is one
of 8 answer classes or UNKNOWN. Present-evidence examples carry a
blurred one-hot of the true answer plus Gaussian noise; absent-evidence
examples carry norm-matched pure noise -- no answer signal, but not a
trivially detectable zero vector, so "does the evidence actually
encode an answer?" is a soft signal-detection judgement, not a norm
check. The base model is trained on a balanced mix and learns both
halves of the contract: answer from the evidence when it is present,
output UNKNOWN when it is absent (held-out entities throughout, so
both behaviours must generalise, not memorise).

Arms per seed (all fine-tunes start from the same paired base
checkpoint, and every arm is scored on the SAME held-out
absent-evidence probe tensors -- the paired audit):
  (a) domain fine-tune, two configurations -- "short" (100 steps,
      lr 1e-3) and "long" (400 steps, lr 5e-3) -- on present-evidence
      examples ONLY, mirroring a domain fine-tune whose data contains
      no abstention examples;
  (b) no-training control: the paired base checkpoint re-scored
      unchanged (the audit's measured delta must be exactly 0);
  (c) EXPLORATORY preservation arm: the "long" configuration trained
      on a balanced present/absent mix instead. Reported as a
      diagnostic only -- the paper claims no mitigation.

Metrics per seed: base abstention rate and present-evidence accuracy;
per arm, the paired delta in abstention rate on the probes and the
target-behaviour gain (change in present-evidence accuracy -- the
fine-tune must demonstrably buy something, or the erosion is
uninteresting). Measured on pop-os (particlegan 0.10.1): base
abstention 0.71-0.73 at base accuracy 0.78-0.86; domain fine-tune
delta -0.58..-0.60 (short) and -0.71..-0.73 (long, i.e. abstention
collapses to ~0) with target gains +0.10..+0.17; control delta
exactly 0; exploratory arm delta -0.02..-0.05. Heterogeneity: the
long configuration erodes ~0.13 more than the short one in every
seed, while seed-to-seed spread within a configuration is small --
in this toy the heterogeneity lives in the fine-tuning
configuration, which is the axis the paper's pairs vary along.
The toy's base abstention (~0.72) is far above the paper's median
8.3% by design -- a competent-but-unsaturated base is what makes
the paired delta measurable; the gate asserts the delta and its
heterogeneity, never the level. Skips cleanly when the installed
particlegan does not provide the EMPF recipe.
"""
import copy

import numpy as np
import pytest
import torch
from torch import nn
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


E = 32            # entities
K = 8             # answer classes (+ UNKNOWN)
UNK = K
N_TRAIN_ENT = 24  # remaining 8 entities are held out (probes / eval)
HID = 64
EV_KEEP = 0.85    # evidence: EV_KEEP * one-hot(answer) + uniform blur
EV_NOISE = 0.25   # Gaussian noise on every evidence vector
BASE_STEPS = 300
CONFIGS = {"short": (100, 1e-3), "long": (400, 5e-3)}
SEEDS = (0, 1, 2)


def make_examples(rng, entities, n, present_frac):
    """x = [entity one-hot (E), evidence (K)]. Present: evidence is a
    blurred one-hot of the true answer + noise, label = answer.
    Absent: evidence is pure noise of matched scale, label = UNKNOWN."""
    ent = rng.integers(0, len(entities), size=n)
    ans = rng.integers(0, K, size=n)
    present = rng.random(n) < present_frac
    x = np.zeros((n, E + K), dtype=np.float32)
    x[np.arange(n), entities[ent]] = 1.0
    ev = np.zeros((n, K), dtype=np.float32)
    ev[present] = (EV_KEEP * np.eye(K, dtype=np.float32)[ans[present]]
                   + (1 - EV_KEEP) / K)
    ev += rng.standard_normal((n, K)).astype(np.float32) * EV_NOISE
    x[:, E:] = ev
    y = np.where(present, ans, UNK).astype(np.int64)
    return torch.tensor(x), torch.tensor(y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(E + K, HID)
        self.fc2 = nn.Linear(HID, K + 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train(model, rng, entities, present_frac, steps, lr, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_examples(rng, entities, bs, present_frac)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def abstention_rate(model, x):
    return float((model(x).argmax(-1) == UNK).float().mean())


@torch.no_grad()
def accuracy(model, x, y):
    return float((model(x).argmax(-1) == y).float().mean())


def run_seed(seed):
    train_ent = np.arange(N_TRAIN_ENT)
    probe_ent = np.arange(N_TRAIN_ENT, E)
    torch.manual_seed(900 + seed)
    base = Net()
    train(base, np.random.default_rng(5000 + seed), train_ent, 0.5,
          BASE_STEPS, 3e-3)
    # The paired audit: identical probe / target tensors for every arm.
    px, _ = make_examples(np.random.default_rng(31337 + seed),
                          probe_ent, 512, 0.0)
    tx, ty = make_examples(np.random.default_rng(777 + seed),
                           train_ent, 512, 1.0)
    out = {"base_abst": abstention_rate(base, px),
           "base_acc": accuracy(base, tx, ty)}
    # (b) no-training control: re-score the paired checkpoint unchanged.
    out["ctrl_d"] = abstention_rate(copy.deepcopy(base), px) - out["base_abst"]
    # (a) domain fine-tunes: present-evidence examples only.
    for name, (steps, lr) in CONFIGS.items():
        ft = copy.deepcopy(base)
        train(ft, np.random.default_rng(6000 + seed), train_ent, 1.0,
              steps, lr)
        out[f"{name}_d"] = abstention_rate(ft, px) - out["base_abst"]
        out[f"{name}_gain"] = accuracy(ft, tx, ty) - out["base_acc"]
    # (c) EXPLORATORY preservation arm: long config, balanced mix.
    steps, lr = CONFIGS["long"]
    ex = copy.deepcopy(base)
    train(ex, np.random.default_rng(7000 + seed), train_ent, 0.5,
          steps, lr)
    out["expl_d"] = abstention_rate(ex, px) - out["base_abst"]
    return out


def test_abstention_erosion_paired_audit():
    _empf_recipe()
    results = {seed: run_seed(seed) for seed in SEEDS}
    deltas = []
    for seed, r in results.items():
        # The paired base is competent at both halves of the contract.
        assert r["base_abst"] >= 0.60, (
            f"seed {seed}: base abstention {r['base_abst']:.3f}")
        assert r["base_acc"] >= 0.70, (
            f"seed {seed}: base accuracy {r['base_acc']:.3f}")
        # The audit itself has no drift: an unchanged checkpoint
        # re-scored on the same probes measures exactly zero.
        assert r["ctrl_d"] == 0.0, f"seed {seed}: control delta {r['ctrl_d']}"
        # (a) Domain fine-tuning erodes abstention by a clear margin in
        # every seed and configuration, while genuinely buying target
        # behaviour (present-evidence accuracy gain).
        for name in CONFIGS:
            deltas.append(r[f"{name}_d"])
            assert r[f"{name}_d"] <= -0.25, (
                f"seed {seed} {name}: abstention delta {r[f'{name}_d']:.3f}")
            assert r[f"{name}_gain"] >= 0.05, (
                f"seed {seed} {name}: target gain {r[f'{name}_gain']:.3f}")
        # (c) EXPLORATORY: mixing abstention examples back into the
        # fine-tune largely preserves abstention in this toy. The paper
        # claims no mitigation; this only shows the erosion above is
        # driven by the fine-tune data's missing abstention examples,
        # not by further training as such.
        assert r["expl_d"] >= -0.20, (
            f"seed {seed}: exploratory delta {r['expl_d']:.3f}")
        assert abs(r["expl_d"]) <= 0.3 * abs(r["long_d"]), (
            f"seed {seed}: exploratory {r['expl_d']:.3f} "
            f"vs long {r['long_d']:.3f}")
    # The paired drop is the median behaviour, not one lucky seed.
    assert float(np.median(deltas)) <= -0.40, (
        f"median abstention delta {float(np.median(deltas)):.3f}")
    # Heterogeneity: the drop's magnitude depends on the fine-tuning
    # configuration -- the long configuration erodes materially more
    # than the short one, so the spread across pairs is real.
    spread = max(deltas) - min(deltas)
    assert spread >= 0.08, f"delta spread across pairs {spread:.3f}"
    mean_short = float(np.mean([results[s]["short_d"] for s in SEEDS]))
    mean_long = float(np.mean([results[s]["long_d"] for s in SEEDS]))
    assert mean_long <= mean_short - 0.05, (
        f"long {mean_long:.3f} vs short {mean_short:.3f}")
