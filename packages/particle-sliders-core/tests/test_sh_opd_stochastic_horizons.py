"""CPU gate for the SH-OPD claim (arXiv:2611.20687): in on-policy
distillation (OPD), randomly varying the rollout horizon per training
batch -- one shared horizon per batch -- matches (or slightly beats)
dense full-horizon supervision at roughly 45% of the per-update
supervision cost, and beats a deterministic short horizon at the SAME
mean cost, because random horizons retain access to deep positions
while a fixed truncation never supervises them, and deep positions
remain learnable via cross-position supervision transfer.

Paper: "Rethinking Supervision Scaling in On-Policy Distillation:
From Dense Supervision to Stochastic Horizons" (Yao, Dou & Liu).
OPD gives dense token-level supervision along student trajectories,
but the benefit does not scale proportionally with cost: controlled
experiments show diminishing returns from additional supervised
tokens, while later positions remain learnable and show substantial
cross-position supervision transfer. SH-OPD randomly varies the
rollout horizon per training batch (one shared horizon per batch)
while preserving access to deeper states, reaching slightly higher
math reasoning accuracy at ~45% of the per-update cost of standard
OPD and competitive code performance at ~60% cost.

Relation to SSD (PR #223, arXiv:2611.12016): SSD allocates the
supervision budget PER POSITION by a divergence threshold (spend
where teacher and student disagree); SH-OPD allocates it PER BATCH
by a random horizon (spend on a random prefix length, regardless of
divergence). Both cut supervised tokens versus dense OPD, by
orthogonal dials -- this gate tests the horizon dial.

Toy operationalization: a student policy must internalize a chained
lookup world over trajectories of length L = 20. Position 0 is a cue
c in {0..3}; at every later position p the correct token is a fixed
LOOKUP TABLE entry applied to the ACTUAL previous token:
target_p = (S[c][prev] + shift[p][c]) % V, where S is a per-cue
permutation of V = 8 tokens SHARED across positions and shift[p][c]
is a per-(position, cue) rotation -- the shared component is what
cross-position transfer carries, the rotation is the
position-specific residual only direct supervision at p can teach
(the paper's gradient analysis finds exactly this decomposition:
shared update components across positions plus position-specific
residuals). A lookup entry for a wrong
previous token is a separate fact, visited only by rollouts that
actually make that error, so -- as in the SSD toy -- on-policy
rollouts matter and errors compound along the student's own
trajectory. The teacher is a scripted oracle that knows the tables;
the student is a small MLP over (cue, position, previous token) that
knows none of them. All arms run student (on-policy) rollouts for
the same number of steps and differ ONLY in which positions of each
batch receive teacher supervision; supervision cost is counted in
supervised positions.

Arms:
  (a) dense: supervise every position of every batch (cost 1.0);
  (b) sh_opd: per batch, draw ONE shared horizon h ~ Uniform{1..19}
      and supervise the first h positions (mean cost 10/19 ~ 0.53
      of dense);
  (c) fixed_short: supervise the first H_FIXED = 10 positions every
      batch (cost 10/19, matched to (b)'s mean); positions 11..19
      NEVER receive supervision in this arm;
  (d) per_seq (diagnostic, not asserted): each sequence in the batch
      draws its own horizon with the same marginal distribution as
      (b) -- same expected cost, no shared batch horizon.

Metrics per seed: per-position accuracy of the student's argmax
against the oracle target given the ACTUAL previous token along
sampled student rollouts, split into early positions (the first
H_FIXED, supervised by every arm) and deep positions (11..19, never
supervised by the fixed-short control); overall accuracy; and each
arm's realized supervised-position fraction (cost).

Iteration log (the loop that produced this gate):
  1. Fully independent per-position permutation tables, 200 steps:
     the discriminating margin is already huge (SH deep 0.60-0.69
     vs fixed-short 0.11-0.15 at matched cost ~0.53), but SH lags
     dense on deep positions by 0.31-0.40 -- a world with zero
     cross-position transfer by construction, which is exactly the
     mechanism the paper says carries deep positions.
  2. Shared-permutation + per-position-rotation tables (the paper's
     shared-component / position-specific-residual decomposition):
     SH deep rises to 0.70-0.75, fixed-short stays 0.10-0.22, but
     the SH-vs-dense deep gap is still 0.25-0.30. A per-position
     diagnostic shows SH accuracy decaying monotonically with depth
     (positions 17-19 at 0.58/0.37/0.37): the tail positions are
     supervised in only 16%/11%/5% of batches and are simply
     exposure-starved at 200 steps.
  3. Train at paper scale (500 steps, batch 96): the tail fills in.
     SH deep 0.955-0.964 vs fixed-short 0.106-0.176 (+0.78 to +0.85)
     at matched cost (SH 0.511-0.531, fixed 0.526), SH overall
     0.978-0.983 vs dense 1.000 (-0.017 to -0.022), SH early ~1.000
     like every arm. The per-sequence diagnostic arm (same marginal
     horizons, no shared batch horizon) also learns the deep
     positions (0.994-0.998) -- the gate's claim is about horizon
     COVERAGE versus fixed truncation, and the shared-batch variant
     the paper proposes is within 0.04 of it.

Measured on pop-os (particlegan 0.10.1), 3 seeds: SH-OPD deep
accuracy 0.955-0.964 vs fixed-short 0.106-0.176 and dense 1.000;
SH-OPD overall 0.978-0.983 vs dense 1.000; supervised fraction
0.511-0.531 (dense 1.0, fixed-short 0.526). Skips cleanly when the
installed particlegan does not provide the EMPF recipe.
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


V = 8
L = 20               # trajectory length; supervised positions 1..19
CUES = 4
EPS_T = 0.05
H_FIXED = 10         # fixed-short control horizon (positions < 10)
STEPS = 500
BATCH = 96
LR = 3e-3
N_EVAL = 1024
N_POS = L - 1        # 19 supervised positions


def make_tables(seed):
    """Lookup tables with the paper's cross-position structure: every
    position shares a per-cue permutation S, and each (position, cue)
    adds its own rotation: target_p = (S[c][prev] + shift[p][c]) % V.
    Supervision at any position teaches S (transfer to all positions);
    a position's own rotation is a small residual only direct
    supervision at that position can teach. Iteration 1 used fully
    independent per-position permutations -- zero transfer by
    construction -- and SH-OPD's deep positions lagged dense by
    0.31-0.40 despite beating the fixed control by ~0.5; the paper's
    mechanism is precisely that transfer exists, so the toy must
    operationalize it. tables[p] has shape (CUES, V)."""
    rng = np.random.default_rng(60000 + seed)
    shared = np.stack([rng.permutation(V) for _ in range(CUES)])
    shifts = rng.integers(0, V, size=(L, CUES))
    tables = np.empty((L, CUES, V), dtype=np.int64)
    for p in range(L):
        for c in range(CUES):
            tables[p, c] = (shared[c] + shifts[p, c]) % V
    return tables


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CUES + V + L, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, V))

    def pos_forward(self, cue, pos, prev):
        x = torch.cat([
            F.one_hot(cue, CUES).float(),
            F.one_hot(prev, V).float(),
            F.one_hot(pos, L).float()], -1)
        return self.net(x)


def oracle_dist(cue, pos, prev, tables_t):
    """Teacher distribution at one position: near-one-hot on the
    table entry for the ACTUAL previous token."""
    B = cue.shape[0]
    dev = cue.device
    tgt = tables_t[pos][cue, prev]
    p = torch.full((B, V), EPS_T / V, device=dev)
    p.scatter_(1, tgt.unsqueeze(1), 1 - EPS_T + EPS_T / V)
    return p


def train_arm(mode, seed, tables):
    torch.manual_seed(20000 + seed)
    gen = torch.Generator().manual_seed(30000 + seed)
    tables_t = torch.tensor(tables)
    student = Student()
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    sup_counts = []
    for _ in range(STEPS):
        cues = torch.randint(0, CUES, (BATCH,), generator=gen)
        # On-policy rollout: student samples its own trajectory.
        tokens = torch.zeros(BATCH, L, dtype=torch.long)
        tokens[:, 0] = cues
        dists = torch.zeros(BATCH, L, V)
        prev = cues.clone()
        with torch.no_grad():
            for pos in range(1, L):
                logits = student.pos_forward(
                    cues, torch.full_like(cues, pos), prev)
                dists[:, pos] = oracle_dist(cues, pos, prev, tables_t)
                nxt = torch.multinomial(F.softmax(logits, -1), 1,
                                        generator=gen).squeeze(1)
                tokens[:, pos] = nxt
                prev = nxt
        # Supervision mask for this batch, per arm. Mask column j
        # corresponds to position j + 1 (positions 1..19). A horizon
        # of h supervises the first h positions (columns 0..h-1).
        mask = torch.zeros(BATCH, N_POS)
        if mode == "dense":
            mask[:] = 1.0
        elif mode == "fixed_short":
            mask[:, :H_FIXED] = 1.0    # first H_FIXED positions, always
        elif mode == "sh_opd":
            h = int(torch.randint(1, L, (1,), generator=gen))
            mask[:, :h] = 1.0          # first h positions, h ~ U{1..19}
        elif mode == "per_seq":
            hs = torch.randint(1, L, (BATCH,), generator=gen)
            cols = torch.arange(N_POS).unsqueeze(0)
            mask = (cols < hs.unsqueeze(1)).float()
        else:
            raise ValueError(mode)
        cues_e = cues.unsqueeze(1).expand(-1, N_POS).reshape(-1)
        pos_e = torch.arange(1, L).unsqueeze(0).expand(BATCH, -1).reshape(-1)
        prev_e = tokens[:, :-1].reshape(-1)
        logits = student.pos_forward(cues_e, pos_e, prev_e).reshape(
            BATCH, N_POS, V)
        logp = F.log_softmax(logits, -1)
        tgt = dists[:, 1:]
        ce = -(tgt * logp).sum(-1)
        loss = (ce * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sup_counts.append(float(mask.mean()))
    return student, float(np.mean(sup_counts))


@torch.no_grad()
def evaluate(student, seed, tables):
    tables_t = torch.tensor(tables)
    gen = torch.Generator().manual_seed(40000 + seed)
    cues = torch.randint(0, CUES, (N_EVAL,), generator=gen)
    tokens = torch.zeros(N_EVAL, L, dtype=torch.long)
    tokens[:, 0] = cues
    prev = cues.clone()
    correct = torch.zeros(N_POS)
    for pos in range(1, L):
        logits = student.pos_forward(cues, torch.full_like(cues, pos), prev)
        pred = logits.argmax(-1)
        tgt = tables_t[pos][cues, prev]
        correct[pos - 1] = float((pred == tgt).float().mean())
        nxt = torch.multinomial(F.softmax(logits, -1), 1,
                                generator=gen).squeeze(1)
        tokens[:, pos] = nxt
        prev = nxt
    # Column j is position j + 1. The fixed arm supervises the first
    # H_FIXED positions (columns 0..H_FIXED-1), so deep = columns
    # H_FIXED.. is exactly its never-supervised set.
    early = correct[:H_FIXED].mean()       # positions 1..H_FIXED
    deep = correct[H_FIXED:].mean()        # positions H_FIXED+1..L-1
    return {
        "early_acc": float(early),
        "deep_acc": float(deep),
        "overall_acc": float(correct.mean()),
        "per_pos": [float(x) for x in correct],
    }


def run_seed(seed):
    _empf_recipe()
    tables = make_tables(seed)
    out = {}
    for mode in ("dense", "sh_opd", "fixed_short", "per_seq"):
        student, sup_frac = train_arm(mode, seed, tables)
        ev = evaluate(student, seed, tables)
        ev["sup_frac"] = sup_frac
        out[mode] = ev
    return out


def test_sh_opd_stochastic_horizons():
    for seed in range(3):
        r = run_seed(seed)
        dense, sh, fx = r["dense"], r["sh_opd"], r["fixed_short"]
        # Cost: SH-OPD runs at roughly half the dense supervision cost,
        # matched to the fixed-short control.
        assert sh["sup_frac"] <= 0.60
        assert abs(sh["sup_frac"] - fx["sup_frac"]) <= 0.06
        # The discriminating measurement: at matched cost, stochastic
        # horizons learn the deep positions the fixed truncation
        # never supervises (demonstrated margin +0.78 to +0.85).
        assert sh["deep_acc"] >= fx["deep_acc"] + 0.50
        # ...and SH-OPD nearly matches dense supervision overall
        # despite using ~half the supervised positions (demonstrated:
        # overall -0.017 to -0.022, deep -0.036 to -0.045 vs dense).
        assert sh["overall_acc"] >= dense["overall_acc"] - 0.05
        assert sh["deep_acc"] >= dense["deep_acc"] - 0.08
        # Early positions are learned by every arm (sanity: the gap
        # above is a horizon-coverage effect, not a broken control;
        # demonstrated fixed-short early accuracy 1.000).
        assert fx["early_acc"] >= 0.90
        assert sh["early_acc"] >= fx["early_acc"] - 0.05
