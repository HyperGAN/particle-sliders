"""CPU gate for the steering-vector-as-training-signal claim
(arXiv:2610.40998): training WITH a steering direction applied to
hidden states during supervised training beats plain supervised
training at internalizing the behavior, the margin grows with the
off-policy gap between the training distribution and the student's
own distribution, and applying the same direction again at inference
-- which helps a plainly trained checkpoint -- degrades an
SV-trained one (double application breaks the intervention regime).

Paper: "Steering Vectors as a Training Signal in LLM Post-Training"
(Chen, Vyas, Yao & Wei). Steering vectors are normally inference-time
residual-stream additions; the paper instead applies the direction
through a hidden-state hook during teacher-forced supervised training
(and as advantage repair on wrong GRPO rollouts). SV injection during
supervised training improves over plain SFT and distillation, with
gains scaling with the off-policy gap, up to 12% when distilling from
thinking-mode teachers; among its failure boundaries, inference-time
application on an SV-trained checkpoint can break the intervention
regime. This gate exercises the supervised-regime core: (1) the
gap-scaling training benefit, and (2) the double-application
boundary. The GRPO advantage-repair regime is out of scope here.

Toy operationalization: a small student (one tanh hidden layer over
Gaussian states, dim 16) must internalize a behavior defined on a
CONTENT component it can only observe noisily: states are
s = c + 1.5 * eps, the label is y = 1[ a . c > 0 ] for a fixed unit
vector a, and the student's native distribution carries no other
signal. Teacher-mode training states are off-policy in a specific,
behavior-shaped way: s_T = c + 1.5 * eps + G * v + y * G * v, where v
is a unit vector orthogonal to a (the behavior / mode direction) and
G is the off-policy gap knob. The +G * v term is the teacher mode's
constant offset; the +y * G * v term is the LEAK -- in teacher-mode
traces, behavior-positive states sit further along the behavior
direction, so the direction itself is a teacher-only shortcut: a
readout that keys on the state's v-component (or on hidden units the
v-component drives) fits the teacher distribution without
internalizing the noisy content rule, and the shortcut strengthens
exactly as the gap grows. Plain SFT takes the shortcut in proportion
to G. The SV is harvested the usual way -- the normalized contrast
of hidden pre-activations between behavior-positive and
behavior-negative teacher states under the initial student -- and
the SV-hook arm trains with hidden pre-activations shifted by
ALPHA * SV. The shift drives the leak-carrying pre-activations
toward tanh saturation for BOTH classes, compressing the leak's
class differential (tanh(G + ALPHA) - tanh(ALPHA) << G) while the
content signal, spread over all units at smaller per-unit scale,
stays in the responsive regime -- so the hooked arm is forced onto
the content rule that the native distribution actually rewards.

Arms (same student, same steps, same teacher data stream):
  (a) plain:  no hook;
  (b) sv:     hook along the harvested contrast direction;
  (c) shuf:   hook along a coordinate-permuted SV (same norm);
  (d) rev:    hook along the reversed SV.
Evaluation is on the student's OWN distribution (native states, no
offset, no leak): behavior accuracy unhooked, and accuracy with the
+SV applied at inference (the double-application cell). A leak-FLIP
diagnostic scores each checkpoint on teacher-style states where the
leak attaches to the negative class instead: the less an arm relied
on the leak, the less the flip costs it.

Iteration log (the loop that produced this gate):
  1. Feedforward MLP, global linear rule, mean-shift gap: plain
     generalized across the shift (0.95-0.99 at every gap) and the
     hooked arm became hook-DEPENDENT (unhooked accuracy below
     plain, restored only with the hook back on) -- a global rule
     has no internalization gap, and a constant hook on top of it is
     pure compensation.
  2. GRU student, token-Markovian behavior rule, teacher-forced
     training / free-running eval: at gap >= 0.85 every arm
     saturated at 1.000; at gap 0.5 the SV arm was direction-
     specific but per-seed noise swamped the margins. A rule that
     lives entirely in the tokens leaves the hidden state nothing
     to internalize.
  3. State-space leak world with CLEAN linear content: plain did
     degrade with the gap (0.97 -> 0.88) and inference-time SV did
     repair plain -- but content was learned in the first steps by
     every arm, so the leak was redundant, not a shortcut, and the
     hook bought nothing.
  4. Quadrant (XOR-like) content: unlearnable in budget -- every
     arm at chance on native states; a shortcut can only starve a
     signal that is learnable-but-slow.
  5. Noisy content (sigma = 1.5) + hook-strength sweep: the SV arm
     beat plain (+0.026 at G = 1.5, ALPHA = 4) with the
     double-application pattern present, but shuffled and reversed
     hooks matched the SV arm on headline accuracy.
  6. Gap sweep + leak-flip diagnostic: the SV margin over plain
     grows with the gap (+0.007 / +0.026 / +0.052 on average at
     G = 0.75 / 1.5 / 3.0 in the sweep harness), every hooked arm
     shows reduced leak reliance under the flip, and re-applying
     +SV at inference degrades the SV-trained checkpoint in every
     seed while its effect on the plain checkpoint is positive on
     average.
  7. Assertion calibration against this file's own code path
     (single-threaded): two draft assertions outran the per-seed
     table. Plain's inference-SV gain is a mean effect (+0.019 /
     +0.016 per gap) with one seed at -0.007 (inside eval noise),
     not a per-seed gain; and the double-application harm is NOT
     cleanly specific against the controls -- the reversed-trained
     arm has one genuinely hurt cell (G = 1.5, seed 1, -0.059) --
     though the SV arm is the only one hurt in all six
     (gap, seed) cells. The gate asserts exactly that: per-seed
     harm for the SV arm, mean-level help for plain, and no
     specificity claim against the controls. What this toy does
     NOT reproduce is direction specificity of the training benefit
     itself: in an isotropic random MLP every large fixed shift
     saturates some leak-carrying units, so the shuffled/reversed
     hooks capture most of the benefit (the SV arm is within 0.027
     of the best control, sometimes below it). That failure is
     recorded as a limitation, not asserted away.

Measured on pop-os (particlegan 0.10.1), ALPHA = 4, this file's
code path: at G = 1.5, native accuracy SV 0.633-0.647 vs plain
0.607-0.611 (margins +0.022..+0.037 per seed); at G = 3.0, SV
0.557-0.571 vs plain 0.506-0.518 (+0.039..+0.065). Inference +SV:
plain +0.019 / +0.016 on average per gap (per-seed -0.007..+0.044);
SV-trained -0.028..-0.074 per seed. Leak flip at G = 3.0: hooked
arms keep +0.043..+0.116 more accuracy than plain. Skips cleanly
when the installed particlegan does not provide the EMPF recipe.
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


H = 16              # state / hidden dimension
NOISE = 1.5         # observation noise on the content (learnable-but-slow)
ALPHA = 4.0         # hook strength, training and inference
STEPS = 250
BATCH = 128
LR = 3e-3
N_EVAL = 4000
GAPS = (1.5, 3.0)   # off-policy gap levels gated here (0.75 in log)


def make_world(seed):
    """Content direction a and behavior direction v (v perp a)."""
    rng = np.random.default_rng(1000 + seed)
    a = rng.standard_normal(H)
    a /= np.linalg.norm(a)
    v = rng.standard_normal(H)
    v -= (v @ a) * a
    v /= np.linalg.norm(v)
    return (torch.tensor(a, dtype=torch.float32),
            torch.tensor(v, dtype=torch.float32))


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(H, H)
        self.l2 = nn.Linear(H, 1)
        with torch.no_grad():
            self.l1.weight.mul_(0.7)
            self.l1.bias.mul_(0.3)

    def forward(self, s, hook=None, alpha=0.0):
        pre = self.l1(s)
        if hook is not None:
            pre = pre + alpha * hook
        return self.l2(torch.tanh(pre)).squeeze(-1), pre


def gen(n, a, v, gap, gen_, teacher, flip=False):
    """Native states (teacher=False) or teacher-mode states with the
    mode offset and the class-conditional leak along v. flip=True
    attaches the leak to the negative class (diagnostic only)."""
    c = torch.randn(n, H, generator=gen_)
    y = (c @ a > 0).float()
    s = c + NOISE * torch.randn(n, H, generator=gen_)
    if teacher:
        leak = (1.0 - y) if flip else y
        s = s + gap * v + leak.unsqueeze(1) * gap * v
    return s, y


def harvest_sv(student, a, v, gap, seed):
    """Contrast direction: mean hidden pre-activation on
    behavior-positive minus behavior-negative teacher states, under
    the initial student."""
    g = torch.Generator().manual_seed(777 + seed)
    s, y = gen(4000, a, v, gap, g, True)
    with torch.no_grad():
        _, pre = student(s)
    d = pre[y == 1].mean(0) - pre[y == 0].mean(0)
    return d / d.norm()


def train_arm(mode, seed, gap):
    a, v = make_world(seed)
    torch.manual_seed(20000 + seed)
    student = Student()
    sv = harvest_sv(student, a, v, gap, seed)
    if mode == "shuf":
        gg = torch.Generator().manual_seed(999 + seed)
        hook = sv[torch.randperm(H, generator=gg)]
    elif mode == "rev":
        hook = -sv
    else:
        hook = sv
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    g = torch.Generator().manual_seed(30000 + seed)
    for _ in range(STEPS):
        s, y = gen(BATCH, a, v, gap, g, True)
        hk = hook if mode != "plain" else None
        logits, _ = student(s, hook=hk, alpha=ALPHA if hk is not None else 0.0)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return student, a, v, sv


@torch.no_grad()
def evaluate(student, a, v, gap, seed, teacher=False, flip=False,
             hook=None, alpha=0.0):
    g = torch.Generator().manual_seed(555 + seed)
    s, y = gen(N_EVAL, a, v, gap, g, teacher, flip)
    logits, _ = student(s, hook=hook, alpha=alpha)
    return float(((logits > 0).float() == y).float().mean())


def run_seed(seed, gap):
    out = {}
    for mode in ("plain", "sv", "shuf", "rev"):
        student, a, v, sv = train_arm(mode, seed, gap)
        out[mode] = {
            "native": evaluate(student, a, v, gap, seed),
            "native_sv": evaluate(student, a, v, gap, seed,
                                  hook=sv, alpha=ALPHA),
            "flip": evaluate(student, a, v, gap, seed,
                             teacher=True, flip=True),
        }
    return out


def test_sv_training_signal():
    _empf_recipe()
    margins = {}
    for gap in GAPS:
        margins[gap] = []
        plain_gain, sv_drop = [], []
        for seed in range(3):
            r = run_seed(seed, gap)
            plain, sv = r["plain"], r["sv"]
            margins[gap].append(sv["native"] - plain["native"])
            plain_gain.append(plain["native_sv"] - plain["native"])
            sv_drop.append(sv["native"] - sv["native_sv"])
            # (1) SV-hook training beats plain SFT on the student's
            # own distribution, per seed, at both gap levels
            # (demonstrated margins: +0.022..+0.037 at G = 1.5,
            # +0.039..+0.065 at G = 3.0).
            assert sv["native"] >= plain["native"] + 0.015
            # The SV arm is NOT above the direction controls on
            # headline accuracy in this toy (iterations 5-7): it is
            # within tolerance of the best of them (demonstrated:
            # SV - best control = -0.027..+0.006 across gaps/seeds).
            best_ctrl = max(r["shuf"]["native"], r["rev"]["native"])
            assert sv["native"] >= best_ctrl - 0.032
            # (2) Double application, SV arm: re-applying +SV at
            # inference degrades the SV-trained checkpoint in every
            # (gap, seed) cell (demonstrated drops 0.028..0.074).
            assert sv["native_sv"] <= sv["native"] - 0.02
            # (2b) Double application, plain arm: the same +SV never
            # hurts the plain checkpoint beyond eval noise
            # (demonstrated per-seed gains -0.007..+0.044).
            assert plain["native_sv"] >= plain["native"] - 0.01
            # Mechanism diagnostic: every hooked arm relies less on
            # the teacher-only leak than plain does -- under the
            # leak flip it keeps more accuracy (demonstrated at
            # G = 3.0: hook arms +0.043..+0.116 over plain per seed;
            # asserted only at the larger gap, where the leak is
            # strong enough for the diagnostic to bite).
            if gap == 3.0:
                for mode in ("sv", "shuf", "rev"):
                    assert r[mode]["flip"] >= plain["flip"] + 0.03
        # (2c) On average across seeds, inference +SV helps plain
        # (demonstrated mean gains +0.019 at G = 1.5, +0.016 at
        # G = 3.0) -- a mean effect, not a per-seed one (iter. 7).
        assert float(np.mean(plain_gain)) >= 0.01
        assert float(np.mean(sv_drop)) >= 0.025
    # (1b) The SV-over-plain margin grows with the off-policy gap,
    # per seed (demonstrated growth: +0.013..+0.043 per seed).
    for seed in range(3):
        assert margins[3.0][seed] >= margins[1.5][seed] + 0.008
