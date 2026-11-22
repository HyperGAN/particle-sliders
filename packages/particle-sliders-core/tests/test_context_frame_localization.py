"""CPU gate for the context-localization claim of Nief, Fu,
Muchane & Holtzman, "Subliminal Learning is a LoRA Artifact"
(arXiv:2606.00831): subliminal behavior is localized to
computation at token positions seen during BOTH fine-tuning and
evaluation -- the shared context / template tokens (e.g. the
default system prompt). The paper's abstract states the two
halves gated here: a Qwen model fine-tuned with its default
system prompt shows no subliminal learning at generation when no
system prompt is included, and dynamic weight grafting --
selectively turning OFF the LoRA adapters at the shared-context
positions -- removes the effect (localization material in the
paper's Sections 5.1 and 4.2). Provenance chain: the claim was
logged as a deployment caveat in this watch's cycle-13 brief via
the RegAmp (arXiv:2611.10397, PR #250) reference chain, with a
toy candidate noted for a future cycle; this is that test
(cycle 14).

Why it is load-bearing for Particle Sliders: the deployment
premise is context-free behavior -- sliders train under an
eliciting context and deploy with NO instruction. If internalized
behavior expresses preferentially through the training frame's
positions, deployment evals that omit the frame mismeasure, and
the harvest frame is load-bearing metadata.

Toy operationalization: a sequence toy with the minimal
structure the claim needs and nothing that hard-wires its
conclusion. Sequences are a FRAME block (LF = 8 tokens, the same
values in every training example -- the shared context) followed
by CONTENT (LC = 6 carrier tokens, freshly drawn per example).
Frame tokens are a shared direction mu_f plus per-token noise;
content tokens are a shared direction mu_c plus larger per-token
noise, so the frame is the CONSTANT part of every input and the
content the varying part -- the distributional fact that defines
a shared context, stated openly. The student is a frozen uniform
pool (output = sum over positions) plus ONE LoRA-style adapter,
delta_t = B A x_t with rank 4, the SAME (A, B) at every position:
the adapter touches frame and content positions identically and
pooling weights every position equally, so no position is
privileged by the architecture. Training distills a teacher's
pooled output on 128 carrier sequences presented WITH frame F:
the target is the base pooled output plus a constant trait shift
Delta (||Delta|| = 1.5; the trait teacher) or plus nothing (the
clean teacher). Only (A, B) train (Adam, 400 full-batch steps,
B initialized at zero, the standard LoRA init).

Readout: on 256 fresh neutral probes, trait expression is the
adapter-induced shift projected onto Delta, normalized by
||Delta||^2 -- 1.0 is full expression of the installed trait,
0 is none. Five eval conditions per seed:
  (i)   matched      frame F present, adapter on everywhere;
  (ii)  absent       content only, no frame;
  (iii) swapped      a DIFFERENT frame F' (same length, fresh
                     tokens, same norm structure) + content;
  (iv)  abl_frame    F present, adapter contribution zeroed at
                     FRAME positions only (the grafting move);
  (v)   abl_content  F present, adapter contribution zeroed at
                     CONTENT positions only (control ablation).

What the toy shows, mechanically: gradient descent puts the
adapter's trait pathway on the input's constant component,
because the trait target is constant across carriers while the
content varies -- the frame sum is the reliable carrier of a
constant shift, so the learned BA aligns with the mean sequence
sum, which the frame dominates. Expression then lives wherever
that pathway is exercised at eval: remove the frame, swap it, or
gate the adapter off at frame positions and the pathway is not
exercised; gate it off at content positions and it still is.
Localization here is a consequence of the data statistics plus a
position-shared adapter, not of any position-specific parameter.

Controls, per seed:
  (i) clean teacher: Delta = 0. The adapter's gradient is
      exactly zero from the zero init, so expression is exactly
      0 in all five conditions -- no trait, no expression,
      anywhere;
  (ii) content-only apparatus arm: a student trained on CONTENT
      ONLY (no frame in training) reaches 0.82-0.99 expression
      when evaluated content-only. The content route CAN carry
      the trait in this architecture when it is the route
      training exercises -- the frame-trained student's failure
      to express without its frame is a fact about where its
      training put the pathway, not an architectural incapacity.

Measured on pop-os (particlegan 0.10.1), 3 seeds, expression
(fraction of Delta recovered along the trait direction, mean
over 256 probes; per-probe SD 0.13-0.17 in the frame-trained
arms, so probe-mean standard errors are ~0.01):
  matched      0.979-1.008
  absent       +0.025..+0.090
  swapped      +0.011..+0.066
  abl_frame    +0.025..+0.090 (identical to absent -- see the
               limitation below)
  abl_content  0.901-0.954
  clean        exactly 0 in every condition
  content-only-trained, evaluated content-only: 0.820-0.991
Assertions below encode exactly these margins with headroom,
no more: matched > 0.90; absent and swapped < 0.20 and at least
0.75 below matched; frame ablation < 0.20 (it removes at least
three quarters of the matched expression); content ablation
> 0.80 and at least 0.70 above the frame ablation; clean
|expression| < 1e-6 everywhere; content-only apparatus > 0.70.

What is NOT asserted: anything about real chat templates,
attention, or tokenizers -- the toy's frame is a position block
in a pooled linear sequence model, not a chat template; the
claim tested is the localization STRUCTURE (shared-position
computation carries the installed behavior), not the paper's
LM-scale numbers. One structural limitation, recorded rather
than hidden: in a purely pooled readout, conditions (ii) and
(iv) coincide numerically (both leave the adapter acting on the
content sum alone), so the toy cannot separate "frame absent"
from "adapter off at frame positions" the way a transformer --
whose downstream processing is position-dependent -- could;
what (iv) vs (v) does isolate, and what is asserted, is that
the adapter's eval-time contribution AT the shared positions
carries the effect. Pure torch linear algebra -- no GAN
machinery, because the claim is about where a distilled
adapter's computation is exercised, not training dynamics.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI).
"""
import functools

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


D = 64                 # token-space dimension
LF = 8                 # frame (shared-context) positions
LC = 6                 # content (carrier) positions
RANK = 4               # LoRA rank of the student's adapter
TRAIT_NORM = 1.5       # ||Delta||, the trait teacher's shift
N_TRAIN = 128          # carrier sequences per training run
N_PROBE = 256          # neutral probes per eval condition
STEPS = 400            # full-batch Adam steps
LR = 0.05
CONDITIONS = ("matched", "absent", "swapped", "abl_frame", "abl_content")


def _unit(rng):
    v = torch.randn(D, generator=rng)
    return v / v.norm()


def _make_world(seed):
    rng = torch.Generator().manual_seed(31000 + seed)
    mu_f = _unit(rng)
    mu_fp = _unit(rng)
    mu_c = _unit(rng)
    delta = TRAIT_NORM * _unit(rng)
    frame = mu_f.unsqueeze(0) + 0.3 * torch.randn(LF, D, generator=rng)
    frame_p = mu_fp.unsqueeze(0) + 0.3 * torch.randn(LF, D, generator=rng)
    return rng, mu_c, delta, frame, frame_p


def _draw_content(rng, mu_c, n):
    return 0.5 * mu_c + 0.5 * torch.randn(n, LC, D, generator=rng)


def _train(world, trait, with_frame):
    """Train only the position-shared adapter (A, B); B init 0."""
    rng, mu_c, delta, frame, _ = world
    target = delta if trait else torch.zeros(D)
    content = _draw_content(rng, mu_c, N_TRAIN)
    if with_frame:
        seqs = torch.cat(
            [frame.unsqueeze(0).expand(N_TRAIN, -1, -1), content], dim=1)
    else:
        seqs = content
    sums = seqs.sum(dim=1)
    A = torch.nn.Parameter(0.1 * torch.randn(RANK, D, generator=rng))
    B = torch.nn.Parameter(torch.zeros(D, RANK))
    opt = torch.optim.Adam([A, B], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        adapter = (sums @ A.T) @ B.T
        loss = ((adapter - target) ** 2).mean()
        loss.backward()
        opt.step()
    return A.detach(), B.detach()


def _expression(params, world, mode):
    """Mean trait expression over fresh neutral probes."""
    A, B = params
    _, mu_c, delta, frame, frame_p = world
    prng = torch.Generator().manual_seed(99000)  # same probes, all modes
    content = _draw_content(prng, mu_c, N_PROBE)
    if mode == "absent":
        seqs = content
        mask = torch.ones(LC)
    elif mode == "swapped":
        seqs = torch.cat(
            [frame_p.unsqueeze(0).expand(N_PROBE, -1, -1), content], dim=1)
        mask = torch.ones(LF + LC)
    else:
        seqs = torch.cat(
            [frame.unsqueeze(0).expand(N_PROBE, -1, -1), content], dim=1)
        if mode == "matched":
            mask = torch.ones(LF + LC)
        elif mode == "abl_frame":
            mask = torch.cat([torch.zeros(LF), torch.ones(LC)])
        elif mode == "abl_content":
            mask = torch.cat([torch.ones(LF), torch.zeros(LC)])
        else:
            raise ValueError(mode)
    sums = (seqs * mask.view(1, -1, 1)).sum(dim=1)
    adapter = (sums @ A.T) @ B.T
    expr = (adapter @ delta) / (delta @ delta)
    return float(expr.mean())


@functools.lru_cache(maxsize=None)
def run_seed(seed):
    world = _make_world(seed)
    trait_params = _train(world, trait=True, with_frame=True)
    clean_params = _train(world, trait=False, with_frame=True)
    content_params = _train(world, trait=True, with_frame=False)
    return dict(
        trait={m: _expression(trait_params, world, m) for m in CONDITIONS},
        clean={m: _expression(clean_params, world, m) for m in CONDITIONS},
        content_only_absent=_expression(content_params, world, "absent"),
    )


def test_matched_frame_expression_dominates_absent_and_swapped():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)["trait"]
        # The trait is installed and expressed under the frame it
        # was trained with.
        assert r["matched"] > 0.90, (
            f"seed {seed}: matched-frame expression "
            f"{r['matched']:.3f}")
        # Train/eval context mismatch largely removes it: no frame
        # at eval, or a different frame, both collapse expression.
        for mode in ("absent", "swapped"):
            assert r[mode] < 0.20, (
                f"seed {seed}: {mode} expression {r[mode]:+.3f} -- "
                f"mismatched eval context should largely remove "
                f"the effect")
            assert r["matched"] > r[mode] + 0.75, (
                f"seed {seed}: matched {r['matched']:.3f} vs "
                f"{mode} {r[mode]:+.3f}")


def test_frame_ablation_removes_expression_content_ablation_does_not():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)["trait"]
        # Grafting move: adapter OFF at the shared (frame)
        # positions removes most of the expression.
        assert r["abl_frame"] < 0.20, (
            f"seed {seed}: frame-ablated expression "
            f"{r['abl_frame']:+.3f}")
        assert r["matched"] > r["abl_frame"] + 0.75, (
            f"seed {seed}: matched {r['matched']:.3f} vs "
            f"frame-ablated {r['abl_frame']:+.3f}")
        # Control ablation: adapter OFF at content positions
        # removes little -- the effect does not live there.
        assert r["abl_content"] > 0.80, (
            f"seed {seed}: content-ablated expression "
            f"{r['abl_content']:.3f}")
        assert r["abl_content"] > r["abl_frame"] + 0.70, (
            f"seed {seed}: content-ablated {r['abl_content']:.3f} "
            f"vs frame-ablated {r['abl_frame']:+.3f} -- the "
            f"localization contrast")


def test_clean_teacher_and_content_only_apparatus():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Clean teacher: no trait installed, no expression in any
        # eval condition.
        for mode in CONDITIONS:
            assert abs(r["clean"][mode]) < 1e-6, (
                f"seed {seed}: clean-teacher expression in "
                f"{mode} is {r['clean'][mode]:+.2e}")
        # Apparatus: the content route CAN carry the trait when
        # training exercises it -- a student trained on content
        # alone expresses strongly on content alone. The frame
        # student's localization is learned, not hard-wired.
        assert r["content_only_absent"] > 0.70, (
            f"seed {seed}: content-only-trained expression on "
            f"content-only eval {r['content_only_absent']:.3f} -- "
            f"the content route must be learnable in principle")
