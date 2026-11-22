"""CPU gate for the balloon-effect claim of Kim & Choi,
"A balloon effect: ban one channel and a hidden trait
finds another" (arXiv:2611.02509). The paper's claim
has three parts. (1) Closing a trait's expression
outlets one by one does NOT remove the trait: across
two models, six output formats and 16 traits, banning
the trait in one channel relocated its expression to
wherever the format still allowed it -- squeeze one
side of the balloon and the trait bulges out of
another (the balloon effect). (2) The relocation is
not removal in disguise: the trait's underlying
loading survives the bans, recoverable from the
channels that were never banned. (3) Acting at the
writer does remove it: decomposing the transfer to
single heads/neurons, patching the circuit that
WRITES the trait into the teacher's output stopped
the chains, sharply reduced the overt leak and halved
trait transfer to a student, while the same number of
random heads gave no detectable reduction.
Provenance: cycle 19 of the Particle Sliders paper
watch. This cycle arbitrates a tension inside the
watch's own ledger: cycle 18's Tang & Yen imprinting
salvage (beside PR #275) says excluding a sparse
divergence-token set reduces transfer, while the
balloon paper says surface bans relocate rather than
remove. The toy below separates the SCREEN (expression
in the banned channel) from the LOADING (a
cross-channel trait probe), the same split as PR #286.

Toy operationalization: the smallest categorical
multi-channel world in which the three parts can be
decided separately. K = 4 output channels (formats),
vocabulary V = 16 tokens, of which the first 4 are
trait-preferred. The base distribution is uniform, so
the base trait mass per channel is 4/16 = 0.25 and the
base trait log-odds displacement is 0. The TEACHER's
per-channel distribution boosts trait-preferred tokens
by exp(alpha_c): the per-channel trait displacement in
logit space is alpha_c. The teacher's TOTAL trait
drive A = sum_c alpha_c = 4.8 is conserved -- and this
conservation, stated plainly, is the paper's mechanism
built into the TEACHER, not the student: the teacher's
trait drive does not shrink when an outlet closes, it
redistributes over the outlets that remain. Baseline:
alpha_c = 1.2 on every channel. When channels are
banned, alpha is 0 on banned channels and A / (K - b)
on each unbanned channel (ban-1: 1.6; ban-2: 2.4).

The STUDENT is a channel-conditional categorical
model -- a per-channel softmax whose logits are the
log smoothed empirical frequencies of its training
samples (add-0.5 smoothing) -- distilled on N = 6000
teacher samples per channel and retrained from scratch
per arm. Ban protocol: the teacher's training outputs
in banned channels are FILTERED -- every
trait-expressing token there is removed from the
training set (the set shrinks; nothing is resampled) --
on top of the teacher-side redistribution above.

Readouts, kept strictly separate (the PR #286 split):
  (a) SCREEN: the student's trait-expression rate per
      channel, i.e. its softmax mass on trait-preferred
      tokens. The screen is what a channel-local audit
      sees.
  (b) LOADING: a cross-channel probe -- per channel,
      the trait displacement d_c = mean trait-token
      logit - mean non-trait-token logit of the student
      (readable in the student's own coordinates, per
      PR #273's law: baseline d_c ~= alpha_c); the
      loading is the SUM of d_c over UNBANNED channels.
      Under the conserved teacher drive this sum is
      A = 4.8 at baseline, and stays A under bans if --
      and only if -- the trait relocates instead of
      being removed.

Arms and controls:
  (1) no-ban baseline: alpha = 1.2 everywhere, no
      filtering;
  (2) ban-1: channel 0 banned (alpha redistributed to
      1.6 on channels 1-3, trait tokens filtered out of
      channel 0's training samples);
  (3) ban-2: channels 0+1 banned (alpha 2.4 on channels
      2-3, trait tokens filtered in both);
  (4) irrelevant-ban control: channel 0's training
      samples lose the same NUMBER of tokens as ban-1
      removed there, but the removed tokens are
      trait-IRRELEVANT (drawn proportionally from the
      non-trait tokens; the teacher's alphas are
      untouched, because no trait outlet was closed, so
      there is nothing to redistribute) -- matched
      filtering volume, no trait outlet closed;
  (5) writer-cut arm: the teacher is regenerated with
      the trait component removed at the source
      (alpha = 0 on every channel -- the paper's
      circuit-patch analogue) -- transfer should
      actually fall.

Measured on pop-os (particlegan 0.10.1), 5 seeds
(baseline loading / ban-1 loading ratio / ban-1
unbanned screen rise / ban-2 loading ratio / ban-2
unbanned screen rise / irrelevant unbanned rise /
writer-cut loading ratio):
  seed 0: 4.859 / 0.996 / +0.095 / 0.996 / +0.257 /
          -0.005 / -0.018
  seed 1: 4.627 / 1.047 / +0.111 / 1.044 / +0.269 /
          +0.007 / +0.016
  seed 2: 4.754 / 1.008 / +0.097 / 1.020 / +0.260 /
          -0.001 / -0.001
  seed 3: 4.728 / 1.015 / +0.104 / 1.011 / +0.266 /
          +0.005 / +0.004
  seed 4: 4.772 / 1.006 / +0.100 / 1.013 / +0.263 /
          +0.003 / -0.002
Per-channel screens behind those rows: baseline
0.506-0.533 on every channel; banned channels 0.000
(the add-0.5 smoothing floor, ~0.0004) under both
bans; ban-1 unbanned channels 0.612-0.635; ban-2
unbanned channels 0.775-0.794; writer-cut channels
0.241-0.262 (the 0.25 base trait mass -- the trait is
gone, not hidden). The irrelevant control's own
filtered channel inflates to 0.690-0.712 -- removing
non-trait tokens mechanically raises that channel's
trait share -- which is exactly why the control is
read on the UNBANNED channels, where it shows no
relocation: unbanned screen rise -0.005..+0.007 and
unbanned displacement within 0.035 of baseline on
every seed. Ban-1 filtered 1,409-1,570 trait tokens
out of channel 0; the control removed the same count
of non-trait tokens from the same channel.

The arbitration, stated plainly: all three parts of
the claim reproduce at toy scale, and the screen /
loading split is what makes them visible. The screen
says the bans worked perfectly -- banned channels at
0.000 -- while the loading says nothing was removed:
the cross-channel probe over unbanned channels
recovers 0.996-1.047 of the baseline loading under
ban-1 and 0.996-1.044 under ban-2, i.e. the conserved
teacher drive reappears in full wherever an outlet
remains, and the unbanned screen rises to show it
(+0.095..+0.111 under ban-1, +0.257..+0.269 under
ban-2). The irrelevant-ban control, at matched
filtering volume, produces none of this. Only the
writer-cut arm's loading falls (ratio -0.018..+0.016,
a 4.7-4.9 margin below every ban arm): acting where
the trait is written removes it; acting where it is
expressed relocates it. Assertions below encode
exactly these margins with modest headroom, no more.
If the toy had shown loading genuinely falling under
bans (no balloon), the honest report would have been
a non-reproduction; the toy was not tuned to fake the
effect -- the only place the effect is built in is
the teacher's conserved trait drive, which IS the
paper's proposed mechanism, and it is built into the
teacher, not the student.

What is NOT asserted: the paper's exact numbers. The
toy does not reproduce "halved trait transfer" as a
value -- the writer-cut loading falls to ~0 here
because the toy's patch removes the whole trait
component, a cleaner cut than any real circuit patch.
Nothing here is asserted about real LM channels, where
formats are not independent categorical outlets, trait
drive conservation is an empirical question rather
than a construction, and filtering a channel's outputs
during fine-tuning is not the same operation as
filtering a static distillation corpus. Two modeling
gifts, stated plainly: the trait direction is known
exactly (the probe reads the planted trait tokens
directly, where the paper must recover traits
behaviorally), and the teacher's redistribution is
exact and instantaneous, where a real teacher under a
ban would redistribute only as far as its own training
dynamics carry it. Pure numpy -- no GAN machinery,
because the claim is about transfer under channel
bans, not training dynamics. Skips cleanly when the
installed particlegan does not provide the EMPF recipe
(the house guard for this paper-test family; PR #211
is what enables this test in CI).
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


K = 4                # output channels (formats)
V = 16               # vocabulary size
N_TRAIT = 4          # trait-preferred tokens (indices 0..3)
N_PER_CHANNEL = 6000  # teacher samples per channel per arm
ALPHA_BASE = 1.2     # baseline per-channel trait boost
TOTAL_DRIVE = K * ALPHA_BASE  # conserved teacher trait drive A
SMOOTH = 0.5         # student add-smoothing
SEED_BASE = 41000    # RNG stream base
SEEDS = (0, 1, 2, 3, 4)

# Arm indices (fixed draw order per seed).
ARM_BASELINE, ARM_BAN1, ARM_BAN2, ARM_IRRELEVANT, ARM_WRITER = range(5)


def _teacher_probs(alpha):
    w = np.ones(V)
    w[:N_TRAIT] = np.exp(alpha)
    return w / w.sum()


def _train_student(counts):
    """Channel-conditional categorical student: per-channel
    softmax over log smoothed empirical frequencies."""
    probs = (counts + SMOOTH) / (counts.sum(axis=1, keepdims=True)
                                 + V * SMOOTH)
    logits = np.log(probs)
    screen = probs[:, :N_TRAIT].sum(axis=1)
    disp = (logits[:, :N_TRAIT].mean(axis=1)
            - logits[:, N_TRAIT:].mean(axis=1))
    return screen, disp


def _sample_counts(rng, alphas):
    counts = np.zeros((K, V), dtype=float)
    for c in range(K):
        counts[c] = rng.multinomial(N_PER_CHANNEL,
                                    _teacher_probs(alphas[c]))
    return counts


def _filter_trait(counts, channels):
    """Remove every trait-expressing token in the given
    channels. Returns (filtered counts, tokens removed
    in the first listed channel)."""
    out = counts.copy()
    removed_first = 0.0
    for i, c in enumerate(channels):
        removed = out[c, :N_TRAIT].sum()
        if i == 0:
            removed_first = removed
        out[c, :N_TRAIT] = 0.0
    return out, removed_first


def _filter_irrelevant(counts, channel, n_remove):
    """Remove n_remove trait-irrelevant tokens from one
    channel, proportionally across the non-trait tokens
    (matched filtering volume, no trait outlet closed)."""
    out = counts.copy()
    nontrait = out[channel, N_TRAIT:]
    total = nontrait.sum()
    remove = np.round(n_remove * nontrait / total)
    # Fix rounding so the removed total matches exactly.
    remove[-1] += n_remove - remove.sum()
    out[channel, N_TRAIT:] = nontrait - remove
    return out


def run_seed(seed):
    """One seed: all five arms, each with its own RNG
    stream, the student retrained from scratch per arm.

    Draw order is fixed -- per arm (baseline, ban-1,
    ban-2, irrelevant, writer), per channel -- so every
    readout of a seed comes from the same draws. The
    irrelevant arm's removal count is ban-1's channel-0
    trait count, so the two arms filter the same volume
    from the same channel.
    """
    def arm_rng(arm):
        return np.random.default_rng(SEED_BASE + seed * 1000
                                     + arm * 17)

    results = {}

    # (1) No-ban baseline.
    counts = _sample_counts(arm_rng(ARM_BASELINE),
                            [ALPHA_BASE] * K)
    screen, disp = _train_student(counts)
    results["baseline"] = {"screen": screen, "disp": disp,
                           "loading": float(disp.sum())}

    # (2) Ban-1: channel 0 closed; drive redistributed.
    a_ban1 = TOTAL_DRIVE / (K - 1)
    counts = _sample_counts(arm_rng(ARM_BAN1),
                            [0.0, a_ban1, a_ban1, a_ban1])
    counts, removed_ban1 = _filter_trait(counts, [0])
    screen, disp = _train_student(counts)
    results["ban1"] = {"screen": screen, "disp": disp,
                       "loading": float(disp[1:].sum()),
                       "removed": float(removed_ban1)}

    # (3) Ban-2: channels 0+1 closed; drive redistributed.
    a_ban2 = TOTAL_DRIVE / (K - 2)
    counts = _sample_counts(arm_rng(ARM_BAN2),
                            [0.0, 0.0, a_ban2, a_ban2])
    counts, _ = _filter_trait(counts, [0, 1])
    screen, disp = _train_student(counts)
    results["ban2"] = {"screen": screen, "disp": disp,
                       "loading": float(disp[2:].sum())}

    # (4) Irrelevant-ban control: same channel, same
    # removed volume, trait-irrelevant tokens only;
    # teacher alphas untouched (nothing redistributes).
    counts = _sample_counts(arm_rng(ARM_IRRELEVANT),
                            [ALPHA_BASE] * K)
    counts = _filter_irrelevant(counts, 0, removed_ban1)
    screen, disp = _train_student(counts)
    results["irrelevant"] = {"screen": screen, "disp": disp,
                             "loading": float(disp[1:].sum())}

    # (5) Writer-cut: trait removed at the source.
    counts = _sample_counts(arm_rng(ARM_WRITER), [0.0] * K)
    screen, disp = _train_student(counts)
    results["writer"] = {"screen": screen, "disp": disp,
                         "loading": float(disp.sum())}

    return results


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_banned_screen_collapses_but_unbanned_screen_rises():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        base = res["baseline"]["screen"]
        # The screen works where the ban points: the
        # banned channels' trait-expression rate is at
        # the smoothing floor (demonstrated ~0.0004 on
        # every seed).
        assert res["ban1"]["screen"][0] <= 0.01, (
            f"seed {seed}: ban-1 banned screen "
            f"{res['ban1']['screen'][0]:.4f}")
        assert res["ban2"]["screen"][0] <= 0.01, (
            f"seed {seed}: ban-2 banned screen[0] "
            f"{res['ban2']['screen'][0]:.4f}")
        assert res["ban2"]["screen"][1] <= 0.01, (
            f"seed {seed}: ban-2 banned screen[1] "
            f"{res['ban2']['screen'][1]:.4f}")
        # ... and the trait bulges out of the channels
        # that remain: the mean unbanned screen RISES
        # against the same channels at baseline
        # (demonstrated +0.095..+0.111 for ban-1,
        # +0.257..+0.269 for ban-2).
        rise1 = (res["ban1"]["screen"][1:].mean()
                 - base[1:].mean())
        assert rise1 >= 0.06, (
            f"seed {seed}: ban-1 unbanned rise {rise1:.3f}")
        rise2 = (res["ban2"]["screen"][2:].mean()
                 - base[2:].mean())
        assert rise2 >= 0.18, (
            f"seed {seed}: ban-2 unbanned rise {rise2:.3f}")


def test_loading_is_conserved_under_bans():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        base_loading = res["baseline"]["loading"]
        # The cross-channel probe over unbanned channels
        # recovers the full baseline loading under both
        # bans (demonstrated ratios 0.996-1.047): bans
        # relocate the
        # trait, they do not eliminate it. The floor is
        # the assertion that matters; the ceiling keeps
        # "conserved" from passing as "amplified".
        for arm in ("ban1", "ban2"):
            ratio = res[arm]["loading"] / base_loading
            assert ratio >= 0.85, (
                f"seed {seed}: {arm} loading ratio "
                f"{ratio:.3f}")
            assert ratio <= 1.20, (
                f"seed {seed}: {arm} loading ratio "
                f"{ratio:.3f}")


def test_irrelevant_ban_no_relocation_and_writer_cut_removes():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        base = res["baseline"]
        # Irrelevant-ban control: matched filtering
        # volume in the same channel, no trait outlet
        # closed -- the unbanned channels show NO
        # relocation pattern (demonstrated rise
        # -0.005..+0.007, displacement within 0.035 of
        # baseline).
        rise_irr = (res["irrelevant"]["screen"][1:].mean()
                    - base["screen"][1:].mean())
        assert abs(rise_irr) <= 0.02, (
            f"seed {seed}: irrelevant unbanned rise "
            f"{rise_irr:.3f}")
        disp_irr = res["irrelevant"]["disp"][1:].mean()
        disp_base = base["disp"][1:].mean()
        assert abs(disp_irr - disp_base) <= 0.10, (
            f"seed {seed}: irrelevant unbanned disp "
            f"{disp_irr:.3f} vs baseline {disp_base:.3f}")
        # Writer-cut: removing the trait at the source
        # is the one arm whose loading actually falls --
        # materially below baseline (demonstrated ratio
        # -0.018..+0.016) and below every ban arm by a
        # wide margin (demonstrated 4.7-4.9).
        writer_ratio = (res["writer"]["loading"]
                        / base["loading"])
        assert writer_ratio <= 0.20, (
            f"seed {seed}: writer loading ratio "
            f"{writer_ratio:.3f}")
        for arm in ("ban1", "ban2"):
            margin = res[arm]["loading"] - res["writer"]["loading"]
            assert margin >= 3.0, (
                f"seed {seed}: {arm} - writer loading "
                f"margin {margin:.3f}")
