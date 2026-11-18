"""CPU gate for the CCS confound-suppression claim (arXiv:2610.14025):
projecting a harvested steering direction onto a closed-form
contrastive-covariance subspace removes a planted topic confound
that cripples the unprojected direction, recovering most of the gap
to an oracle that deletes the planted direction explicitly -- while
a random subspace of the same rank does not.

Paper: "Contrastive Covariance Subspaces for Activation Steering"
(Xenos, Plitsis, Foteinopoulou, Panagakis & Patras). CCS builds, for
a target class t, the grand mean mu_G over ALL class means, the
displacement delta_t = mu_t - mu_G, and the per-class scatters about
that common center, S_c = Sigma_c + delta_c delta_c^T (within-class
covariance plus the class's own displacement, Eq. 3). The
contrastive matrix (Eq. 5)

    A_t = S_t - (r / |N_t|) * sum_{c in N_t} S_c ,   r = 1,

rewards directions along which the target scatters and penalises
directions along which the nuisance classes scatter on average; its
top-k eigenvectors W_t span the subspace, P_t = W_t W_t^T, and the
steering vector is v_CCS = P_t delta_t (Eq. 6) -- one
eigendecomposition, no training. Headline results: best steering
method in all 12 model-domain settings across three LLMs, +48-70%
relative over the next-best method on AxBench, up to +151% over
difference-of-means. THE IN-PAPER TOY (Sec. 3.2, Fig. 3) is what
this test operationalises: oversampling one topic cluster to ~90%
of the harvest data makes the unprojected (grand-mean) direction
align with the planted topic direction u at cos 0.849 and drops its
AxBench score by 46% (0.565 -> 0.306); CCS stays near-orthogonal to
u (cos 0.071), loses only 12% (retains 88%), and recovers most of
the gap to an Oracle (0.583) that deletes u from the direction
using ground-truth topic labels CCS never sees.

Toy operationalization: synthetic class-conditional states in
R^24 with KNOWN latent structure, in a per-seed random orthonormal
basis Q. Two axes do the work:

  p -- the true concept axis (col 0 of Q): the target class mean is
       displaced along it and the ground-truth behavior readout is
       projection on p. The target class also VARIES along p
       (variance 9.0): target-concept variation is what CCS's
       objective rewards.
  u -- the planted topic / confound axis (col 1 of Q): it does not
       enter the behavior readout, but six nuisance classes vary
       strongly along it (variance 6.0, vs the target's 0.1), so u
       is a direction the target does NOT own but the nuisance pool
       does -- exactly the shared variation A_t penalises.

Classes (8 total): the target (mean +1.0 on p, variances above),
six nuisance classes (each with its own mean on a distinct
other axis, variance 6.0 on u, 0.1 on p) and one neutral class
(isotropic 0.4). All other variances 0.4. Two harvest conditions
per seed, N = 800 states per class and condition:

  balanced: target mean as above;
  skewed:   the planted confound -- the target mean additionally
            shifts by BETA = 1.6 along u (the toy form of the
            paper's ~90% single-topic oversampling), so the naive
            displacement delta_t is entangled with u.

Arms (all from the SAME skewed harvest; unit-normalised, so every
comparison is at matched steering norm):
  (a) naive:  delta_t = mu_t - mu_G, unprojected (the paper's GMC
      arm; plain difference-of-means against the grand mean);
  (b) CCS:    P_t delta_t with P_t from the top-K = 8 eigenvectors
      of A_t = S_t - mean_{c != t} S_c (r = 1, Eqs. 5-6 verbatim,
      scatters computed empirically about mu_G per Eq. 3);
  (c) oracle: delta_t - u (u^T delta_t), the planted direction
      deleted with ground-truth knowledge (paper's Oracle);
  (d) random: P_R delta_t, P_R a random rank-K projector (control:
      any rank-K projection shrinks and rotates the direction;
      only the contrastive one should remove u specifically).

Metrics per seed (>= 3 seeds), per arm and condition:
  concept gain:        v_hat . p (behavior readout shift at unit
                       steering norm);
  confound displacement: |v_hat . u| (off-target movement along
                       the planted topic axis);
  steering efficacy:   gain / sqrt(1 - gain^2) -- concept gain per
                       unit total off-target displacement;
  skew loss:           1 - gain_skewed / gain_balanced, the paper's
                       headline degradation number;
  gap recovery (CCS):  (gain_CCS - gain_naive) /
                       (gain_oracle - gain_naive) on the skewed
                       harvest.

Measured on pop-os (particlegan 0.10.1): skewed-harvest confound
cosine 0.827-0.864 (naive) vs 0.110-0.136 (CCS) vs 0.000 (oracle)
vs 0.277-0.548 (random); concept gain skewed 0.473-0.521 (naive),
0.861-0.933 (CCS), 0.912-0.939 (oracle), 0.361-0.493 (random);
skew loss 45-49% (naive, paper: 46%) vs 3-11% (CCS, paper: 12%);
CCS gap recovery 0.88-0.99; efficacy skewed 0.54-0.61 (naive),
1.69-2.59 (CCS), 0.39-0.57 (random). Assertions below encode
exactly these margins, no more.

What is NOT asserted: the paper's LLM benchmark numbers (AxBench /
Emotion / BIG5 / TruthfulQA scores, the +48-70% and +151% relative
improvements, the human study, multi-turn results) -- the toy has
no language model, judge or generation; the balanced-vs-skewed
score here is a pure readout projection, not the paper's judged
AxBench score, so only the toy's margin STRUCTURE (naive loses
roughly half, CCS loses roughly a tenth, CCS near-oracle and
near-orthogonal to the confound) is claimed, not the exact
0.565/0.306/0.543 point values. The paper's k = 32 / tuned-r
settings are not used: the toy is faithful at K = 8, r = 1 on
R^24, and no claim is made that these transfer to real residual
streams. Robustness to nuisance-pool size (paper Sec. 4.4.2) is
not asserted -- the pool is fixed at six nuisance classes.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI). Pure numpy linear
algebra -- no GAN machinery, because the claim is about harvested
directions, not training dynamics.
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


D = 24             # state dimension
K = 8              # CCS subspace rank (paper uses k = 32 on real models)
R = 1.0            # contrastive weight (paper's default, Eq. 5)
BETA = 1.6         # planted topic shift along u in the skewed harvest
N_PER = 800        # states per class per condition
N_NUIS = 6         # nuisance classes (plus target + one neutral = 8)
VAR_T = (9.0, 0.1, 0.4)    # target variances along (p, u, other)
VAR_N = (0.1, 6.0, 0.4)    # nuisance variances along (p, u, other)
VAR_O = 0.4                # neutral class / residual variance


def _unit(x):
    return x / np.linalg.norm(x)


def _variances(spec):
    v = np.full(D, spec[2])
    v[0], v[1] = spec[0], spec[1]
    return v


def run_seed(seed):
    rng = np.random.default_rng(11000 + seed)
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    p, u = q_basis[:, 0], q_basis[:, 1]

    # Class means in Q coordinates and per-class variances.
    mus, specs = [], []
    m0 = np.zeros(D)
    m0[0] = 1.0
    mus.append(m0)
    specs.append(_variances(VAR_T))
    for i in range(N_NUIS):
        m = np.zeros(D)
        m[2 + i] = 1.0            # each nuisance owns a distinct axis
        mus.append(m)
        specs.append(_variances(VAR_N))
    mus.append(np.zeros(D))       # neutral class
    specs.append(np.full(D, VAR_O))
    n_classes = len(mus)

    def harvest(skewed):
        means = [m.copy() for m in mus]
        if skewed:
            means[0][1] += BETA   # plant the topic confound
        return [(means[i] + rng.standard_normal((N_PER, D))
                 * np.sqrt(specs[i])) @ q_basis.T
                for i in range(n_classes)]

    def arms(data):
        class_means = np.array([x.mean(0) for x in data])
        mu_g = class_means.mean(0)
        delta = class_means[0] - mu_g
        # Scatters about the grand mean (Eq. 3) and the contrastive
        # matrix A_t = S_t - r * mean_{c != t} S_c (Eq. 5).
        scatters = []
        for x in data:
            z = x - mu_g
            scatters.append(z.T @ z / len(x))
        a_t = scatters[0] - R * np.mean(scatters[1:], axis=0)
        _, eigvecs = np.linalg.eigh(a_t)
        w_t = eigvecs[:, -K:]
        proj = w_t @ w_t.T
        wrand, _ = np.linalg.qr(rng.standard_normal((D, K)))
        proj_r = wrand @ wrand.T
        return dict(
            naive=_unit(delta),
            ccs=_unit(proj @ delta),
            oracle=_unit(delta - u * (u @ delta)),
            rand=_unit(proj_r @ delta),
            proj_p=float(np.linalg.norm(proj @ p)),
            proj_u=float(np.linalg.norm(proj @ u)),
        )

    def score(vec):
        gain = float(vec @ p)
        off = float(np.sqrt(max(1e-12, 1.0 - gain ** 2)))
        return dict(gain=gain, conf=abs(float(vec @ u)), eff=gain / off)

    bal, skw = arms(harvest(False)), arms(harvest(True))
    out = {}
    for name in ("naive", "ccs", "oracle", "rand"):
        b, s = score(bal[name]), score(skw[name])
        out[name] = dict(bal=b, skw=s, loss=1.0 - s["gain"] / b["gain"])
    out["proj_p"], out["proj_u"] = skw["proj_p"], skw["proj_u"]
    return out


def test_ccs_suppresses_planted_topic_confound():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        naive, ccs = r["naive"], r["ccs"]
        oracle, rand = r["oracle"], r["rand"]

        # Trap guard: the plant must genuinely entangle the naive
        # direction -- balanced gain high, skewed confound alignment
        # at the paper's level (GMC cos with u: 0.849).
        assert naive["bal"]["gain"] > 0.9, (
            f"seed {seed}: balanced naive gain {naive['bal']['gain']:.3f}")
        assert naive["skw"]["conf"] > 0.7, (
            f"seed {seed}: naive confound cosine {naive['skw']['conf']:.3f} "
            f"-- the planted topic did not entangle the naive direction")

        # The skew cripples the unprojected direction (paper: -46%).
        assert naive["loss"] > 0.35, (
            f"seed {seed}: naive skew loss {naive['loss']:.3f}")
        assert naive["skw"]["gain"] < 0.6, (
            f"seed {seed}: naive skewed gain {naive['skw']['gain']:.3f}")

        # CCS stays near-orthogonal to the confound (paper: 0.071)
        # and loses only a small fraction under skew (paper: 12%).
        assert ccs["skw"]["conf"] < 0.25, (
            f"seed {seed}: CCS confound cosine {ccs['skw']['conf']:.3f}")
        assert ccs["skw"]["conf"] < naive["skw"]["conf"] - 0.5, (
            f"seed {seed}: CCS confound {ccs['skw']['conf']:.3f} vs "
            f"naive {naive['skw']['conf']:.3f}")
        assert ccs["loss"] < 0.20, (
            f"seed {seed}: CCS skew loss {ccs['loss']:.3f}")
        assert ccs["loss"] < naive["loss"] - 0.25, (
            f"seed {seed}: skew loss CCS {ccs['loss']:.3f} vs naive "
            f"{naive['loss']:.3f}")

        # CCS recovers most of the gap to the oracle, which deletes
        # the planted direction with ground-truth labels.
        assert oracle["skw"]["conf"] < 0.01, (
            f"seed {seed}: oracle confound {oracle['skw']['conf']:.3f} "
            f"-- oracle must delete u exactly by construction")
        assert ccs["skw"]["gain"] > 0.8, (
            f"seed {seed}: CCS skewed gain {ccs['skw']['gain']:.3f}")
        assert ccs["skw"]["gain"] > oracle["skw"]["gain"] - 0.10, (
            f"seed {seed}: CCS gain {ccs['skw']['gain']:.3f} vs oracle "
            f"{oracle['skw']['gain']:.3f}")
        recovery = ((ccs["skw"]["gain"] - naive["skw"]["gain"])
                    / (oracle["skw"]["gain"] - naive["skw"]["gain"]))
        assert recovery > 0.75, (
            f"seed {seed}: CCS gap recovery {recovery:.3f}")

        # Efficacy at matched norm: concept gain per unit off-target
        # displacement -- CCS several-fold the naive direction.
        assert ccs["skw"]["eff"] > 1.4, (
            f"seed {seed}: CCS efficacy {ccs['skw']['eff']:.3f}")
        assert ccs["skw"]["eff"] > 2.0 * naive["skw"]["eff"], (
            f"seed {seed}: efficacy CCS {ccs['skw']['eff']:.3f} vs "
            f"naive {naive['skw']['eff']:.3f}")

        # Random-subspace control of the same rank must NOT match
        # CCS: projection alone is not the mechanism.
        assert rand["skw"]["gain"] < ccs["skw"]["gain"] - 0.25, (
            f"seed {seed}: random gain {rand['skw']['gain']:.3f} vs "
            f"CCS {ccs['skw']['gain']:.3f}")
        assert rand["skw"]["eff"] < 0.8, (
            f"seed {seed}: random efficacy {rand['skw']['eff']:.3f}")
        assert ccs["skw"]["eff"] > rand["skw"]["eff"] + 0.8, (
            f"seed {seed}: efficacy CCS {ccs['skw']['eff']:.3f} vs "
            f"random {rand['skw']['eff']:.3f}")


def test_contrastive_subspace_keeps_concept_drops_confound():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The subspace itself, before any direction is projected:
        # it retains the concept axis (the target varies along it)
        # and excludes the confound axis (only nuisances vary
        # along it) -- the contrastive objective in miniature.
        assert r["proj_p"] > 0.8, (
            f"seed {seed}: ||P_t p|| {r['proj_p']:.3f} -- the "
            f"contrastive subspace should retain the concept axis")
        assert r["proj_u"] < 0.3, (
            f"seed {seed}: ||P_t u|| {r['proj_u']:.3f} -- the "
            f"contrastive subspace should exclude the confound axis")
