#!/usr/bin/env python3
"""Read-only CPU checks of the music GAN versus the pinned Gaussian kernels.

Counterexamples establish properties of the objective, not that the LoRA can
reach every constructed hidden state or that a particular property caused an
audible failure. No model, checkpoint, or running experiment is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conceptmod.textsliders.lm_adv import SpanTransformerD, cap_penalty, rp_d_loss, rp_g_loss
from conceptmod.textsliders.lm_gan import feature_mean_surrogate

REVISION = "b5ee35f9b24cf35a6b1346ba2f0856c4877aa336"
STATE = ROOT / "models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_state.pt"
SPANS = ROOT / "analysis/gan_bcap/critic_failure_probe_20260904.pt"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def maximum_error(a, b):
    return float((a.detach() - b.detach()).abs().max())


class ScaledPolynomial(nn.Module):
    in_mode = "scaled"

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("input_scale", torch.tensor(float(scale)))
        self.quadratic = nn.Parameter(torch.tensor([0.8, 1.2, -0.6]))
        self.linear = nn.Parameter(torch.tensor([2.0, -1.0, 1.5]))

    def forward(self, raw):
        x = raw / self.input_scale
        return (self.quadratic * x.square() + self.linear * x).sum(-1)


def kernel_parity(upstream):
    losses = module(upstream / "lib/gan_loss.py", "audit_upstream_losses")
    regularizers = module(upstream / "lib/grad_regularizers.py", "audit_upstream_regularizers")
    reference = losses.GANLoss(loss_type="logistic", mode="rp")
    real = torch.randn(7, requires_grad=True)
    fake = torch.randn(7, requires_grad=True)
    result = {}
    for name, ours, theirs in (
        ("discriminator", rp_d_loss(real, fake), reference.d_loss(real, fake)),
        ("generator", rp_g_loss(fake, real), reference.g_loss(fake, real)),
    ):
        a = torch.autograd.grad(ours, (real, fake), retain_graph=True)
        b = torch.autograd.grad(theirs, (real, fake), retain_graph=True)
        torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        result[name] = {"loss_error": maximum_error(ours, theirs),
                        "logit_gradient_error": max(maximum_error(x, y) for x, y in zip(a, b))}
    ours_d = ScaledPolynomial(13.0)
    ref_d = copy.deepcopy(ours_d)
    ref_d.input_scale.fill_(1.0)
    real, fake = torch.randn(4, 3) * 13, torch.randn(4, 3) * 13
    fake[0].zero_()  # Repaired stem must include a fresh adapter's origin.
    ours, stats = cap_penalty(ours_d, real, fake)
    theirs, _ = regularizers.GradRegularizer("b_cap", 1.0).penalty(ref_d, real / 13, fake / 13, 0)
    a = torch.autograd.grad(ours, tuple(ours_d.parameters()))
    b = torch.autograd.grad(theirs, tuple(ref_d.parameters()))
    torch.testing.assert_close(ours, theirs, atol=2e-5, rtol=2e-6)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-6)
    assert stats["fake_kept"] == 4
    result["cap"] = {"loss": float(ours.detach()), "loss_error": maximum_error(ours, theirs),
                     "parameter_gradient_error": max(maximum_error(x, y) for x, y in zip(a, b)),
                     "fake_rows_kept_including_origin": stats["fake_kept"],
                     "coordinates": "raw hidden units divided by the fixed teacher RMS, scale=13"}
    return result


def load_critic(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for name, expected in state["signature"]["sources"].items():
        assert sha(ROOT / "conceptmod/textsliders" / name) == expected, name
    settings = state["signature"]["settings"]
    weights = state["modules"]["critic"]
    assert settings["adv_arch"] == "tx" and settings["adv_condition"] == "none"
    assert settings["adv_in"] == "scaled"
    critic = SpanTransformerD(weights["proj.weight"].shape[1], width=settings["adv_width"],
                              n_layers=settings["adv_layers"], n_heads=settings["adv_heads"],
                              in_mode=settings["adv_in"], readout=settings["adv_readout"])
    critic.load_state_dict(weights, strict=True)
    return critic.eval().requires_grad_(False), settings


@torch.no_grad()
def critic_blind_spots(critic, spans):
    x, real, mask = spans["fake"].float(), spans["real"].float(), spans["mask"]
    w = critic.proj.weight.double()
    rank = int(torch.linalg.matrix_rank(w))
    q = torch.linalg.qr(w.T, mode="reduced").Q[:, :rank]

    def null_part(value):
        value = value.double()
        return value - (value @ q) @ q.T

    def energy_fraction(value):
        valid = value[mask].double()
        return float(null_part(valid).square().sum() / valid.square().sum())

    perturb = null_part(torch.randn_like(x)) * mask[..., None]
    perturb *= 2 * real[mask].double().square().mean().sqrt() / perturb[mask].square().mean().sqrt()
    changed = x + perturb.float()
    features, scores = critic.features(x, mask), critic(x, mask)
    feature_error = maximum_error(critic.features(changed, mask), features)
    score_error = maximum_error(critic(changed, mask), scores)
    torch.testing.assert_close(critic.features(changed, mask), features, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(critic(changed, mask), scores, atol=2e-5, rtol=2e-5)
    permuted = x.clone()
    for row in range(len(x)):
        positions = mask[row].nonzero().flatten()
        permuted[row, positions[:-1]] = x[row, positions[:-1].flip(0)]
    torch.testing.assert_close(critic.features(permuted, mask), features, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(critic(permuted, mask), scores, atol=2e-5, rtol=2e-5)
    real_features = critic.features(real, mask)
    fm = F.mse_loss(features.mean(0), real_features.mean(0))
    doubled_fm = F.mse_loss((2 * features).mean(0), (2 * real_features).mean(0))
    doubled_score = critic.head(critic.out_norm(2 * features)).squeeze(-1)
    return {
        "projection": {"input_dim": w.shape[1], "rank": rank, "nullity": w.shape[1] - rank,
                       "perturbation_rms_over_teacher_rms": 2.0,
                       "projected_perturbation_max_abs": float((perturb @ w.T).abs().max()),
                       "score_max_abs_change": score_error, "feature_max_abs_change": feature_error,
                       "teacher_energy_fraction_in_current_nullspace": energy_fraction(real),
                       "student_teacher_error_energy_fraction_in_current_nullspace": energy_fraction(x - real),
                       "scope": "Fixed saved critic, per-token linear projection. Its row space can rotate during training; reachability and audio impact are untested."},
        "token_permutation": {"score_max_abs_change": maximum_error(critic(permuted, mask), scores),
                              "feature_max_abs_change": maximum_error(critic.features(permuted, mask), features),
                              "scope": "Permutes contextual hidden vectors, preserving the final valid token. This is not a claim that permuting lyric text has no effect."},
        "pooled_feature_scale": {"raw_fm": float(fm), "doubled_fm_ratio": float(doubled_fm / fm),
                                 "score_max_abs_change": maximum_error(doubled_score, scores),
                                 "scope": "Counterfactual rescaling at the pooled-feature interface; not a trained fix or an exact network reparameterization."},
    }


def subobjective_counterexamples():
    # Shared feature distribution is correct but the per-row assignment is wrong.
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.zero_()
        critic.bias.zero_()
    real = torch.tensor([[-2., 0.], [2., 0.], [0., -2.], [0., 2.]])
    fake = real.roll(1, 0).clone().requires_grad_(True)
    dloss = rp_d_loss(critic(real).flatten(), critic(fake).flatten())
    dgrads = torch.autograd.grad(dloss, tuple(critic.parameters()))
    mean_fake, mean_real = fake.detach().mean(0), real.mean(0)
    fm = sum(feature_mean_surrogate(row[None], mean_fake, mean_real) / 4 for row in fake)
    gloss = rp_g_loss(critic(fake).flatten(), critic(real).flatten()) + fm
    ggrad = torch.autograd.grad(gloss, fake)[0]
    assert torch.count_nonzero(ggrad) == 0
    assert all(torch.count_nonzero(x) == 0 for x in dgrads)
    # The FM term by itself also accepts collapsing all samples to the mean.
    collapsed = real.mean(0).repeat(4, 1).requires_grad_(True)
    collapsed_fm = F.mse_loss(collapsed.mean(0), real.mean(0))
    collapse_grad = torch.autograd.grad(collapsed_fm, collapsed)[0]
    assert torch.count_nonzero(collapse_grad) == 0
    return {"row_permutation": {"per_row_mse": float(F.mse_loss(fake.detach(), real)),
                                "batch_fm": float(fm.detach()), "d_loss": float(dloss.detach()),
                                "d_parameter_gradient_norm": float(torch.cat([x.flatten() for x in dgrads]).norm()),
                                "g_input_gradient_norm": float(ggrad.norm()),
                                "scope": "Stationary witness for unconditional adversarial plus batch-FM subgame with a flat critic; the full objective also includes rowwise end regularization."},
            "centroid_collapse": {"batch_fm": float(collapsed_fm.detach()),
                                  "fm_gradient_norm": float(collapse_grad.norm()),
                                  "scope": "Mean matching alone, not a claim that a sufficiently capable adversarial critic cannot detect collapsed outputs."}}


def lora_geometry():
    a, b = torch.randn(3, 7), torch.randn(5, 3)
    effective = b @ a
    # Identical effective adapters with opposite factor signs.
    averaged_factors = ((b + -b) / 2) @ ((a + -a) / 2)
    averaged_effective = (b @ a + (-b) @ (-a)) / 2
    torch.testing.assert_close(effective, averaged_effective)
    assert torch.count_nonzero(averaged_factors) == 0
    # The same effective finite step represented in two fixed factor gauges.
    da = torch.randn_like(a) * 0.01
    ordinary_step = float(da.norm())
    rescaled_step = float((da / 10).norm())
    torch.testing.assert_close(b @ (a + da) - b @ a,
                               (10 * b) @ ((a + da) / 10) - (10 * b) @ (a / 10),
                               rtol=1e-4, atol=2e-6)
    # Check the actual LoRA wrapper: changing a multiplier after forward does
    # not detach an existing graph without gradient-checkpoint recomputation.
    from conceptmod.textsliders.lora import LoRAModule
    base = nn.Linear(7, 5, bias=False).requires_grad_(False)
    adapter = LoRAModule("audit", base, multiplier=1, lora_dim=3, alpha=3)
    adapter.apply_to()
    with torch.no_grad():
        adapter.lora_down.weight.copy_(a)
        adapter.lora_up.weight.copy_(b)
    x = torch.randn(4, 7)
    plus = base(x)
    adapter.multiplier = 0.0
    zero = base(x)
    anchor = F.mse_loss(zero, adapter.org_forward(x))
    zero_grads = torch.autograd.grad(anchor, tuple(adapter.parameters()))
    plus_grads = torch.autograd.grad(plus.square().mean(), tuple(adapter.parameters()))
    assert all(torch.count_nonzero(g) == 0 for g in zero_grads)
    assert all(torch.count_nonzero(g) > 0 for g in plus_grads)
    return {"factor_average_relative_effective_error": float((averaged_factors - averaged_effective).norm() / effective.norm()),
            "equivalent_function_step_parameter_norm_ratio": ordinary_step / rescaled_step,
            "zero_scale_anchor_gradient_norm": float(torch.cat([x.flatten() for x in zero_grads]).norm()),
            "prior_plus_forward_gradient_survives_later_zero_multiplier": True,
            "scope": "Algebraic counterexamples, not evidence that consecutive training checkpoints undergo a sign flip. Effective-delta averaging may require higher rank or validated recompression."}


def end_margin_counterexample():
    # Same expression as trainer._frame_margins: reordering semantic-code
    # logits preserves stop probability while changing which code is likely.
    semantic = torch.tensor([8., 0., -2., -4.], dtype=torch.float64)
    moved = semantic.roll(1)
    eos = torch.tensor([-3.], dtype=torch.float64)
    before = eos[0] - semantic.logsumexp(0)
    after = eos[0] - moved.logsumexp(0)
    p = torch.cat([semantic, eos]).log_softmax(0)
    q = torch.cat([moved, eos]).log_softmax(0)
    kl = (p.exp() * (p - q)).sum()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert kl > 5
    return {"end_margin_squared_error": float((before - after).square()),
            "semantic_plus_eos_kl_nats": float(kl),
            "original_most_likely_code": int(semantic.argmax()),
            "changed_most_likely_code": int(moved.argmax()),
            "scope": "Logit-space counterexample to a duration-only constraint; reachability through the fixed output head and LoRA is not asserted."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, default=Path("/tmp/opencode/ParticleGAN"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(407)
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REVISION, revision
    tracked_changes = subprocess.check_output(["git", "-C", str(args.upstream), "diff", "HEAD", "--", "lib/gan_loss.py", "lib/grad_regularizers.py", "examples/100gaussians.py"], text=True)
    assert not tracked_changes
    paths = [STATE, SPANS, Path(__file__)] + [ROOT / "conceptmod/textsliders" / name for name in (
        "lm_adv.py", "lm_gan.py", "lora.py", "train_lm_slider_music3.py",
        "lm_gan_state.py", "lm_particles.py", "slider_targets.py")]
    before = {str(path): sha(path) for path in paths}
    result = {"upstream_revision": revision, "seed": 407, "device": "cpu",
              "kernel_parity": kernel_parity(args.upstream)}
    critic, settings = load_critic(STATE)
    result["checkpoint_source_fingerprints_match"] = True
    spans = torch.load(SPANS, map_location="cpu", weights_only=True)[0]
    assert int(spans["step"]) == 600
    result["saved_critic_600"] = critic_blind_spots(critic, spans)
    result["subobjective_counterexamples"] = subobjective_counterexamples()
    result["lora_geometry"] = lora_geometry()
    result["end_margin_counterexample"] = end_margin_counterexample()
    result["settings"] = settings
    result["provenance"] = before
    result["sources_unchanged"] = all(sha(path) == before[str(path)] for path in paths)
    assert result["sources_unchanged"]
    result["limitations"] = ["Cached spans are prompt-only bf16 captures, not a replay of training geometry.",
                             "No generated-audio effects or LoRA reachability are inferred from constructed hidden inputs.",
                             "Parity checks cover representative kernel values and derivatives, not every numerical trajectory."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("settings", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
