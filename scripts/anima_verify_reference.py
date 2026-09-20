"""Recheck losses and derivatives against the exact captured ParticleGAN sources."""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/anima/correctness/reference-reverification.json")
    args = parser.parse_args()
    if args.dimension < 2:
        parser.error("--dimension must be at least two")
    import torch
    from torch import nn
    from torch.nn import functional as F
    from lumen_studio.contracts import atomic_json, file_hash
    from lumen_studio.vendor import reference as vendored
    from lumen_studio.vendor.grad_regularizers import GradRegularizer
    torch.set_num_threads(2)
    pinned = json.loads((ROOT / "lumen_studio/vendor/provenance.json").read_text())
    scope = dict(torch=torch, nn=nn, F=F, math=math, REFERENCE=vendored.REFERENCE.copy())
    sources = {}
    for name, sha in pinned["files"].items():
        path = args.reference_root / name
        assert file_hash(path) == sha, f"Reference changed: {name}"
        sources[name] = ast.parse(path.read_text())
    parts = {
        "analysis/slider2d/adv.py": ["rp_d_loss", "rp_g_loss"],
        "conceptmod/textsliders/lm_adv.py": ["_SetBlock"],
        "conceptmod/textsliders/particle_bridge_gan.py": ["mlp", "bound_score", "register_paired_error_norm",
            "RoutedMLP", "particle_vic", "noise_std", "GlobalMixErrorCritic"],
    }
    for name, symbols in parts.items():
        nodes = [n for n in sources[name].body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in symbols]
        assert len(nodes) == len(symbols)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), scope)
    path = args.reference_root / "analysis/slider2d/grad_regularizers.py"
    spec = importlib.util.spec_from_file_location("pinned_particle_cap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = dict(passed=False, source_hashes=pinned["files"], dimension=args.dimension, checks=[])
    for seed in (7, 29):
        targets = torch.stack((torch.zeros(args.dimension), torch.ones(args.dimension)))
        kwargs = dict(tokens=8, width=48, layers=1, heads=4, score_bound=8)
        torch.manual_seed(seed)
        original = scope["GlobalMixErrorCritic"](targets, **kwargs)
        torch.manual_seed(seed)
        actual = vendored.GlobalMixErrorCritic(targets, **kwargs)
        assert all(torch.equal(v, actual.state_dict()[k]) for k, v in original.state_dict().items())
        for amplitude in (.02, 1.):
            real, fake = (torch.randn(8, args.dimension) * amplitude for _ in range(2))
            for step in (1, 3, 4, 8):
                results = []
                for critic, d_loss, cap_type in ((original, scope["rp_d_loss"], module.GradRegularizer),
                                                (actual, vendored.rp_d_loss, GradRegularizer)):
                    adv = d_loss(critic(real), critic(fake))
                    penalty, _ = cap_type(arm="b_cap", coeff=1, kappa=1, lazy_k=4).penalty(
                        critic, real, fake, step=step, collect_stats=False)
                    gradients = torch.autograd.grad(adv + penalty, tuple(critic.parameters()))
                    results.append((adv, penalty, gradients))
                assert torch.equal(results[0][0], results[1][0])
                assert torch.equal(results[0][1], results[1][1])
                delta = max(float((a - b).abs().max()) for a, b in zip(results[0][2], results[1][2]))
                assert delta == 0
                report["checks"].append(dict(seed=seed, amplitude=amplitude, step=step,
                    d_loss=float(results[0][0].detach()), cap=float(results[0][1].detach()), gradient_max_abs=delta))
            predictions = fake.clone().requires_grad_(True)
            expected = scope["rp_g_loss"](original(real).detach(), original(predictions))
            got = vendored.rp_g_loss(actual(real).detach(), actual(predictions))
            assert torch.equal(expected, got)
            assert torch.equal(torch.autograd.grad(expected, predictions)[0], torch.autograd.grad(got, predictions)[0])
        particles = torch.randn(64, 4, requires_grad=True)
        expected, got = scope["particle_vic"](particles), vendored.particle_vic(particles)
        assert torch.equal(expected, got)
        assert torch.equal(torch.autograd.grad(expected, particles)[0], torch.autograd.grad(got, particles)[0])
    assert any(c["cap"] > 0 for c in report["checks"] if c["step"] % 4 == 0)
    report.update(passed=True, generator_loss_gradient_max_abs=0., particle_vic_gradient_max_abs=0.)
    atomic_json(args.output, report)
    print(json.dumps(dict(passed=True, dimension=args.dimension,
                         nonlinear_cap_checks=len(report["checks"]), max_gradient_error=0)))


if __name__ == "__main__":
    main()
