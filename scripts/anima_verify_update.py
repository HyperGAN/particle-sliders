"""Compare complete D/G/VIC optimizer updates with the pinned reference procedure."""
import argparse
import ast
import copy
from contextlib import contextmanager
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
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    import torch
    from torch import nn
    from torch.nn import functional as F
    from lumen_studio.contracts import atomic_json, file_hash
    from lumen_studio.game import Game, Sampler
    from lumen_studio.vendor import reference
    torch.set_num_threads(2)
    torch.set_default_dtype(getattr(torch, args.dtype))
    atol, rtol = (2e-5, 2e-4) if args.dtype == "float32" else (2e-9, 2e-8)
    pinned = json.loads((ROOT / "lumen_studio/vendor/provenance.json").read_text())
    scope = dict(torch=torch, nn=nn, F=F, math=math, contextmanager=contextmanager,
                 REFERENCE=reference.REFERENCE.copy())
    symbols = {
        "analysis/slider2d/adv.py": ["rp_d_loss", "rp_g_loss"],
        "conceptmod/textsliders/lm_adv.py": ["_SetBlock", "param_grad_norm"],
        "conceptmod/textsliders/particle_bridge_gan.py": ["mlp", "bound_score", "register_paired_error_norm",
            "RoutedMLP", "particle_vic", "noise_std", "GlobalMixErrorCritic", "_identity_context", "update"],
    }
    for name, names in symbols.items():
        source = args.reference_root / name
        assert file_hash(source) == pinned["files"][name]
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    cap_path = args.reference_root / "analysis/slider2d/grad_regularizers.py"
    assert file_hash(cap_path) == pinned["files"]["analysis/slider2d/grad_regularizers.py"]
    spec = importlib.util.spec_from_file_location("pinned_update_cap", cap_path)
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    scope["make_grad_regularizer"] = lambda **kwargs: cap.GradRegularizer(**kwargs)

    class Network(nn.Module):
        def __init__(self, amplitude):
            super().__init__()
            self.particles = nn.Parameter(torch.randn(128, 4))
            self.net = reference.RoutedMLP(16, 16)
            self.amplitude = amplitude
        def forward(self, features):
            return self.net(features, self.particles) * self.amplitude

    report = dict(passed=False, source_hashes=pinned["files"], checks=[], dtype=args.dtype,
                  fixture="identical precomputed D/G rows, noise, and VIC draws; already-normalized paired fields",
                  tolerance=dict(atol=atol, rtol=rtol))
    for amplitude, noise_start in ((.02, .02), (1., 3.5)):
        for microbatch in (1, 8):
            for step in (1, 3, 4, 8):
                torch.manual_seed(29)
                norm = dict(scale=torch.ones(10, 16), edit_rms=torch.full((10,), noise_start * .28))
                actual = Game(Network(amplitude), norm, count=5, seed=7, microbatch=microbatch)
                original_net = copy.deepcopy(actual.adapter)
                original_critic = scope["GlobalMixErrorCritic"](torch.stack((torch.zeros(16), torch.ones(16))),
                    tokens=8, width=48, layers=1, heads=4, score_bound=8)
                original_critic.load_state_dict(actual.critic.state_dict())
                # The reference normally whitens absolute teacher/student states.
                # Supply pre-normalized paired fields with a zero teacher here;
                # normalization itself has a separate real-cache source audit.
                original_critic.target_mean.zero_()
                original_critic.target_std.fill_(1.)
                original_g = torch.optim.Adam([
                    dict(params=[p for n, p in original_net.named_parameters() if n != 'particles'], lr=2e-5),
                    dict(params=[original_net.particles], lr=6e-3)], betas=(0., .999), weight_decay=0.)
                original_d = torch.optim.Adam(original_critic.parameters(), lr=9e-4,
                                              betas=(0., .999), weight_decay=0.)
                features, targets = torch.randn(5, 16), torch.randn(5, 16) * amplitude
                replay = Sampler(5, 7)
                replay.load_state_dict(actual.sampler.state_dict())
                fixtures = [replay.draw(phase, 16) for phase in ('d', 'g')]
                sigma = reference.noise_std(step - 1, start=actual.noise_starts[0], decay_steps=1600, hold=1.)

                class FixedSampler:
                    def __init__(self):
                        self.phase = 0
                    def batch(self, dimension, device, unused_schedule_sigma):
                        indices, noise = fixtures[self.phase]
                        self.phase += 1
                        return torch.tensor(indices), noise * torch.tensor(sigma)
                    def vic_rows(self, count, device):
                        return torch.randperm(count, generator=replay.generators['vic'])[:64].to(device)

                expected = scope['update'](original_net, original_critic, original_g, original_d,
                    torch.zeros(5, 16), lambda i, phase: original_net(features[i:i+1]) - targets[i:i+1],
                    sampler=FixedSampler(), step=step)
                got = actual.update(lambda ids: actual.adapter(features[ids]) - targets[ids], [0] * 5, step)
                assert expected['d_rows'] == got['d_rows'] and expected['g_rows'] == got['g_rows']
                errors = dict(parameters=0., gradients=0., optimizer=0.)
                failures = []

                def compare(a, b, kind):
                    if isinstance(a, torch.Tensor):
                        errors[kind] = max(errors[kind], float((a - b).abs().max()))
                        if not torch.allclose(a, b, atol=atol, rtol=rtol):
                            index = int((a - b).abs().flatten().argmax())
                            failures.append(dict(kind=kind, shape=list(a.shape), index=index,
                                expected=float(a.flatten()[index]), actual=float(b.flatten()[index]),
                                max_abs=float((a - b).abs().max())))
                    elif isinstance(a, dict):
                        assert a.keys() == b.keys()
                        for key in a:
                            compare(a[key], b[key], kind)
                    elif isinstance(a, (tuple, list)):
                        assert len(a) == len(b)
                        for x, y in zip(a, b):
                            compare(x, y, kind)
                    else:
                        assert a == b

                for first, second in ((original_net, actual.adapter), (original_critic, actual.critic)):
                    for a, b in zip(first.parameters(), second.parameters()):
                        compare(a.detach(), b.detach(), 'parameters')
                        if a.grad is not None:
                            compare(a.grad, b.grad, 'gradients')
                compare(original_g.state_dict(), actual.g.state_dict(), 'optimizer')
                compare(original_d.state_dict(), actual.d.state_dict(), 'optimizer')
                for a, b in (('d_adv', 'd_adv'), ('g_adv', 'g_adv'), ('d_pen', 'd_penalty'), ('particle_vic', 'vic')):
                    if not math.isclose(expected[a], got[b], abs_tol=atol, rel_tol=rtol):
                        failures.append(dict(kind=a, expected=expected[a], actual=got[b]))
                for key in replay.generators:
                    assert torch.equal(replay.generators[key].get_state(), actual.sampler.generators[key].get_state())
                with torch.no_grad():
                    held_out = torch.randn(32, 16) * sigma
                    critic_output_error = float((original_critic(held_out) - actual.critic(held_out)).abs().max())
                    generator_output_error = float((original_net(features) - actual.adapter(features)).abs().max())
                report['checks'].append(dict(amplitude=amplitude, microbatch=microbatch, step=step,
                    cap=got['d_penalty'], max_abs=errors, failures=failures,
                    critic_output_max_abs=critic_output_error, generator_output_max_abs=generator_output_error))
    assert any(c['cap'] > 0 for c in report['checks'] if c['step'] % 4 == 0)
    report['passed'] = not any(c['failures'] for c in report['checks'])
    atomic_json(ROOT / f'artifacts/anima/correctness/update-reverification-{args.dtype}.json', report)
    print(json.dumps(dict(passed=report['passed'], dtype=args.dtype, checks=len(report['checks']),
        failing_cases=[dict(amplitude=c['amplitude'], microbatch=c['microbatch'], step=c['step'],
                            failures=len(c['failures'])) for c in report['checks'] if c['failures']],
        max_abs={k: max(c['max_abs'][k] for c in report['checks']) for k in ('parameters', 'gradients', 'optimizer')})))


if __name__ == '__main__':
    main()
