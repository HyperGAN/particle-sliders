from pathlib import Path

import pytest
import torch
from torch import nn

from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.particles import Mixer

torch.set_num_threads(2)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.Sequential(nn.Linear(4, 4))

    def forward(self, x, embed):
        h = self.to_q(x) + .2 * self.to_k(x) + self.to_v(embed)
        return self.to_out(torch.tanh(h))


class TinyRuntime:
    def __init__(self, *args, **kwargs):
        torch.manual_seed(103)
        self.device = torch.device("cpu")
        self.transformer = TinyTransformer().eval().requires_grad_(False)
        self.mixer = Mixer(self.transformer)
        self.identity = dict(model="tiny-test-only", sha256="a" * 64)
        self.calls = []

    def encode(self, prompt):
        value = sum(prompt.encode()) % 23 / 23
        return torch.tensor([[[value, value ** 2, 1 - value, .5]]])

    def noise(self, seed, width, height):
        return torch.randn(1, 4, 1, 2, 2, generator=torch.Generator().manual_seed(seed))

    def scheduler(self, steps):
        from diffusers import FlowMatchEulerDiscreteScheduler
        import numpy as np
        sched = FlowMatchEulerDiscreteScheduler(shift=3.)
        sched.set_timesteps(sigmas=np.linspace(1., 1 / steps, steps))
        return sched

    def predict(self, latent, timestep, embedding):
        if not torch.is_grad_enabled():
            self.calls.append((latent.clone(), torch.as_tensor(timestep).clone(), embedding.clone()))
        x = latent.flatten(2).transpose(1, 2)
        time = torch.as_tensor(timestep).reshape(-1, 1, 1) / 1000
        v = self.transformer(x, embedding) * (.5 + time)
        return v.transpose(1, 2).reshape_as(latent)

    def step(self, scheduler, velocity, timestep, latent):
        return scheduler.step(velocity, timestep, latent, return_dict=False)[0]

    def render(self, prompt, seed, width, height, steps, progress=None, cancelled=None):
        from PIL import Image
        from lumen_studio.runtime import Cancelled
        for i in range(steps):
            if cancelled and cancelled():
                raise Cancelled()
            if progress:
                progress((i + 1) / steps)
        return Image.new("RGB", (width, height), (42, 53, 64))

    def close(self):
        self.mixer.close()


@pytest.fixture
def tiny():
    runtime = TinyRuntime()
    yield runtime
    runtime.close()


def tiny_cache(path, runtime, split="train"):
    path = Path(path)
    path.mkdir()
    identity = dict(model=runtime.identity, split=split, variation="candlelit", manifest_sha256="b" * 64)
    fingerprint = digest(identity)
    shards = []
    for seed in range(4):
        records = []
        row = dict(id=f"row-{seed}", definition=f"definition-{seed}", character=f"character-{seed}",
                   shared=dict(framing="waist-up view"), bare=False)
        for trajectory in ("neutral", "positive"):
            z = runtime.noise(seed, 512, 512)
            for pos in range(10):
                t = 1000 - pos * 100
                with torch.no_grad():
                    vn = runtime.predict(z, t, runtime.encode("neutral"))
                    vp = runtime.predict(z, t, runtime.encode("positive"))
                records.append(dict(position=pos, timestep=float(t), trajectory=trajectory,
                    latent=z, neutral=vn, positive=vp))
                z = z - (vn if trajectory == "neutral" else vp) * .1
        file = path / f"{seed}.pt"
        save_tensor_file(file, dict(records=records, fingerprint=fingerprint, row=row,
            seed=seed, embedding=runtime.encode("neutral")))
        shards.append(dict(path=file.name, count=20, sha256=file_hash(file), seed=seed, row=row["id"]))
    atomic_json(path / "index.json", dict(identity=identity, fingerprint=fingerprint, shards=shards))
    return TargetCache(path)
