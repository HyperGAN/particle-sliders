"""Freeze the selected native prediction pairs on both sampler trajectories."""
from functools import lru_cache
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
import torch

from . import digest, read_json, write_json
from .image_run import pin


def save_tensors(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = tmp.name
        try:
            torch.save(data, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            os.unlink(name)
            raise
    os.replace(name, path)


def offload(value):
    if isinstance(value, tuple):
        return tuple(offload(t) for t in value)
    return value.detach().cpu().clone()


@torch.no_grad()
def capture(run, scene):
    host = run.host
    for key in ("neutral_image", "positive_image", "matte"):
        if pin(scene[key]["path"]) != scene[key]:
            raise ValueError("teacher image or matte changed")
    binding = digest(dict(profile=run.identity, scene=scene, layout="both-native-trajectories-late-cues-v1"))
    path = run.root / "targets" / (binding + ".pt")
    receipt = path.with_suffix(".json")
    if receipt.exists():
        result = read_json(receipt)
        if result["binding"] != binding or pin(path) != result["shard"]:
            raise ValueError("frozen native targets changed")
        return result

    plain, treated, explicit = (host.encode(scene[k]) for k in ("neutral", "positive", "explicit"))
    with Image.open(scene["matte"]["path"]) as image:
        mask = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32).copy()).to(host.runtime.device)
    mask = mask[None,None] / 255
    rows = []
    for trajectory, name in enumerate(("neutral","positive")):
        def observe(position, sample, time, condition, native):
            if trajectory == 0:
                neutral, positive = native.float(), host.runtime.predict(sample, time, treated).float()
            else:
                neutral, positive = host.runtime.predict(sample, time, plain).float(), native.float()
            residual = positive - neutral
            mixed = residual
            if position >= math.ceil(.7 * host.profile["steps"]):
                cue = host.runtime.predict(sample, time, explicit).float()
                mixed = residual.lerp(cue-neutral, .35)
            weight = .45 + .55 * torch.nn.functional.interpolate(mask, size=sample.shape[-2:], mode="area")
            if sample.ndim == 5:
                weight = weight.unsqueeze(2)
            weighted = mixed * weight
            norm = weighted.norm()
            if not torch.isfinite(norm) or norm <= 1.e-10:
                raise ValueError("degenerate native teacher field")
            target = neutral + weighted * (residual.norm() / norm)
            if not torch.isfinite(target).all():
                raise ValueError("nonfinite native teacher field")
            rows.append(dict(sample=offload(sample), time=offload(time), target=offload(target),
                             neutral=offload(neutral), features=offload(host.features(plain, time, position, trajectory))))
        replay = host.observe(scene[name], scene["seed"], observe)
        with Image.open(scene[name + "_image"]["path"]) as original:
            if replay.convert("RGB").tobytes() != original.convert("RGB").tobytes():
                raise ValueError(f"native {name} trajectory did not reproduce its selected teacher")
    count = 2 * host.profile["steps"]
    if len(rows) != count:
        raise ValueError("incomplete native sampler trace")
    save_tensors(path, dict(rows=rows, plain=offload(plain), treated=offload(treated)))
    result = dict(binding=binding, shard=pin(path), count=count, response_shape=list(rows[0]["target"].shape))
    write_json(receipt, result)
    return result


class TrainingRows:
    def __init__(self, host, shards):
        self.host, self.shards = host, shards
        self.count = shards[0]["count"]
        self.response_shape = shards[0]["response_shape"]
        if any(s["count"] != self.count or s["response_shape"] != self.response_shape for s in shards):
            raise ValueError("native target geometries differ")
        self.load = lru_cache(maxsize=3)(self._load)

    def identity(self):
        return [s["shard"]["sha256"] for s in self.shards]

    def __len__(self):
        return self.count * len(self.shards)

    def _load(self, i):
        return torch.load(self.shards[i]["shard"]["path"], map_location="cpu", weights_only=True)

    def __getitem__(self, index):
        group, position = divmod(index, self.count)
        shard = self.load(group)
        return self.host.training_row(shard["rows"][position], shard["plain"], shard["treated"])

    def features(self):
        return torch.stack([r["features"] for i in range(len(self.shards)) for r in self.load(i)["rows"]])
