"""Exact native caption optimizer-boundary recovery and immutable frozen readouts."""

import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch


def copy_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: copy_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [copy_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(copy_cpu(v) for v in value)
    return value


def finite(value):
    if torch.is_tensor(value) and not value.isfinite().all():
        raise ValueError("Nonfinite caption recovery state")
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            finite(v)


def digest_fixed(fixed):
    h = hashlib.sha256()
    for row in fixed:
        for key in ["targets", "neutral"]:
            value = row[key].detach().float().cpu().contiguous()
            h.update(key.encode())
            h.update(value.numpy().tobytes())
        h.update(
            json.dumps(
                {k: row[k] for k in ["positive", "neutral_text"]}, sort_keys=True
            ).encode()
        )
    return h.hexdigest()


class CaptionState:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def save(self, step, c):
        k, keys, pos, has, cached = np.random.get_state()
        state = dict(
            identity=self.identity,
            step=step,
            fixed_sha256=digest_fixed(c["fixed"]),
            network=copy_cpu(c["network"].state_dict()),
            critic=copy_cpu(c["critic"].state_dict()),
            critic_attributes={
                name: getattr(c["critic"], name)
                for name in ["noise_start", "normalization"]
                if hasattr(c["critic"], name)
            },
            g=copy_cpu(c["g"].state_dict()),
            d=copy_cpu(c["d"].state_dict()),
            ema=copy_cpu(c["ema"]),
            sampler=copy_cpu(c["sampler"].state_dict()),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else [],
            python_rng=random.getstate(),
            numpy_rng=dict(
                kind=k, keys=keys.tolist(), pos=pos, has_gauss=has, cached=cached
            ),
        )
        finite(state)
        path = self.root / f"step-{step:06d}.pt"
        if path.exists():
            raise FileExistsError(path)
        temporary = self.root / (".pending-" + uuid.uuid4().hex)
        with temporary.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.rename(path)
        metadata = dict(
            identity=self.identity, step=step, file=path.name, sha256=checksum
        )
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(metadata, indent=2) + "\n")
        pending.replace(path.with_suffix(".json"))
        return metadata

    def restore(self, c):
        manifests = sorted(self.root.glob("step-*.json"), reverse=True)
        if not manifests:
            return 0
        meta = json.loads(manifests[0].read_text())
        p = self.root / meta["file"]
        if (
            meta["identity"] != self.identity
            or hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]
        ):
            raise ValueError("Caption recovery identity/checksum mismatch")
        state = torch.load(p, map_location="cpu", weights_only=True)
        finite(state)
        if (
            state["identity"] != self.identity
            or state["step"] != meta["step"]
            or state["fixed_sha256"] != digest_fixed(c["fixed"])
        ):
            raise ValueError("Caption recovery target bank changed")
        c["network"].load_state_dict(state["network"], strict=True)
        c["critic"].load_state_dict(state["critic"], strict=True)
        for name, v in state["critic_attributes"].items():
            setattr(c["critic"], name, v)
        c["g"].load_state_dict(state["g"])
        c["d"].load_state_dict(state["d"])
        c["sampler"].load_state_dict(state["sampler"])
        if c["ema"].keys() != state["ema"].keys():
            raise ValueError("Caption EMA keys changed")
        for key, v in state["ema"].items():
            c["ema"][key].copy_(v)
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state(
            (
                n["kind"],
                np.asarray(n["keys"], dtype=np.uint32),
                n["pos"],
                n["has_gauss"],
                n["cached"],
            )
        )
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["step"]


class FrozenReadouts:
    """Opt-in immutable CPU cache; wraps only a frozen Bonsai backend.

    Live serving is required only during population. Two independent requests
    must match bit-for-bit before a prompt can be reused. Cache keys include the
    verified model/runtime/pooling/normalization identity, not just prompt text.
    """

    def __init__(self, backend, root, identity):
        if backend is not None and backend.device.type != "cpu":
            raise ValueError("Frozen readout cache is CPU-only")
        self.backend = backend
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.device = backend.device if backend is not None else torch.device("cpu")
        self.dummy = backend.dummy if backend is not None else False

    def encode(self, text):
        from conceptmod.textsliders.bonsai_gguf_backend import EncodedText

        key = hashlib.sha256((self.identity + "\0" + text).encode()).hexdigest()
        path = self.root / (key + "-tokens.json")
        if path.exists():
            meta = json.loads(path.read_text())
            if meta["identity"] != self.identity or meta["text"] != text:
                raise ValueError("Frozen tokenizer cache identity mismatch")
            return EncodedText(ids=meta["ids"], text=text)
        if self.backend is None:
            raise FileNotFoundError("Frozen official tokenizer cache missing")
        enc = self.backend.encode(text)
        path.write_text(
            json.dumps(
                dict(identity=self.identity, text=text, ids=enc.ids), sort_keys=True
            )
            + "\n"
        )
        return enc

    def hidden(self, enc):
        key = hashlib.sha256((self.identity + "\0" + enc.text).encode()).hexdigest()
        path = self.root / (key + ".pt")
        if path.exists():
            meta = json.loads(path.with_suffix(".json").read_text())
            if (
                meta["identity"] != self.identity
                or meta["text"] != enc.text
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]
            ):
                raise ValueError("Frozen readout cache identity/checksum mismatch")
            out = torch.load(path, weights_only=True)
        else:
            if self.backend is None:
                raise FileNotFoundError("Frozen official readout cache missing")
            first = self.backend.hidden(enc)
            second = self.backend.hidden(enc)
            if not torch.equal(first, second):
                raise ValueError(
                    "Frozen official readout not bitwise repeatable; persistent cache disabled"
                )
            out = first.detach().cpu().clone()
            pending = self.root / (".pending-" + uuid.uuid4().hex)
            with pending.open("wb") as stream:
                torch.save(out, stream)
                stream.flush()
                os.fsync(stream.fileno())
            checksum = hashlib.sha256(pending.read_bytes()).hexdigest()
            pending.rename(path)
            path.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        identity=self.identity,
                        text=enc.text,
                        sha256=checksum,
                        independent_requests_exact=True,
                    ),
                    indent=2,
                )
                + "\n"
            )
        if (
            out.shape != (1, 5120)
            or out.dtype != torch.float32
            or not out.isfinite().all()
        ):
            raise ValueError("Invalid frozen readout cache")
        return out.clone()

    def teacher_hidden(self, enc):
        return self.hidden(enc)
