"""Shared image-runtime contract; model implementations live under backends."""
from typing import Protocol


class Cancelled(Exception):
    pass


class ImageRuntime(Protocol):
    identity: dict

    def render(self, prompt, seed, width=768, height=768, steps=10, progress=None, cancelled=None): ...
    def close(self): ...


def create_runtime(model_dir, device="cuda:0", *, backend="anima", **kwargs):
    if backend == "anima":
        from .backends.anima import TurboRuntime
        return TurboRuntime(model_dir, device, **kwargs)
    raise ValueError(f"Unsupported image backend: {backend}")
