"""Content-bound native renders shared by the two character-vibe products."""
from importlib.resources import files
import json
import os
from pathlib import Path
from uuid import uuid4

from . import digest, file_digest, read_json, write_json
from .scene_grid import spec


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_digest(path))


class ImageRun:
    def __init__(self, directory, host, phrase):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.vibe = phrase.strip()
        if not self.vibe:
            raise ValueError("a character vibe phrase is required")
        self.grid = spec(self.vibe)
        self.panel = json.loads(files(__package__).joinpath("image_panel.json").read_text())
        self.identity = dict(version="character-vibe-v1", model=host.runtime.identity,
                             profile=host.profile, grid=self.grid, panel=self.panel)
        path = self.root / "spec.json"
        if path.exists() and read_json(path) != self.identity:
            raise ValueError("this run belongs to another phrase, model, grid or sampler")
        write_json(path, self.identity)

    def image(self, prompt, seed, *, raw=None, strength=0.):
        cfg = dict(model=self.host.runtime.identity, profile=self.host.profile, prompt=prompt,
                   seed=seed, raw=raw, strength=strength)
        path = self.root / "images" / (digest(cfg) + ".png")
        receipt = path.with_suffix(".json")
        if receipt.exists():
            saved = read_json(receipt)
            if saved["request"] != cfg or pin(path) != saved["image"]:
                raise ValueError("cached native image differs from its receipt")
            return saved["image"]
        path.parent.mkdir(exist_ok=True)
        image = self.host.render(prompt, seed, strength)
        temporary = path.with_suffix(f".{uuid4().hex}.pending")
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
        result = pin(path)
        write_json(receipt, dict(request=cfg, image=result))
        return result

    def candidates(self, scene, attempt):
        seed = scene["seed"] + 104729 * attempt
        return dict(scene=scene["id"], seed=seed, attempt=attempt,
                    neutral=self.image(scene["neutral"], seed),
                    options=[dict(t, image=self.image(t["positive"], seed)) for t in scene["treatments"]])
