"""NTC's model catalog. Model settings belong to a backend, not the UI."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    backend: str
    default_steps: int
    supported_steps: tuple[int, ...]
    guidance: float
    default_size: int
    supports_mix: bool
    resolution_presets: tuple[tuple[str, int, int, str], ...] = ()

    def public(self):
        return dict(id=self.id, label=self.label, backend=self.backend, default_steps=self.default_steps,
                    supported_steps=list(self.supported_steps), guidance=self.guidance,
                    default_size=self.default_size, supports_mix=self.supports_mix,
                    resolution_presets=[dict(label=label, width=width, height=height, icon=icon)
                                        for label, width, height, icon in self.resolution_presets])


ANIMA = ModelSpec("anima-turbo-v1.1", "Anima Turbo 1.1", "anima", 10, (8, 10, 12), 1., 768, True,
                 (("Square", 768, 768, "□"),
                  ("Portrait", 768, 1024, "▯"),
                  ("Landscape", 1024, 768, "▭"),
                  ("Large square", 1024, 1024, "□"),
                  ("Small square", 512, 512, "□")))
# Additional backends can register their own specs and runtime factories here.
# Only installed, implemented models are exposed to Studio clients.
MODELS = {ANIMA.id: ANIMA}


def get_model(ident):
    try:
        return MODELS[ident]
    except KeyError:
        raise ValueError(f"Model is not available: {ident}") from None
