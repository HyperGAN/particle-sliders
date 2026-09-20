"""Product availability, separate from immutable training and model identities."""
from .contracts import VARIATIONS


RETIRED = {
    "theatrical": "Retired after its pilot changed clothing without a useful lighting gain.",
}
MIXER_VARIATIONS = tuple(name for name in VARIATIONS if name not in RETIRED)
DRAFT_ATMOSPHERES = {
    "dusk": {"label": "Dusk", "description": "Amber/violet twilight"},
}
