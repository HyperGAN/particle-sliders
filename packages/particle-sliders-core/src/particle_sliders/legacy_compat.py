"""Retired stamp metadata, without vendoring a loss or regularizer kernel."""
from dataclasses import asdict, field, make_dataclass
import json
from pathlib import Path

_METADATA = json.loads(Path(__file__).with_name("legacy_stamp.json").read_text())


def _to_dict(self):
    return asdict(self)


# Preserve attribute access, frozen dataclass fields and independent dictionary
# copies. This is historical metadata, not an upstream Recipe or checkpoint.
LegacyLockedShared = make_dataclass(
    "LegacyLockedShared",
    [(name, type(value), field(default=value))
     for name, value in _METADATA["fields"].items()],
    namespace={"to_dict": _to_dict, "__module__": __name__}, frozen=True,
)
LEGACY_LOCKED_SHARED = LegacyLockedShared()
