from pathlib import Path
import hashlib
import json
import secrets
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CAMPAIGN = ROOT / "analysis/uni16_fresh3400_20260912"
MODELS = ROOT / "models/uni16-fresh3400-v1"
AUDIO = ROOT / "eval/listen/uni16-fresh3400-20260912"
PAGE = ROOT / "eval/listen/uni16-release-audit-20260914"
sys.path[:0] = [str(ROOT), str(ROOT.parent / ".cache/slider-quality/python"), str(ROOT.parent)]


def read(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{secrets.token_hex(5)}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); tmp.replace(p)


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def catalog():
    return read(CAMPAIGN / "catalog.json")["sliders"]


def weights(item, label):
    if label == "published660":
        return Path(item["published_weights"])
    if label == "step600":
        return MODELS / item["id"] / f"{item['id']}-warmup600.safetensors"
    run = MODELS / item["id"] / f"{item['id']}-fresh3400"
    return run / f"{run.name}_{label}.safetensors"


def verify():
    p = read(WORK / "protocol.json")
    if not p or sha(CAMPAIGN / "manifest.json") != p["training_manifest_sha256"]:
        raise ValueError("Missing audit protocol or changed training manifest")
    for name, expected in p["sources"].items():
        if sha(name) != expected:
            raise ValueError(f"Scoring source changed: {name}")
    return p
