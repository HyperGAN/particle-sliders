"""Original adult characters, isolated splits, and lighting-only definitions."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from .contracts import Definition, Shared, VARIATIONS, atomic_json, digest, paired_prompts, validate_pair

DATA = Path(__file__).resolve().parents[1] / "configs/anima"
FRAMINGS = ("upper body, waist-up view, eye-level camera", "full body, entire figure in frame, eye-level camera",
            "portrait, head and shoulders, eye-level camera", "cowboy shot, thighs in frame, eye-level camera")
SCENES = ("in an unadorned stone corridor with bare stone walls", "in a quiet courtyard", "beside a plain plaster wall",
          "in a wood-paneled room", "on a balcony above a village", "in an empty arched hall")
POSES = ("standing upright, facing viewer, arms at sides",
         "sitting upright on a plain wooden chair, facing viewer, feet flat on floor, hands resting in lap",
         "facing viewer, head upright", "standing upright, facing viewer, both hands on hips")


def load_definitions():
    return [Definition(**d) for d in json.loads((DATA / "definitions.json").read_text())]


def load_characters():
    return json.loads((DATA / "characters.json").read_text())


def compile_manifest(split="train", *, definitions=None, characters=None):
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown dataset split")
    catalog = load_characters() if characters is None else characters
    characters = [c for c in catalog if c["split"] == split]
    definitions = load_definitions() if definitions is None else definitions
    rows = []
    for variation in VARIATIONS:
        defs = [d for d in definitions if d.variation == variation]
        if split == "train":
            cases = [(i, (i % 6 if r == 0 else (i % 6 + (1 if i < 6 else 5)) % 6), r)
                     for r in range(2) for i in range(12)]
        elif split == "dev":
            cases = [(i % 4, i, i // 4) for i in range(8)]
            # Bare prompts are separate fixtures with no quality/medium scaffolding.
            cases += [(i, 6 + i % 2, 2) for i in range(4)]
        else:
            cases = [(i, 6 + i % 2, i // 4) for i in range(8)]
        for index, (ci, di, turn) in enumerate(cases):
            char, definition = characters[ci], defs[di]
            bare = (split == "dev" and turn == 2) or (split == "test" and ci >= 4)
            if bare:
                definition = replace(definition, neutral_lighting="")
            shared = Shared(character=char["description"], outfit=char["outfit"],
                pose=POSES[(ci + turn) % 4], framing=FRAMINGS[(ci + turn) % 4],
                scene=SCENES[(ci // 2 + turn * 3) % 6],
                **({"quality": "", "medium": ""} if bare else {}))
            seeds = list(range(1000, 1016)) if split == "train" else (
                [29001, 29027] if split == "dev" else [81013, 82007, 83003, 84011])
            row = dict(id=f"{variation}-{split}-{index:02}", variation=variation, split=split,
                character=char["id"], definition=definition.id, definition_split=definition.split,
                seeds=seeds, bare=bare, presentation=char["presentation"], skin=char["skin"],
                **paired_prompts(shared, definition))
            validate_pair(row)
            rows.append(row)
    manifest = dict(schema=1, split=split, definitions_sha256=digest([asdict(d) for d in definitions]),
        characters_sha256=digest(catalog), rows=rows)
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest, definitions=definitions, characters=catalog)
    return manifest


def validate_manifest(manifest, *, definitions=None, characters=None):
    if digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
        raise ValueError("Manifest hash mismatch")
    if len({r["id"] for r in manifest["rows"]}) != len(manifest["rows"]):
        raise ValueError("Duplicate row ids")
    catalog = load_characters() if characters is None else characters
    chars = {c["id"]: c for c in catalog}
    definitions = {d.id: d for d in (load_definitions() if definitions is None else definitions)}
    if manifest["characters_sha256"] != digest(catalog) or manifest["definitions_sha256"] != digest(
            [asdict(d) for d in definitions.values()]):
        raise ValueError("Manifest catalog provenance changed")
    for row in manifest["rows"]:
        validate_pair(row)
        if row["split"] != manifest["split"] or chars[row["character"]]["split"] != row["split"]:
            raise ValueError("Character split leakage")
        character = chars[row["character"]]
        if (row["shared"]["character"] != character["description"]
                or row["shared"]["outfit"] != character["outfit"]
                or any(row[k] != character[k] for k in ("presentation", "skin"))):
            raise ValueError("Shared character fields differ from the character catalog")
        if (row["shared"]["pose"] not in POSES or row["shared"]["framing"] not in FRAMINGS
                or row["shared"]["scene"] not in SCENES):
            raise ValueError("Unrecognized shared scene or framing")
        d = definitions[row["definition"]]
        if row.get("bare"):
            if row["split"] == "train":
                raise ValueError("Bare prompts are held-out wording tests")
            d = replace(d, neutral_lighting="")
        if (d.lighting, d.neutral_lighting, d.variation, d.split) != (
            row["positive_lighting"], row["neutral_lighting"], row["variation"], row["definition_split"]):
            raise ValueError("Definition differs from the source catalog")
        if row["split"] == "train" and row["definition_split"] != "train":
            raise ValueError("Evaluation definition leaked into training")
    if manifest["split"] == "train":
        for variation in VARIATIONS:
            rows = [r for r in manifest["rows"] if r["variation"] == variation]
            if len(rows) != 24 or set(Counter(r["character"] for r in rows).values()) != {2}:
                raise ValueError("Unbalanced character coverage")
            if set(Counter(r["definition"] for r in rows).values()) != {4}:
                raise ValueError("Unbalanced definition coverage")
            if any(len(r["seeds"]) != 16 for r in rows):
                raise ValueError("Expected 16 seeds per row")


def write_manifests(directory, include_test=False):
    directory = Path(directory)
    for split in ("train", "dev", "test") if include_test else ("train", "dev"):
        atomic_json(directory / f"{split}.json", compile_manifest(split))


def archive_catalog(root):
    """Retain the exact catalogs behind existing exports before revising a definition."""
    bundle = dict(schema=1, definitions=[asdict(d) for d in load_definitions()],
                  characters=load_characters(),
                  **{split: compile_manifest(split) for split in ("train", "dev")})
    bundle["sha256"] = digest(bundle)
    path = Path(root) / "manifests/catalogs" / (bundle["sha256"] + ".json")
    if path.exists() and json.loads(path.read_text()) != bundle:
        raise ValueError("Archived catalog changed")
    if not path.exists():
        atomic_json(path, bundle)
    return path


def compatible_manifest_hashes(root, split, variation):
    """Accept a historic global hash only when this variation's full rows are identical.

    Changing one variation changes the whole manifest hash. This comparison
    preserves the provenance of unaffected runs without rewriting any export,
    cached target, metric fixture or previously rendered image.
    """
    if split not in ("train", "dev") or variation not in VARIATIONS:
        raise ValueError("Compatibility is limited to training/development variations")
    current = compile_manifest(split)
    expected = [r for r in current["rows"] if r["variation"] == variation]
    hashes = {current["sha256"]}
    for path in sorted((Path(root) / "manifests/catalogs").glob("*.json")):
        bundle = json.loads(path.read_text())
        if digest({k: v for k, v in bundle.items() if k != "sha256"}) != bundle.get("sha256"):
            raise ValueError("Archived catalog hash mismatch")
        definitions = [Definition(**d) for d in bundle["definitions"]]
        archived = bundle[split]
        validate_manifest(archived, definitions=definitions, characters=bundle["characters"])
        rows = [r for r in archived["rows"] if r["variation"] == variation]
        if rows == expected:
            hashes.add(archived["sha256"])
    return hashes
