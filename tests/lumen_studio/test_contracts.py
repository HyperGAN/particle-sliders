import copy
from collections import Counter

import pytest

from lumen_studio.contracts import Definition, Shared, digest, normalized_strengths, paired_prompts, validate_pair
from lumen_studio.dataset import compile_manifest, load_characters, load_definitions, validate_manifest


def test_balanced_isolated_pairs():
    chars = load_characters()
    assert Counter(c["split"] for c in chars) == {"train": 12, "dev": 4, "test": 8}
    assert len({c["description"] for c in chars}) == 24
    assert all("adult " in c["description"] for c in chars)
    defs = load_definitions()
    for variation in ("candlelit", "moonlit", "theatrical"):
        assert Counter(d.split for d in defs if d.variation == variation) == {"train": 6, "eval": 2}
    manifests = [compile_manifest(split) for split in ("train", "dev", "test")]
    train, dev, final = manifests
    assert len(train["rows"]) == 72
    for row in train["rows"]:
        validate_pair(row)
        clause = row["neutral_lighting"]
        assert row["neutral"] == Shared(**row["shared"]).prompt() + (", " + clause if clause else "")
        assert len(row["seeds"]) == 16
    for definition in [d for d in defs if d.split == "train"]:
        rows = [r for r in train["rows"] if r["definition"] == definition.id]
        assert len({r["character"] for r in rows}) == 4
        assert Counter(r["presentation"] for r in rows) == {"woman": 2, "man": 2}
        assert len({r["skin"] for r in rows}) == 3
        assert len({r["shared"]["scene"] for r in rows}) >= 3
    assert any(r["bare"] for r in dev["rows"])
    for manifest in (dev, final):
        assert any(r["bare"] for r in manifest["rows"])
        assert all(not r["neutral_lighting"] and not r["shared"]["quality"] for r in manifest["rows"] if r["bare"])
    for a, b in ((train, dev), (train, final), (dev, final)):
        assert not {r["character"] for r in a["rows"]} & {r["character"] for r in b["rows"]}
        assert not {s for r in a["rows"] for s in r["seeds"]} & {s for r in b["rows"] for s in r["seeds"]}


def test_rejects_wrong_neutral_and_shared_fields():
    manifest = compile_manifest()
    for field in ("neutral", "positive"):
        bad = copy.deepcopy(manifest)
        bad["rows"][0][field] += ", a red hat"
        bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
        with pytest.raises(ValueError, match="shared content"):
            validate_manifest(bad)


def test_explicit_neutral_is_recorded():
    shared = Shared("adult man", "navy coat", "standing", "waist-up", "stone hall")
    definition = Definition("test", "candlelit", "eval", "warm directional light", "diffuse neutral illumination")
    pair = paired_prompts(shared, definition)
    assert pair["neutral"].endswith(definition.neutral_lighting)
    assert pair["positive"] == shared.prompt() + ", " + definition.lighting


def test_consistent_pair_with_mislabeled_character_is_rejected():
    manifest = compile_manifest()
    row = manifest['rows'][0]
    row['shared']['character'] = 'an unrelated adult character'
    definition = next(d for d in load_definitions() if d.id == row['definition'])
    row.update(paired_prompts(Shared(**row['shared']), definition))
    manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='character catalog'):
        validate_manifest(manifest)


def test_archived_catalog_compatibility_requires_identical_variation_rows(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from lumen_studio import dataset
    original = {split: compile_manifest(split) for split in ("train", "dev")}
    path = dataset.archive_catalog(tmp_path)
    definitions = [replace(d, lighting=d.lighting + ", darker shadow edges")
                   if d.variation == "theatrical" else d for d in load_definitions()]
    # A draft can be checked without replacing the active catalog.
    draft = compile_manifest("train", definitions=definitions)
    assert draft["sha256"] != original["train"]["sha256"]
    assert compile_manifest("train") == original["train"]
    monkeypatch.setattr(dataset, "load_definitions", lambda: definitions)
    for split in ("train", "dev"):
        for variation in ("candlelit", "moonlit"):
            assert original[split]["sha256"] in dataset.compatible_manifest_hashes(tmp_path, split, variation)
        assert original[split]["sha256"] not in dataset.compatible_manifest_hashes(tmp_path, split, "theatrical")
    corrupted = json.loads(path.read_text())
    corrupted["train"]["rows"][0]["neutral"] += ", unrelated outfit"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="Archived catalog hash"):
        dataset.compatible_manifest_hashes(tmp_path, "train", "candlelit")


@pytest.mark.parametrize("energy", [0, .25, .5, 1])
def test_normalized_budget(energy):
    s = normalized_strengths(dict(candlelit=2, moonlit=3, theatrical=5), energy)
    assert sum(s.values()) == pytest.approx(energy)
    assert s["candlelit"] == pytest.approx(energy * .2)
    assert normalized_strengths({}, energy) == dict.fromkeys(s, 0.)
    assert normalized_strengths(dict(candlelit=1e308, moonlit=1e308), energy)["candlelit"] == energy / 2


@pytest.mark.parametrize("mix,energy", [({"bad": 1}, .5), ({"moonlit": -1}, .5),
    ({"moonlit": float('inf')}, .5), ({}, float('nan')), ({}, 1.1)])
def test_invalid_mix(mix, energy):
    with pytest.raises(ValueError):
        normalized_strengths(mix, energy)
