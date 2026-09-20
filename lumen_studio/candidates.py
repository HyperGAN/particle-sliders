"""Draft atmosphere pairs, isolated from registered sliders and training inputs."""
from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re

from .contracts import Shared, digest, file_hash, paired_prompts
from .dataset import compile_manifest


@dataclass(frozen=True)
class CandidateLighting:
    lighting: str
    neutral_lighting: str = ""


def compile_candidate(path):
    """Reuse balanced shared fields and the production prompt assembler only.

    New names remain drafts: this neither registers an adapter nor changes the
    frozen runtime identity, existing definitions, or prepared target caches.
    """
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec.get("schema") != 1 or not re.fullmatch(r"[a-z][a-z0-9-]+", spec["id"]):
        raise ValueError("Invalid candidate identity")
    definitions = spec["definitions"]
    if [d["split"] for d in definitions] != ["train"] * 6 + ["eval"] * 2:
        raise ValueError("A candidate needs six training definitions and two evaluation paraphrases")
    clauses = [CandidateLighting(d["lighting"], d.get("neutral_lighting", "")) for d in definitions]
    if any(not d.lighting.strip() or d.lighting == d.neutral_lighting for d in clauses):
        raise ValueError("Positive lighting must be present and differ from neutral")
    bundle = dict(schema=1, candidate=spec["id"], label=spec["label"], status="draft",
                  registered_slider=False, source_spec=spec, source_spec_sha256=file_hash(path),
                  compiler_sha256=file_hash(__file__))
    for split in ("train", "dev"):
        source = compile_manifest(split)
        templates = [r for r in source["rows"] if r["variation"] == "candlelit"]
        rows = []
        for i, template in enumerate(templates):
            di = int(template["definition"].rsplit("-", 1)[1]) - 1
            lighting = clauses[di]
            if template["bare"]:
                lighting = replace(lighting, neutral_lighting="")
            pair = paired_prompts(Shared(**template["shared"]), lighting)
            row = dict(template, id=f"{spec['id']}-{split}-{i:02}", variation=spec["id"],
                       definition=f"{spec['id']}-{di+1:02}", definition_split=definitions[di]["split"], **pair)
            assert row["shared"] == template["shared"]
            assert all(row[key] == value for key, value in paired_prompts(Shared(**row["shared"]), lighting).items())
            rows.append(row)
        if split == "train" and (len(rows) != 24
                or set(Counter(r["character"] for r in rows).values()) != {2}
                or set(Counter(r["definition"] for r in rows).values()) != {4}):
            raise ValueError("Candidate coverage differs from the balanced training layout")
        manifest = dict(schema=1, split=split, candidate=spec["id"], rows=rows,
                        shared_source_sha256=source["sha256"], characters_sha256=source["characters_sha256"])
        manifest["sha256"] = digest(manifest)
        bundle[split] = manifest
    bundle["sha256"] = digest(bundle)
    return bundle
