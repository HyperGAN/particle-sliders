"""Compare frozen teacher wording on failed development cases; do not alter training."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from lumen_studio.api import Generation, generation_payloads
    from lumen_studio.contracts import Shared, atomic_json, digest, paired_prompts, validate_pair
    from lumen_studio.dataset import compile_manifest, load_definitions
    from lumen_studio.provenance import model_identity
    from lumen_studio.store import Store
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=("white-wording", "minimal-lighting", "unspecified-neutral", "minimal-pair", "facial-shadows", "contrast-alternatives", "illumination-alternatives", "definition01-direction", "definition01-preservation", "definition03-facial", "definition05-lighting", "definition05-coherent", "definition05-contrast", "definition05-face", "definition05-low-light"), default="white-wording")
    args = parser.parse_args()
    root = ROOT / "artifacts/anima"
    store = Store(root / "studio/studio.sqlite3")
    identity = model_identity(root / "model")
    definitions = {d.id: d for d in load_definitions()}
    rows = [r for r in compile_manifest("dev")["rows"] if r["variation"] == "theatrical"][2:4]
    if args.round == "facial-shadows":
        rows += [r for r in compile_manifest("train")["rows"] if r["variation"] == "theatrical"][8:10]
    if args.round in ("contrast-alternatives", "illumination-alternatives"):
        training = [r for r in compile_manifest("train")["rows"] if r["variation"] == "theatrical"]
        rows = [training[13], training[15]]
    if args.round in ("definition01-direction", "definition01-preservation", "definition03-facial", "definition05-lighting", "definition05-coherent", "definition05-contrast", "definition05-face", "definition05-low-light"):
        training = [r for r in compile_manifest("train")["rows"] if r["variation"] == "theatrical"]
        development = [r for r in compile_manifest("dev")["rows"] if r["variation"] == "theatrical"]
        rows = ([training[13], development[2]] if args.round == "definition03-facial" else
                [training[i] for i in (4, 10, 15, 23)] + [development[4]] if args.round.startswith("definition05-") else
                [training[i] for i in (0, 6, 17, 19)] + [development[0]])
        if args.round == "definition05-low-light":
            rows = [training[10], training[15], development[4]]
    payloads, pairs = [], []
    for row in rows:
        definition = definitions[row["definition"]]
        validate_pair(row)
        if args.round == "white-wording":
            candidates = [("teacher_without_white", definition.lighting.replace("white ", ""))]
            conditions = [("neutral", row["neutral"]), ("teacher_original", row["positive"])]
        elif args.round == "minimal-lighting":
            candidates = [("without_chiaroscuro", definition.lighting.replace(", low-key chiaroscuro", "")),
                ("directional_shadows", "directional lighting, deep shadows"),
                ("shadow_contrast", "strong shadow contrast")]
            conditions = []
        elif args.round == "minimal-pair":
            candidates = [("neutral_directional_deep", "neutral directional illumination, deep shadow contrast"),
                ("neutral_diffuse_deep", "neutral diffuse illumination, deep shadow contrast"),
                ("neutral_oblique_deep", "neutral oblique illumination, deep shadow contrast")]
            conditions = []
        elif args.round == "facial-shadows":
            candidates = [("append_facial_shadows", definition.neutral_lighting + ", deep facial shadows"),
                ("diffuse_facial_shadows", "neutral diffuse illumination, deep facial shadows"),
                ("append_directional_facial", definition.neutral_lighting
                    + ", narrow directional illumination across the face, deep facial shadows")]
            conditions = []
        elif args.round == "contrast-alternatives":
            candidates = [(word + "_contrast", "neutral diffuse illumination, " + word + " shadow contrast")
                          for word in ("sharp", "rich", "heightened", "dark")]
            candidates += [("deep_shadows", "neutral diffuse illumination, deep shadows")]
            candidates += [("append_" + label, definition.neutral_lighting + ", " + clause)
                           for label, clause in (("dark_shadows", "dark shadows"),
                                                 ("hard_shadows", "hard shadows"),
                                                 ("low_key", "low-key lighting"))]
            conditions = [("neutral", row["neutral"]), ("teacher_original", row["positive"])]
        elif args.round == "illumination-alternatives":
            candidates = [(word + "_illumination", word + " diffuse illumination, moderate shadow contrast")
                          for word in ("dim", "subdued", "faint", "low-key")]
            candidates += [("soft_low_key", "soft low-key illumination, moderate shadow contrast"),
                           ("dim_neutral", "dim neutral illumination, moderate shadow contrast"),
                           ("moderately_deep", "neutral diffuse illumination, moderately deep shadow contrast"),
                           ("append_subdued", definition.neutral_lighting + ", subdued lighting")]
            conditions = [("neutral", row["neutral"]), ("teacher_original", row["positive"])]
        elif args.round == "definition01-preservation":
            candidates = [(label, definition.neutral_lighting + ", " + clause) for label, clause in (
                ("dim_lighting", "dim lighting"), ("subdued_lighting", "subdued lighting"),
                ("low_key_lighting", "low-key lighting"), ("soft_shadows", "soft shadows"),
                ("facial_shading", "facial shading"))]
            conditions = [("neutral", row["neutral"])]
        elif args.round == "definition01-direction":
            candidates = [("directional_deep", "neutral directional illumination, deep shadow contrast"),
                          ("strong_contrast", "neutral diffuse illumination, strong shadow contrast"),
                          ("stark_contrast", "neutral diffuse illumination, stark shadow contrast"),
                          ("dramatic_contrast", "neutral diffuse illumination, dramatic shadow contrast")]
            conditions = [("neutral", row["neutral"]), ("teacher_original", row["positive"])]
        elif args.round == "definition03-facial":
            candidates = [(label, definition.neutral_lighting + ", " + clause) for label, clause in (
                ("deep_facial", "deep facial shadows"), ("angular_facial", "angular facial shadows"),
                ("soft_facial", "soft facial shadows"), ("face_contrast", "facial shadow contrast"),
                ("shadowed_features", "shadowed facial features"))]
            conditions = [("neutral", row["neutral"])]
        elif args.round == "definition05-lighting":
            candidates = [("faint_diffuse", "faint diffuse illumination, moderate shadow contrast"),
                          ("soft_low_key", "soft low-key illumination, moderate shadow contrast")]
            candidates += [(label, definition.neutral_lighting + ", " + clause) for label, clause in (
                ("sculpted_shadows", "sculpted shadows"), ("soft_shadows", "soft shadows"),
                ("subdued_shadows", "subdued shadows"))]
            conditions = [("neutral", row["neutral"])]
        elif args.round == "definition05-coherent":
            candidates = [(label, definition.neutral_lighting + ", " + clause) for label, clause in (
                ("deep_shadows", "deep shadows"), ("stronger_shadows", "stronger shadows"),
                ("pronounced_shadows", "pronounced shadows"), ("rich_contrast", "rich shadow contrast"),
                ("dramatic_shading", "dramatic shading"))]
            conditions = [("neutral", row["neutral"])]
        elif args.round == "definition05-contrast":
            candidates = [(word + "_contrast", "neutral diffuse illumination, " + word + " shadow contrast")
                          for word in ("bold", "intense", "heavy", "distinct", "pronounced")]
            conditions = [("neutral", row["neutral"])]
        elif args.round == "definition05-face":
            candidates = [(label, definition.neutral_lighting + ", " + clause) for label, clause in (
                ("facial_shadows", "facial shadows"), ("shadowed_face", "shadowed face"),
                ("shaded_features", "shaded facial features"), ("dim_lighting", "dim lighting"),
                ("low_lighting", "low lighting"))]
            conditions = [("neutral", row["neutral"])]
        elif args.round == "definition05-low-light":
            candidates = [(label, definition.neutral_lighting + ", " + clause) for label, clause in (
                ("low_light", "low light"), ("low_illumination", "low illumination"),
                ("lower_illumination", "lower illumination"), ("low_ambient_light", "low ambient light"),
                ("subdued_ambient_light", "subdued ambient light"))]
            conditions = [("neutral", row["neutral"])]
        else:
            candidates = []
            changed = dict(row, **paired_prompts(Shared(**row["shared"]),
                replace(definition, neutral_lighting="")))
            validate_pair(changed)
            assert changed["positive"] == row["positive"] and changed["shared"] == row["shared"]
            pairs.append(dict(case=row["id"], original=row, candidate=changed,
                              condition="unspecified_neutral"))
            conditions = [("unspecified_neutral", changed["neutral"])]
        for label, lighting in candidates:
            assert lighting != definition.lighting
            changed = dict(row, **paired_prompts(Shared(**row["shared"]), replace(definition, lighting=lighting)))
            validate_pair(changed)
            assert changed["neutral"] == row["neutral"] and changed["shared"] == row["shared"]
            pair = dict(case=row["id"], original=row, candidate=changed)
            if args.round != "white-wording":
                pair["condition"] = label
            pairs.append(pair)
            conditions.append((label, changed["positive"]))
        for condition, prompt in conditions:
            request = Generation(prompt=prompt, seed=row["seeds"][0], width=512, height=512, steps=10, energy=0., mix={})
            payload = generation_payloads(request, store, identity)[0]
            payload.update(purpose="theatrical_diagnostic", case=row["id"], character=row["character"],
                           condition=condition, definition=row["definition"])
            if args.round != "white-wording":
                payload["diagnostic_round"] = args.round
            payloads.append(payload)
    manifest = dict(model=identity, pairs=pairs, training_unchanged=True, diagnostic_only=True)
    manifest["sha256"] = digest(manifest)
    path = root / f"diagnostics/theatrical-{args.round}/manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Diagnostic manifest changed")
    if not path.exists():
        atomic_json(path, manifest)
    print(json.dumps(store.enqueue(payloads, group="theatrical-diagnostic-" + digest(payloads))))


if __name__ == "__main__":
    main()
