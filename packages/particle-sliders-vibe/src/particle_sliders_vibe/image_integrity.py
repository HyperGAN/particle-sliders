"""Anima structural check applied to one finished native image."""
from .image_judge import JEV

DEFECTS = {
    "no_readable_image": (
        "Has the render collapsed largely into texture, color, glare or noise, so a recognizable "
        "figure and coherent illustration are missing? A readable fantasy figure or a dim scene is normal."
    ),
    "split_or_duplication": (
        "Does the candidate visibly break into separate panels or repeat the main character "
        "because of an obvious render error?"
    ),
    "broken_structure": (
        "Is there a clear structural rendering failure, such as a duplicated head, fused or "
        "detached body parts, incoherent face or torso, torn geometry, or smeared shapes that "
        "erase structure? Require a specific visible defect. Stylized costumes, sharp armor, "
        "occlusion and cropping are normal."
    ),
}
STRUCTURE_DETAILS = {
    "accidental_parts": (
        "Is a body part obviously duplicated, attached in the wrong place or visibly separated "
        "from the figure by a drawing error? Count only visible evidence."
    ),
    "accidental_face": (
        "Is the visible face plainly malformed by extra eyes, misplaced features or disintegrating "
        "shapes? An intentional mask, stylized shading and expression are valid."
    ),
    "accidental_geometry": (
        "Does an obvious drawing error tear apart or smear the figure's contours so it ceases "
        "to read as a single formed figure? Coherent hair, armor and props can overlap."
    ),
}


def check(reader, image):
    checks = reader.decide(
        (image,), JEV,
        "Inspect the single finished image for visible rendering failures.",
        {
            name: {"type": "noul", "instructions": prompt}
            for name, prompt in DEFECTS.items()
        },
    )
    if checks is None:
        return {"status": "unmeasured", "checks": None}
    details = None
    for name, probability in list(checks.items()):
        if 0.5 <= probability < 0.8:
            if name == "broken_structure":
                details = reader.decide(
                    (image,), JEV,
                    "Inspect the candidate for specific visible drawing errors.",
                    {
                        key: {"type": "noul", "instructions": text}
                        for key, text in STRUCTURE_DETAILS.items()
                    },
                )
                checks[name] = max(details.values()) if details else None
            else:
                independently = reader.decide(
                    (image,), JEV,
                    "Look at the actual visible structures before deciding.",
                    {name: {"type": "noul", "instructions": DEFECTS[name]}},
                )
                checks[name] = independently[name] if independently else None
    known = [p for p in checks.values() if p is not None]
    status = (
        "fail"
        if any(p >= 0.8 for p in known)
        else (
            "pass" if len(known) == len(checks) and max(known) < 0.5 else "unmeasured"
        )
    )
    outcome = {"status": status, "checks": checks}
    if details:
        outcome["structural_details"] = details
    return outcome
