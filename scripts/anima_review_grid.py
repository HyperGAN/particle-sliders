"""Build blind A/B development grids; keep image identities in a separate map."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variation", choices=["candlelit", "moonlit", "theatrical"])
    parser.add_argument("--step", type=int)
    parser.add_argument("--checkpoint", help="Exact immutable checkpoint hash when a run was revised")
    parser.add_argument("--pilot", action="store_true", help="Latest full-strength case for each pilot character")
    parser.add_argument("--unreviewed", action="store_true", help="Only cases without saved review notes")
    parser.add_argument("--purpose", choices=("development", "step_compatibility"), default="development")
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    parser.add_argument("--output", type=Path, default=Path("artifacts/anima/review"))
    args = parser.parse_args()

    def get(path):
        with urllib.request.urlopen(args.url + "/api/" + path, timeout=30) as response:
            return json.load(response)

    images = []
    offset = 0
    while True:
        page = get(f"history?limit=500&offset={offset}")
        images.extend(page)
        if len(page) < 500:
            break
        offset += 500
    if args.pilot and args.purpose != "development":
        raise ValueError("Pilot qualification uses development cadence renders")
    candidates = [i for i in images if i["metadata"].get("purpose") == args.purpose
                  and args.variation in i["metadata"]["checkpoints"] and i["metadata"]["energy"] == 1.]
    if args.step is not None:
        candidates = [i for i in candidates if i["metadata"]["sampling_step"] == args.step]
    if args.checkpoint:
        candidates = [i for i in candidates if i["metadata"]["checkpoints"][args.variation] == args.checkpoint]
    if args.pilot:
        candidates = [i for i in candidates if i["metadata"]["sampling_step"] <= 200]
        by_character = {}
        for image in sorted(candidates, key=lambda i: i["metadata"]["sampling_step"], reverse=True):
            by_character.setdefault(image["metadata"]["character"], image)
        candidates = list(by_character.values())
    if args.unreviewed:
        reviewed = {r["image_id"] for r in get("reviews")}
        candidates = [i for i in candidates if i["id"] not in reviewed]
    candidates.sort(key=lambda i: (i["metadata"]["case"], i["metadata"]["steps"]))
    if not candidates:
        raise ValueError("No full-strength development samples are ready")
    rng = random.SystemRandom()
    label = 'pilot' if args.pilot else args.step
    suffix = '' if args.purpose == 'development' else '-steps'
    destination = args.output / f"{args.variation}-{label}{suffix}"
    destination.mkdir(parents=True, exist_ok=True)
    mapping_path = destination / "mapping.json"
    previous = json.loads(mapping_path.read_text()) if mapping_path.exists() else None
    if previous is not None:
        previous_ids = [next(row[s]["id"] for s in ("A", "B") if row[s]["metadata"]["energy"] == 1.)
                        for row in previous]
        if previous_ids != [item["id"] for item in candidates]:
            raise ValueError("This blind grid is immutable; use a fresh output directory for changed coverage")
    pairs = []
    for index, image in enumerate(candidates):
        other = get("images/" + image["id"] + "/comparison")
        if other is None:
            raise ValueError(f"Matched Off image is not ready: {image['id']}")
        pair = [image, other]
        if previous is None:
            rng.shuffle(pair)
        else:
            order = [previous[index][s]["id"] for s in ("A", "B")]
            if set(order) != {item["id"] for item in pair}:
                raise ValueError("Matched comparison changed")
            pair.sort(key=lambda item: order.index(item["id"]))
        pairs.append(pair)
    mapping = []

    def bitmap(item):
        path = destination / (item["id"] + ".png")
        if not path.exists():
            with urllib.request.urlopen(args.url + "/api/images/" + item["id"], timeout=30) as response:
                path.write_bytes(response.read())
        return Image.open(io.BytesIO(path.read_bytes())).convert("RGB")

    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = list(pool.map(bitmap, [item for pair in pairs for item in pair]))
    for start in range(0, len(pairs), 4):
        chunk = pairs[start:start + 4]
        canvas = Image.new("RGB", (1048, len(chunk) * 554), "#15191d")
        draw = ImageDraw.Draw(canvas)
        for row, pair in enumerate(chunk):
            case_number = start + row + 1
            steps = pair[0]["metadata"]["steps"]
            draw.text((12, row * 554 + 6), f"{args.variation} | Case {case_number} | {steps} steps | A                            B", fill="white")
            for col, item in enumerate(pair):
                im = pixels[2 * (start + row) + col]
                im.thumbnail((512, 512))
                canvas.paste(im, (12 + col * 524 + (512 - im.width) // 2, row * 554 + 28))
            mapping.append(dict(case=case_number, A=pair[0], B=pair[1]))
        canvas.save(destination / f"blind-{start // 4 + 1}.jpg", quality=95)
    mapping_path.write_text(json.dumps(mapping, indent=2))
    template = destination / "ratings.template.json"
    if not template.exists():
        template.write_text(json.dumps(dict(mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            reviews=[dict(case=row["case"], preferred=None, blind=None, notes="",
                character=None, outfit=None, pose=None, composition=None, medium=None, quality=None,
                unwanted_objects=None, recurring_features=None, major_regression=None) for row in mapping]), indent=2))
    print(json.dumps(dict(directory=str(destination), cases=len(pairs),
                         grids=[str(p) for p in destination.glob('blind-*.jpg')])))


if __name__ == "__main__":
    main()
