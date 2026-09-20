"""Make blinded energy-sweep grids from a completed held-out Studio audit."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    parser.add_argument("--output", type=Path, default=Path("artifacts/anima/review/final"))
    parser.add_argument("--rows-per-grid", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()

    def get(path):
        with urllib.request.urlopen(args.url + "/api/" + path, timeout=30) as response:
            return json.load(response)

    images, offset = [], 0
    while True:
        page = get(f"history?limit=500&offset={offset}")
        images.extend(i for i in page if i["metadata"].get("purpose") in ("final_test", "mixture_audit"))
        if len(page) < 500:
            break
        offset += 500
    audits = {i["metadata"]["audit"] for i in images}
    if len(audits) != 1 or len(images) != 512:
        raise ValueError(f"Expected one completed 512-image audit; found {len(audits)} audits and {len(images)} images")
    groups = defaultdict(list)
    for image in images:
        m = image["metadata"]
        groups[(m["family"], m["case"], m["seed"])].append(image)
    if len(groups) != 128 or any(sorted(i["metadata"]["energy"] for i in group) != [0., .25, .5, 1.]
                                 for group in groups.values()):
        raise ValueError("Audit sweeps are incomplete or duplicated")

    args.output.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output / "mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        mapped_ids = {i["id"] for group in mapping["groups"] for i in group["columns"].values()}
        if mapped_ids != {i["id"] for i in images} or mapping["audit"] != next(iter(audits)):
            raise ValueError("Existing blind mapping belongs to different images")
    else:
        shuffled = list(groups.items())
        rng = random.SystemRandom()
        rng.shuffle(shuffled)
        mapping = dict(audit=next(iter(audits)), groups=[])
        for number, (key, group) in enumerate(shuffled, 1):
            rng.shuffle(group)
            mapping["groups"].append(dict(case=number, family=key[0],
                columns=dict(zip("ABCD", group))))
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")

    def fetch(image):
        path = args.output / (image["id"] + ".png")
        if not path.exists():
            with urllib.request.urlopen(args.url + "/api/images/" + image["id"], timeout=30) as response:
                path.write_bytes(response.read())
        return path

    ordered = [i for group in mapping["groups"] for i in group["columns"].values()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, ordered))
    # Do not expose energy, checkpoint, image ID, or source filename on a grid.
    for start in range(0, len(mapping["groups"]), args.rows_per_grid):
        chunk = mapping["groups"][start:start + args.rows_per_grid]
        canvas = Image.new("RGB", (1560, 434 * len(chunk)), "#15191d")
        draw = ImageDraw.Draw(canvas)
        for row, group in enumerate(chunk):
            draw.text((12, row * 434 + 6), f"Case {group['case']} | {group['family']}", fill="white")
            for col, (letter, item) in enumerate(group["columns"].items()):
                draw.text((12 + col * 390, row * 434 + 23), letter, fill="white")
                with Image.open(args.output / (item["id"] + ".png")) as source:
                    im = source.convert("RGB")
                    im.thumbnail((384, 384))
                    canvas.paste(im, (3 + col * 390 + (384 - im.width) // 2, row * 434 + 43))
        canvas.save(args.output / f"blind-{start // args.rows_per_grid + 1:03}.jpg", quality=95)
    form = args.output / "ratings.template.json"
    if not form.exists():
        form.write_text(json.dumps(dict(mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            reviews=[dict(case=g["case"], family=g["family"], blind=None,
            ratings={letter: dict(atmosphere_strength=None, character=None, outfit=None,
                pose=None, composition=None, medium=None, quality=None,
                unwanted_objects=None, recurring_features=None, major_regression=None,
                notes="") for letter in "ABCD"})
            for g in mapping["groups"]]), indent=2) + "\n")
    print(json.dumps(dict(directory=str(args.output), groups=len(groups), images=len(images),
        instructions="Inspect blind grids and record ratings before opening mapping.json. Review full-size images for ambiguous details.")))


if __name__ == "__main__":
    main()
