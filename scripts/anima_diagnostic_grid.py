"""Download immutable frozen-teacher wording diagnostics for visual review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round")
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    args = parser.parse_args()
    items, offset = [], 0
    while True:
        with urllib.request.urlopen(f"{args.url}/api/history?limit=500&offset={offset}", timeout=30) as response:
            page = json.load(response)
        items.extend(i for i in page if i["metadata"].get("purpose") == "theatrical_diagnostic"
                     and i["metadata"].get("diagnostic_round") == args.round)
        if len(page) < 500:
            break
        offset += 500
    if not items:
        raise ValueError("No completed diagnostic images")
    groups = {}
    for item in items:
        m = item["metadata"]
        if m["checkpoints"] or m["energy"]:
            raise ValueError("Diagnostic must use the frozen model")
        group = groups.setdefault(m["case"], {})
        if m["condition"] in group:
            raise ValueError("Duplicate diagnostic condition")
        group[m["condition"]] = item
    for group in groups.values():
        neutral = group["neutral"]["metadata"]
        for item in group.values():
            if any(item["metadata"][k] != neutral[k] for k in (
                    "seed", "width", "height", "steps", "model_identity", "group_id")):
                raise ValueError("Unmatched diagnostic settings")
    destination = Path("artifacts/anima/diagnostics") / ("theatrical-" + args.round)
    destination.mkdir(parents=True, exist_ok=True)
    evidence = destination / "images.json"
    ordered = sorted(items, key=lambda i: (i["metadata"]["case"], i["metadata"]["condition"]))
    if evidence.exists() and json.loads(evidence.read_text()) != ordered:
        raise ValueError("Diagnostic evidence changed")
    evidence.write_text(json.dumps(ordered, indent=2) + "\n")

    def download(item):
        path = destination / (item["id"] + ".png")
        if not path.exists():
            with urllib.request.urlopen(f"{args.url}/api/images/{item['id']}", timeout=30) as response:
                path.write_bytes(response.read())
        with Image.open(path) as im:
            return item["id"], im.convert("RGB").resize((384, 384))

    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = dict(pool.map(download, ordered))
    paths = []
    for case, group in sorted(groups.items()):
        conditions = ["neutral"] + sorted(k for k in group if k != "neutral")
        canvas = Image.new("RGB", (1152, 416 * ((len(conditions) + 2) // 3)), "#15191d")
        draw = ImageDraw.Draw(canvas)
        for i, condition in enumerate(conditions):
            x, y = (i % 3) * 384, (i // 3) * 416
            draw.text((x + 8, y + 6), case + " | " + condition, fill="white")
            canvas.paste(pixels[group[condition]["id"]], (x, y + 32))
        path = destination / (case + ".jpg")
        canvas.save(path, quality=95)
        paths.append(str(path))
    print(json.dumps(dict(images=len(items), grids=paths)))


if __name__ == "__main__":
    main()
