"""Inspect immutable frozen-teacher candidate pairs before preparing any targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    parser.add_argument("--output", type=Path, default=Path("artifacts/anima/definition-candidates/review"))
    args = parser.parse_args()

    def get(path):
        with urllib.request.urlopen(args.url + "/api/" + path, timeout=30) as response:
            return json.load(response)

    images, offset = [], 0
    while True:
        page = get(f"history?limit=500&offset={offset}")
        images.extend(i for i in page if i["metadata"].get("purpose") == "definition_candidate"
                      and i["metadata"].get("candidate") == args.candidate)
        if len(page) < 500:
            break
        offset += 500
    pairs = []
    for index in range(args.start, args.start + args.count):
        case = f"theatrical-{args.split}-{index:02}"
        pair = {}
        for side in ("neutral", "positive"):
            found = [i for i in images if i["metadata"]["case"] == case
                     and i["metadata"]["reference_side"] == side]
            if len(found) != 1:
                raise ValueError(f"Expected one completed {case}/{side}; got {len(found)}")
            pair[side] = found[0]
        a, b = (pair[side]["metadata"] for side in ("neutral", "positive"))
        keys = ("candidate", "manifest_sha256", "case", "definition", "character", "seed",
                "width", "height", "steps", "model_identity", "group_id")
        if any(a[k] != b[k] for k in keys) or a["checkpoints"] or b["checkpoints"] or a["energy"] or b["energy"]:
            raise ValueError("Unmatched frozen teacher pair")
        pairs.append(pair)
    prefix = "rows" if args.split == "train" else "dev-rows"
    destination = args.output / args.candidate[:12] / f"{prefix}-{args.start:02}-{args.start + args.count - 1:02}"
    destination.mkdir(parents=True, exist_ok=True)
    evidence = destination / "images.json"
    if evidence.exists() and json.loads(evidence.read_text()) != pairs:
        raise ValueError("Candidate review images changed")
    evidence.write_text(json.dumps(pairs, indent=2) + "\n")

    def bitmap(item):
        path = destination / (item["id"] + ".png")
        if not path.exists():
            with urllib.request.urlopen(args.url + "/api/images/" + item["id"], timeout=30) as response:
                path.write_bytes(response.read())
        return Image.open(path).convert("RGB")

    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = list(pool.map(bitmap, [p[s] for p in pairs for s in ("neutral", "positive")]))
    grids = []
    for start in range(0, len(pairs), 2):
        chunk = pairs[start:start + 2]
        canvas = Image.new("RGB", (1024, len(chunk) * 542), "#15191d")
        draw = ImageDraw.Draw(canvas)
        for row, pair in enumerate(chunk):
            for column, side in enumerate(("neutral", "positive")):
                draw.text((column * 512 + 8, row * 542 + 8),
                          pair[side]["metadata"]["case"] + " | " + side, fill="white")
                canvas.paste(pixels[2 * (start + row) + column], (column * 512, row * 542 + 30))
        path = destination / f"pairs-{start // 2 + 1}.jpg"
        canvas.save(path, quality=95)
        grids.append(str(path))
    print(json.dumps(dict(directory=str(destination), pairs=len(pairs), grids=grids)))


if __name__ == "__main__":
    main()
