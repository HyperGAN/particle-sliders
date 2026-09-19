"""Warm content-addressed CPU metric caches as reference audio becomes ready."""
import json
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT))
from analysis.uni16_20260906.score import DESCRIPTIONS
from analysis.gan_bcap import autonomous_audio as audio
import yaml

audio.CONCEPTS = DESCRIPTIONS
judge = audio.Judge()
protocol = json.loads((WORK / "protocol.json").read_text())
rows = yaml.safe_load(Path(protocol["evaluation"]["prompts"]).read_text())["rows"]
page = ROOT / "eval/listen/fresh-seed-lofi-20260910"
saved = WORK / "reference-measurements.json"
records = json.loads(saved.read_text())["records"] if saved.exists() else []
seen = {r["sha256"] for r in records}
while True:
    markers = set(page.glob("prewarm-row-*/*/checkpoint.json")) | set(page.glob("row-*/*/checkpoint.json"))
    for marker in sorted(markers):
        row = int(marker.parent.parent.name.rsplit("-", 1)[1])
        for wav in sorted(marker.parent.glob("*.wav")):
            digest = audio.F.file_hash(wav)
            if digest in seen:
                continue
            measurement = judge.measure(wav, rows[row]["lyrics"], "lofi", 20.)
            records.append(measurement)
            seen.add(digest)
            audio.F.write_json(WORK / "reference-measurements.json", dict(records=records))
            print(f"Measured {len(records)} references; CE={measurement['enjoyment']:.3f}", flush=True)
    if len(list(page.glob("row-*/*/checkpoint.json"))) == 16:
        break
    time.sleep(10)
