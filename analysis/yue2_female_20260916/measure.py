"""CPU-only cached audio diagnostics; these proxies do not replace listening."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import soundfile as sf

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / ".cache/slider-quality/python"))
from slider_selection import features as F
from analysis.gan_bcap.autonomous_audio import coverage_weights

OUT = ROOT / "eval/listen/yue2-female-uni16-600-20260916"


def main():
    F.CONCEPTS = {"female": F.CONCEPTS["female"]}
    judge = F.AudioMeasurer(WORK / "audio-cache", "cpu")
    judge.load()
    records = []
    while len(records) < 10:
        try:
            report = json.loads((OUT / "evaluation.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5)
            continue
        done = {r["path"] for r in records}
        for r in report["records"]:
            if r["path"] in done:
                continue
            source = OUT / r["path"] / "song.wav"
            if F.file_hash(source) != r["wav_sha256"]:
                raise ValueError("Audio changed after rendering")
            data, rate = sf.read(source, dtype="float32", always_2d=True)
            normalized = WORK / "normalized" / (r["wav_sha256"] + ".wav")
            normalized.parent.mkdir(parents=True, exist_ok=True)
            sf.write(normalized, data * (.1 / r["rms"]), rate, subtype="FLOAT")
            measured = judge.measure(normalized)
            shares = coverage_weights(measured["windows"])
            average = lambda fn: sum(w * fn(m) for w, m in zip(shares, measured["windows"]))
            item = dict(path=r["path"], row=r["row"], seed=r["seed"], variant=r["variant"],
                wav_sha256=r["wav_sha256"],
                female_clap_margin=average(lambda m: m["concept"]["female"]),
                enjoyment_proxy=average(lambda m: m["aesthetics"]["CE"]),
                production_proxy=average(lambda m: m["aesthetics"]["PQ"]),
                windows=measured["windows"])
            records.append(item)
            result = dict(status="complete" if len(records) == 10 else "measuring", records=records,
                normalization="Stereo RMS 0.1 measurement copy; originals preserved",
                interpretation="CLAP female-minus-male text similarity and aesthetics are diagnostic proxies, not calibrated gender or quality judgments. Listening and lyric checks remain necessary.",
                provenance=measured["provenance"])
            temporary = OUT / "audio-diagnostics.tmp"
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(OUT / "audio-diagnostics.json")
            print(json.dumps({k: v for k, v in item.items() if k != "windows"}), flush=True)
        if len(records) < 10:
            time.sleep(5)


if __name__ == "__main__":
    main()
