#!/usr/bin/env python3
"""Audit existing LM transcripts without rendering, ASR, or changing scores.

The two added text statistics are diagnostics, not proposed acceptance gates.
No transcripts, captions, or lyric sheets are emitted. Uses only the stdlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def words(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("cached_lyrics.csv"))
    args = parser.parse_args()
    listen = ROOT / "eval/listen"
    labels = json.loads((ROOT / "eval/lm_score_labels.json").read_text())["labels"]
    sources = {listen / name / "lm_scores.json" for name in labels}
    sources.update((listen / "gan-bcap-repair").glob("*/lm_scores.json"))
    rows, skipped = [], []
    for path in sorted(sources):
        if not path.is_file():
            skipped.append(f"{path.parent.name}: no cached scores")
            continue
        payload = path.read_bytes()
        score = json.loads(payload)
        reference = words(score["lyrics"])
        if not reference:
            raise ValueError(f"Empty lyric sheet: {path}")
        vocabulary = set(reference)
        folder = str(path.parent.relative_to(listen))
        found = False
        for channel in score["channels"].values():
            if abs(channel["scale"]) != 1:
                continue
            found = True
            hypothesis = words(channel["text"])
            # Membership precision tolerates repeats, but cannot check order.
            precision = (sum(word in vocabulary for word in hypothesis)
                         / len(hypothesis)) if hypothesis else 0.0
            rows.append({
                "folder": folder,
                "locked_folder_label": labels.get(folder, {}).get("verdict", "UNLABELED"),
                "scale": channel["scale"],
                "frozen_folder_verdict": score["verdict"],
                "word_recall": channel["lyric_recall"],
                "sheet_vocabulary_precision": precision,
                "single_sheet_wer": edit_distance(reference, hypothesis) / len(reference),
                "reference_words": len(reference),
                "transcribed_words": len(hypothesis),
                "frozen_E": score.get("E"),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
            })
        if not found:
            skipped.append(f"{folder}: no exact unit-scale clip")
    if not rows:
        raise ValueError("No cached unit-scale measurements found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unit-scale rows from "
          f"{len({row['folder'] for row in rows})} folders to {args.out}")
    for reason in skipped:
        print(f"Skipped: {reason}")


if __name__ == "__main__":
    main()
