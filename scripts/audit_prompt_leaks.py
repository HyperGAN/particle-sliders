#!/usr/bin/env python3
"""Caption-leak auditor for slider prompt YAMLs.

Encodes the v7 caption policy as checks:

1. BPM pinning — target/positive/negative/neutral must carry the SAME explicit
   BPM in every row, except files whose axis IS tempo (`prompts-tempo*`).
   This is the leak that shipped energy/distortion taught the teacher to move
   tempo/loudness: pos/neg carried 168-vs-52 and 140-vs-88 BPM swings, so the
   teacher velocity delta g*(vel_pos - vel_neg) contained a tempo component
   the transformer LoRA faithfully learned.
2. Hybrid-neutral ban — neutrals must be real middle songs. Wording like
   "partly sung, partly spoken" / "halfway between talking and singing"
   describes BOTH poles at once and produces clips that are both (the
   rapslow failure mode).
3. Edge-file structure — `*-edge.yaml` single-direction controls must have
   negative == target in every row (that substitution is what turns the
   pos-neg axis into vel_pos - vel_neu); non-edge files must NOT.

Usage:
  python scripts/audit_prompt_leaks.py                # table for all prompts-*.yaml
  python scripts/audit_prompt_leaks.py --strict FILE [FILE ...]   # exit 1 on violation

Exit code 0 = clean, 1 = at least one violation (or missing BPM cell).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "conceptmod" / "textsliders" / "data"

BPM_RE = re.compile(r"BPM[:\s]+(\d{2,3})", re.IGNORECASE)
HYBRID_NEUTRAL_RE = re.compile(
    r"partly sung|partly spoken|partly rapped|halfway between (?:talking|speaking) and singing"
    r"|half rap|half-rap|between rap(?:ping)? and sing(?:ing)?",
    re.IGNORECASE,
)
CELLS = ("target", "positive", "negative", "neutral")


def _bpm(text: str | None) -> int | None:
    if not text:
        return None
    m = BPM_RE.search(text)
    return int(m.group(1)) if m else None


def audit_file(path: Path) -> list[str]:
    """Return a list of violation strings (empty = clean)."""
    problems: list[str] = []
    rel = path.relative_to(ROOT)
    is_tempo_axis = path.name.startswith("prompts-tempo")
    is_edge = ".edge" in path.stem or path.stem.endswith("-edge")
    # Uni-v1 is plus-only: no opposite pole. negative := target (yaml shape).
    is_uni = "-uni-" in path.stem
    is_plus_only = is_edge or is_uni

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict):
        rows = doc.get("rows") or []
    else:
        rows = []
    if not rows:
        return [f"{rel}: no rows"]

    for i, row in enumerate(rows):
        bpms = {cell: _bpm(row.get(cell)) for cell in CELLS}
        missing = [c for c, b in bpms.items() if b is None]
        if missing:
            problems.append(f"{rel} row {i}: missing BPM in {','.join(missing)}")
        values = {b for b in bpms.values() if b is not None}
        if not is_tempo_axis and len(values) > 1:
            spread = max(values) - min(values)
            problems.append(
                f"{rel} row {i}: BPM not pinned ({bpms['positive']} vs {bpms['negative']}, "
                f"spread {spread}) — teacher will learn a tempo component"
            )
        neutral_text = str(row.get("neutral") or "")
        if HYBRID_NEUTRAL_RE.search(neutral_text):
            problems.append(f"{rel} row {i}: hybrid-delivery neutral banned — write a real middle song")

        neg = str(row.get("negative") or "")
        tgt = str(row.get("target") or "")
        if is_plus_only and neg != tgt:
            kind = "uni-v1" if is_uni else "edge"
            problems.append(f"{rel} row {i}: {kind} file must set negative := target")
        if not is_plus_only and neg and tgt and neg == tgt:
            problems.append(f"{rel} row {i}: negative equals target outside an edge/uni control file")

    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="prompt YAMLs (default: every prompts-*.yaml under data/)")
    p.add_argument("--strict", action="store_true", help="exit 1 on any violation")
    args = p.parse_args(argv)

    paths = [Path(f) for f in args.files] or sorted(DATA.glob("prompts-*.yaml"))
    total = 0
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        problems = audit_file(path)
        status = "OK " if not problems else "FAIL"
        print(f"[{status}] {path.name}")
        for problem in problems:
            print(f"       {problem}")
        total += len(problems)

    print(f"\n{len(paths)} files, {total} violation(s)")
    if args.strict and total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
