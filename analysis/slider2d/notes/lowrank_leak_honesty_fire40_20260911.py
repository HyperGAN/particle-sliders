#!/usr/bin/env python3
"""Fire #40: reanalyze lowrank_soft_continuum under leak_max honesty gate.

No GPU. No new trains. Reads existing continuum JSON only.
"""
from __future__ import annotations
import json
from pathlib import Path

NOTES = Path(__file__).resolve().parent
LOCK = 0.20
SRC = NOTES / "lowrank_soft_continuum_20260910.json"
OUT = NOTES / "lowrank_leak_honesty_fire40_20260911.json"

def main():
    d = json.loads(SRC.read_text())
    # ... see companion .md; primary artifact written by Fire #40 runner
    print("see", OUT, "and .md — re-run Fire #40 notebook if regenerating")

if __name__ == "__main__":
    main()
