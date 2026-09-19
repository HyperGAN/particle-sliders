#!/usr/bin/env python
"""Multi-seed confirm aggregator + re-rolled-zero null diagnostic.

Reads only artefacts that `scripts/lm_score.py` already wrote (per-folder
`lm_scores.json` and the content-hash embedding cache). It never re-measures
audio, never changes a threshold, and never writes into a listen folder.

Two jobs:

1. **aggregate** — for each candidate, read the U1 / U2 / U4 channel values at
   the unit rung `u` on seeds 7 / 11 / 13 and apply the FROZEN rules of
   `LM-SCORING.md` at that one rung. The per-seed folders contain only
   `{0, u}`, so their own folder verdict already reads at `u`; the seed-7
   folders carry extra rungs, so the rung is re-read here to compare like with
   like. Thresholds come from the scored folder's own `thresholds` block --
   this script has no threshold constants of its own.

2. **null** — the re-rolled-zero anchor. `LM-SCORING.md` ("What the calibration
   pass decided", item 1) says the uni null is structurally broken because no
   same-caption re-rolled zero render existed; the >=3-seed stage is exactly
   where one appears. For each candidate this takes the three scale-0 clips
   (seeds 7 / 11 / 13 -- same neutral caption, same lyrics, same checkpoint,
   different seed), reads their cached embeddings, and reports:

     honest_anchor = median cosine distance between the seed pairs of zeros
                     ("same song, different take")
     frozen anchor = cross_anchor from the scored folder (median cosine
                     distance between distinct *foreign-caption* pool clips)

   and whether `same_song[u]` sits below each. Disagreement between the two
   anchors is REPORTED, not acted on: U4's 0.90 x cross_anchor threshold is
   frozen and this script does not propose a replacement.

usage:
    python scripts/lm_confirm_null.py aggregate [--tsv out.tsv]
    python scripts/lm_confirm_null.py null
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIRM = REPO / "eval/listen/lm-confirm"
CACHE = REPO / "eval/lm_score_cache"
SEEDS = (7, 11, 13)

# (name, unit rung u, seed-7 folder holding the rung)
CANDIDATES = [
    ("breath-lm-uni-lyric", 0.75, "eval/listen/uni-lyric-derate/breath-lm-uni-lyric"),
    ("sexy-lm-uni-lyric", 0.75, "eval/listen/uni-lyric-derate/sexy-lm-uni-lyric"),
    ("gender-lm-uni-lyric", 0.75, "eval/listen/uni-lyric-derate/gender-lm-uni-lyric"),
    ("distortion-lm-uni-lyric", 0.25, "eval/listen/uni-lyric-derate/distortion-lm-uni-lyric"),
    ("energy-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/energy-lm-uni-lyric"),
    ("triphop-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/triphop-lm-uni-lyric"),
    ("tempo-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/tempo-lm-uni-lyric"),
    ("live-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/live-lm-uni-lyric"),
    ("rapslow-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/rapslow-lm-uni-lyric"),
    ("rhyme-prose-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/rhyme-prose-lm-uni-lyric"),
    ("joy-somber-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/joy-somber-lm-uni-lyric"),
    ("grit-smooth-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/grit-smooth-lm-uni-lyric"),
    ("tender-fierce-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/tender-fierce-lm-uni-lyric"),
    ("distortion-clean-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/distortion-clean-lm-uni-lyric"),
    ("gender-male-lm-uni-lyric", 1.0, "eval/listen/uni-lyric/gender-male-lm-uni-lyric"),
    ("hurt-lm-uni-prefix", 1.0, "eval/listen/uni-v3/hurt-lm-uni-prefix"),
    ("gender-lm-uni-v2", 1.0, "eval/listen/uni-v2/gender-lm-uni-v2"),
    ("breath-lm-uni-v2", 1.0, "eval/listen/uni-v2/breath-lm-uni-v2"),
    ("yearn-lm-uni-v2", 1.0, "eval/listen/uni-v2/yearn-lm-uni-v2"),
]


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------

def folder_for(name: str, seed: int, src7: str) -> Path:
    return REPO / (src7 if seed == 7 else f"eval/listen/lm-confirm/{name}-s{seed}")


def scores(folder: Path) -> dict | None:
    p = folder / "lm_scores.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_emb(path: Path) -> list[float] | None:
    """The embedding lm_score.py already measured for this exact file."""
    digest = sha1_file(path)
    hits = sorted(CACHE.glob(f"{digest}-v*.json"))
    if not hits:
        return None
    return json.loads(hits[-1].read_text(encoding="utf-8")).get("emb")


def cos_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / (na * nb) if na and nb else 0.0)


def rung_key(res: dict, u: float) -> str | None:
    key = f"{u:+g}"
    return key if key in res["channels"] else None


# --------------------------------------------------------------------------
# gate re-read at one rung, with the scored folder's own frozen thresholds
# --------------------------------------------------------------------------

def gates_at(res: dict, u: float) -> dict:
    T = res["thresholds"]
    key = rung_key(res, u)
    if key is None:
        return {"error": f"no rung {u:g} in {sorted(res['channels'])}"}
    c = res["channels"][key]
    zero = res["channels"]["+0"]

    # U1: rms floor at every scale + the ln-rms swing at the unit rung.
    rms_floor_ok = all(ch["rms_ratio"] >= T["rms_ratio_min"]
                       for ch in res["channels"].values() if ch["scale"] != 0)
    u1 = rms_floor_ok and abs(c["d_ln_rms"]) <= T["d_ln_rms_max"]

    # U2: absolute floor and the drop against max(base, REF+).
    rec = c["lyric_recall"]
    hold = max(zero["lyric_recall"], res["refs"]["ref_plus"]["lyric_recall"])
    u2 = rec >= T["recall_floor"] and rec >= hold - T["recall_drop"]

    # U4: cosine distance to this seed's own zero clip vs the frozen anchor.
    anchor = res["null"]["cross_anchor"]
    limit = T["same_song_frac"] * anchor
    u4 = c["same_song"] <= limit

    # U5 is a veto only at >= 45 s; below that it warns (LM-SCORING.md).
    u5_ok = bool(c["natural_end"]) or c["tail_ratio"] <= T["tail_ratio_max"]
    return {
        "rung": key,
        "U1": bool(u1), "U2": bool(u2), "U4": bool(u4),
        "U5": bool(u5_ok), "U5_is_veto": res["requested_s"] >= T["ending_veto_duration_s"],
        "recall": rec, "recall_hold": hold, "recall_floor": T["recall_floor"],
        "same_song": c["same_song"], "anchor": anchor, "same_frac": c["same_song"] / anchor,
        "dsp_proj": c["dsp_proj"], "proj": c["proj"],
        "d_ln_rms": c["d_ln_rms"], "rms_ratio": c["rms_ratio"],
        "duration": c["duration"], "natural_end": bool(c["natural_end"]),
        "tail_ratio": c["tail_ratio"], "requested_s": res["requested_s"],
        "text": c.get("text", ""),
    }


def status_of(per_seed: dict) -> tuple[str, list[str]]:
    """CONFIRMED-no-harm = U1/U2/U4 on all 3 seeds; NEAR = 2/3; else REFUTED."""
    fails = []
    ok = 0
    for seed in SEEDS:
        g = per_seed.get(seed)
        if g is None or "error" in g:
            fails.append(f"s{seed}:missing")
            continue
        bad = [k for k in ("U1", "U2", "U4") if not g[k]]
        if bad:
            fails.append(f"s{seed}:{'+'.join(bad)}")
        else:
            ok += 1
    if ok == 3:
        return "CONFIRMED-no-harm", fails
    if ok == 2:
        return "NEAR", fails
    return "REFUTED", fails


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_aggregate(args) -> int:
    rows = []
    for name, u, src7 in CANDIDATES:
        per_seed = {}
        for seed in SEEDS:
            res = scores(folder_for(name, seed, src7))
            per_seed[seed] = gates_at(res, u) if res else None
        status, fails = status_of(per_seed)
        end = scores(CONFIRM / f"{name}-end60")
        end_g = gates_at(end, u) if end else None
        rows.append({"name": name, "u": u, "seeds": per_seed,
                     "status": status, "fails": fails, "end60": end_g})
        signs = "".join("+" if (per_seed[s] and per_seed[s].get("dsp_proj", 0) > 0)
                        else ("-" if per_seed[s] else "?") for s in SEEDS)
        print(f"\n### {name} @ {u:g}   {status}   dsp sign(7/11/13)={signs}")
        for seed in SEEDS:
            g = per_seed[seed]
            if not g:
                print(f"  s{seed:<3} MISSING")
                continue
            print(f"  s{seed:<3} U1={'P' if g['U1'] else 'F'} U2={'P' if g['U2'] else 'F'} "
                  f"U4={'P' if g['U4'] else 'F'}  rec={g['recall']:.3f}(hold "
                  f"{max(g['recall_hold']-0.35, g['recall_floor']):.3f}) "
                  f"same/anch={g['same_frac']:.3f}  dsp={g['dsp_proj']:+.3f} "
                  f"proj={g['proj']:+.3f}  dur={g['duration']:.1f}s")
        if end_g:
            veto = "VETO" if end_g["U5_is_veto"] else "warn"
            print(f"  end60 U5={'PASS' if end_g['U5'] else 'FAIL'} ({veto}) "
                  f"natural_end={end_g['natural_end']} tail={end_g['tail_ratio']:.3f} "
                  f"dur={end_g['duration']:.1f}/{end_g['requested_s']:.0f}s")
        if fails:
            print(f"  failing: {', '.join(fails)}")
    if args.tsv:
        out = Path(args.tsv)
        cols = ["name", "u", "status"]
        for s in SEEDS:
            cols += [f"U1_s{s}", f"U2_s{s}", f"U4_s{s}", f"recall_s{s}",
                     f"same_frac_s{s}", f"dsp_s{s}"]
        cols += ["end60_U5", "end60_natural_end", "end60_tail", "end60_dur", "fails"]
        lines = ["\t".join(cols)]
        for r in rows:
            vals = [r["name"], f"{r['u']:g}", r["status"]]
            for s in SEEDS:
                g = r["seeds"][s]
                vals += ([str(g["U1"]), str(g["U2"]), str(g["U4"]),
                          f"{g['recall']:.4f}", f"{g['same_frac']:.4f}",
                          f"{g['dsp_proj']:+.4f}"] if g else [""] * 6)
            e = r["end60"]
            vals += ([str(e["U5"]), str(e["natural_end"]), f"{e['tail_ratio']:.4f}",
                      f"{e['duration']:.2f}"] if e else [""] * 4)
            vals.append(";".join(r["fails"]))
            lines.append("\t".join(vals))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def cmd_null(args) -> int:
    print("re-rolled-zero anchor: median cosine distance between the SAME-caption")
    print("scale-0 clips of seeds 7/11/13, against the frozen cross-caption anchor.\n")
    print(f"{'candidate':32s} {'honest':>7} {'frozen':>7} {'ratio':>6}  "
          f"{'same_song@u per seed (7/11/13)':>34}  verdict-flip")
    for name, u, src7 in CANDIDATES:
        zeros, anchors, same = {}, {}, {}
        for seed in SEEDS:
            f = folder_for(name, seed, src7)
            wav = f / "01_slider_neutral_base_zero.wav"
            if not wav.is_file():
                continue
            emb = cached_emb(wav)
            if emb:
                zeros[seed] = emb
            res = scores(f)
            if res:
                g = gates_at(res, u)
                if "error" not in g:
                    anchors[seed] = g["anchor"]
                    same[seed] = g["same_song"]
        pairs = [(a, b) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]
                 if a in zeros and b in zeros]
        if len(pairs) < 2 or not anchors:
            print(f"{name:32s} {'-':>7} {'-':>7} {'-':>6}  (incomplete)")
            continue
        dists = [cos_dist(zeros[a], zeros[b]) for a, b in pairs]
        honest = statistics.median(dists)
        frozen = statistics.median(anchors.values())
        cells, flips = [], []
        for seed in SEEDS:
            if seed not in same:
                cells.append("   -  ")
                continue
            below_h = same[seed] <= honest
            below_f = same[seed] <= 0.90 * frozen
            cells.append(f"{same[seed]:.4f}{'<' if below_h else '>'}h")
            if below_h != below_f:
                flips.append(f"s{seed}:{'honest-only' if below_h else 'frozen-only'}")
        print(f"{name:32s} {honest:7.4f} {frozen:7.4f} {honest/frozen:6.2f}  "
              f"{' '.join(cells):>34}  {','.join(flips) if flips else '-'}")
        if args.verbose:
            print("      pairwise zero distances: "
                  + ", ".join(f"{a}-{b}:{d:.4f}" for (a, b), d in zip(pairs, dists)))
    print("\nh = honest same-caption re-rolled-zero anchor (raw median, no 0.90 factor).")
    print("verdict-flip names seeds where 'below the honest anchor' and the FROZEN")
    print("U4 rule (same_song <= 0.90 x cross_anchor) disagree. Thresholds unchanged.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--tsv", default=None)
    a.set_defaults(fn=cmd_aggregate)
    n = sub.add_parser("null")
    n.add_argument("--verbose", action="store_true")
    n.set_defaults(fn=cmd_null)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
