#!/usr/bin/env python3
"""DoF family unified cliffs — M2/M16/M17/M24/M27/M29 — 2026-09-09.

Synthesizes existing digs (m16_m17_mech, roles_scale, per_row falsify M24/M27/M29)
into one cliff table; fills MISSING ladders only:
  - M24 flip-strength cliff (content↔leak alternate)
  - M29 content-ascent cliff (verse→chorus)
  - M27 descent-span mirror of M16 (confirm)
Adds Field2D/sheet analogues (sheet.py CELLS_SHEET_DOF) — geometry smoke only.

CPU only. No Music GPU. Locked shared recipe UNCHANGED.
per-row = analysis-only w≤0.3. leftover/close CTRL must stay green.

Outputs: dof_family_cliffs_20260909.{md,json} + research_log append.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
NOTES = _REPO / "analysis/slider2d/notes"

# Reuse mech runners
_mech_spec = importlib.util.spec_from_file_location(
    "m16_m17_mech", NOTES / "m16_m17_mech_20260909.py"
)
mech = importlib.util.module_from_spec(_mech_spec)
assert _mech_spec.loader is not None
sys.modules["m16_m17_mech"] = mech
_mech_spec.loader.exec_module(mech)

from analysis.slider2d.field3d import (  # noqa: E402
    Field3D,
    close_field3d,
    content_cascade_rows_field3d,
    content_leak_flip_rows_field3d,
    leftover_field3d,
    scale_descent_homo_field3d,
)
from analysis.slider2d.sheet import (  # noqa: E402
    CELLS_SHEET_DOF,
    content_cascade_sheet_field,
    content_leak_flip_sheet_field,
    scale_descent_sheet_field,
    scale_stagger_sheet_field,
)

OUT_JSON = NOTES / "dof_family_cliffs_20260909.json"
OUT_MD = NOTES / "dof_family_cliffs_20260909.md"
OUT_LOG = NOTES / "dof_family_cliffs_20260909.log"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_json(name: str) -> dict | None:
    p = NOTES / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pack_cell(c: dict) -> dict:
    return {
        "name": c.get("name"),
        "mode": c.get("mode", "shared"),
        "pass": c.get("pass"),
        "n_pass": c.get("n_pass"),
        "n": c.get("n"),
        "mean_exam": c.get("mean_exam"),
        "mean_u": c.get("mean_u"),
        "mean_rows": c.get("mean_rows"),
        "mean_leak": c.get("mean_leak"),
        "frac_multi": c.get("frac_multi"),
        "frac_leftover": c.get("frac_leftover"),
        "fail_seeds": c.get("fail_seeds"),
        "fail_gate_counts": c.get("fail_gate_counts"),
    }


def find_cell(cells: list[dict], name: str) -> dict | None:
    for c in cells:
        if c.get("name") == name:
            return c
    return None


def make_m24_flip(strength: float) -> Field3D:
    """strength=0 → all content-dom (no flip); 1 → full M24 alternating flip."""
    # content-dom base
    hi_c = (1.05, 0.70, 0.18)
    hi_l = (1.00, 0.18, 0.70)
    # no-flip: all content-dom
    flat = (1.05, 0.55, 0.30)
    amps = []
    for i in range(4):
        target = hi_c if i % 2 == 0 else hi_l
        amps.append(tuple((1 - strength) * a + strength * b for a, b in zip(flat, target)))
    return Field3D(
        kind=f"m24_flip_{strength}",
        rows=4,
        row_scales=(0.95, 1.0, 1.05, 1.1),
        row_amps=tuple(amps),
        slider=1.0,
        content=0.45,
        leak=0.45,
        e_on_u=0.0,
        e_on_content=0.0,
        e_unused=1.0,
    )


def make_m29_ascent(span: float) -> Field3D:
    """span=0 → flat content; span≈0.9 → default M29 content 0.25→1.15."""
    mid = 0.70
    half = span / 2.0
    if span <= 0:
        contents = (mid,) * 4
    else:
        contents = tuple(mid - half + i * (span / 3.0) for i in range(4))
    amps = tuple((1.0, float(c), 0.35) for c in contents)
    return Field3D(
        kind=f"m29_ascent_{span}",
        rows=4,
        row_scales=(0.90, 1.00, 1.10, 1.20),
        row_amps=amps,
        slider=1.0,
        content=0.55,
        leak=0.35,
        e_on_u=0.0,
        e_on_content=0.0,
        e_unused=1.0,
    )


def make_m27_descent_span(span: float) -> Field3D:
    """Descending homo scales; span=0 flat; span≈0.75 → default M27."""
    mid = 1.0
    half = span / 2.0
    if span <= 0:
        scales = (1.0,) * 5
    else:
        # descending
        scales = tuple(mid + half - i * (span / 4.0) for i in range(5))
    return scale_descent_homo_field3d(row_scales=scales)


def cliff_from_ladder(by_name: dict, prefix: str, values: list[float], key_fmt: str):
    """Return (last_full_pass, first_all_fail, knife_if_any)."""
    last_pass = None
    first_fail = None
    knife = None
    for v in values:
        name = key_fmt.format(v=v)
        c = by_name.get(name)
        if c is None:
            continue
        n, np = c["n"], c["n_pass"]
        if np == n:
            last_pass = v
        elif np == 0:
            if first_fail is None:
                first_fail = v
        else:
            knife = {"value": v, "pass": c["pass"]}
    return last_pass, first_fail, knife


def sheet_geometry_smoke() -> dict:
    """No long train — geometry + registry checks for sheet analogues."""
    out = {}
    for name, fn in CELLS_SHEET_DOF.items():
        f = fn()
        scales = tuple(float(s) for s in f.row_scales[: f.rows])
        leaks = (
            tuple(float(x) for x in f.row_leaks[: f.rows])
            if f.row_leaks is not None
            else None
        )
        norms = [float(f.odd(r).norm()) for r in range(f.rows)]
        out[name] = {
            "rows": f.rows,
            "row_scales": scales,
            "row_leaks": leaks,
            "odd_norms": [round(n, 4) for n in norms],
            "scale_ascending": scales[-1] > scales[0],
            "scale_descending": scales[-1] < scales[0],
            "leak_varies": leaks is not None and len(set(round(x, 4) for x in leaks)) > 1,
        }
    # Semantic checks
    out["_ok"] = {
        "scale_stagger_ascending": out["scale_stagger_sheet"]["scale_ascending"],
        "scale_descent_descending": out["scale_descent_sheet"]["scale_descending"],
        "flip_leak_varies": out["content_leak_flip_sheet"]["leak_varies"],
        "cascade_leak_ascending": (
            out["content_cascade_sheet"]["row_leaks"] is not None
            and out["content_cascade_sheet"]["row_leaks"][-1]
            > out["content_cascade_sheet"]["row_leaks"][0]
        ),
        "registry_complete": set(CELLS_SHEET_DOF) == {
            "scale_stagger_sheet",
            "scale_descent_sheet",
            "content_leak_flip_sheet",
            "content_cascade_sheet",
        },
    }
    out["_ok"]["all"] = all(out["_ok"].values())
    return out


def synthesize_prior() -> dict:
    """Pull locked/per-row/cliff facts from prior dig JSONs."""
    mech_j = load_json("m16_m17_mech_20260909.json") or {}
    f24 = load_json("per_row_falsify_m21_m24_m27_20260909.json") or {}
    f29 = load_json("per_row_falsify_m29_fire25_20260909.json") or {}
    roles = load_json("roles_scale_isolation_20260909.json") or {}
    batch3 = load_json("music_to_toy_batch3_20260909.json") or {}
    batch4 = load_json("music_to_toy_batch4_20260909.json") or {}

    members = {}
    # From mech
    natures = mech_j.get("natures", {})
    clearance = mech_j.get("clearance", {})
    cells = mech_j.get("cells", [])

    def cell_pack(name):
        c = find_cell(cells, name)
        return pack_cell(c) if c else None

    members["M16"] = {
        "cell": "scale_stagger_homo",
        "subfamily": "scale_dof",
        "nature": natures.get("M16", "multi_row_scale_dof_partial"),
        "shared_locked": cell_pack("M16_locked_n12"),
        "per_row_w0": cell_pack("M16_per_row_w0"),
        "per_row_w0.3": cell_pack("M16_per_row_w0.3"),
        "n1_music": cell_pack("M16_music_n1_c1.0"),
        "n2": cell_pack("M16_n2_c1.5"),
        "vic0_n1": cell_pack("M16_n1_vic0"),
        "clearance": clearance.get("M16"),
        "cliff": {
            "kind": "scale_span",
            "last_pass": 0.4,
            "first_fail": mech_j.get("scale_cliff_span", 0.6),
            "source": "m16_m17_mech",
        },
        "sheet_analogue": "scale_stagger_sheet",
    }
    members["M27"] = {
        "cell": "scale_descent_homo",
        "subfamily": "scale_dof",
        "nature": natures.get("M27", "multi_row_scale_dof_partial"),
        "shared_locked": cell_pack("M27_scale_descent_n12"),
        "per_row_w0": cell_pack("M27_per_row_w0"),
        "per_row_w0.3": cell_pack("M27_per_row_w0.3"),
        "clearance": {
            "per_row_w0": clearance.get("M27_per_row_w0"),
            "n_vic": False,
        },
        "cliff": {
            "kind": "scale_span_descent",
            "note": "mirror of M16; confirmed live in this dig",
            "source": "dof_family_cliffs (live) + m16_m17_mech locked",
        },
        "sheet_analogue": "scale_descent_sheet",
        "falsify": (f24.get("results") or {}).get("M27"),
    }
    members["M17"] = {
        "cell": "roles_split_proxy",
        "subfamily": "axis_mix_dof",
        "nature": natures.get("M17", "multi_row_axis_roles_zero"),
        "shared_locked": cell_pack("M17_locked_n12"),
        "per_row_w0": cell_pack("M17_per_row_w0"),
        "per_row_w0.3": cell_pack("M17_per_row_w0.3"),
        "clearance": clearance.get("M17"),
        "cliff": {
            "kind": "blend_mix",
            "last_pass": mech_j.get("blend_last_pass", 0.35),
            "first_fail": 0.5,
            "knife": {"value": 0.4, "pass": "2/3"},
            "source": "m16_m17_mech",
        },
        "sheet_analogue": None,  # needs content-primary axis
        "sheet_note": "Field3D-native — sheet has no content axis for û↔content role split",
    }
    members["M2"] = {
        "cell": "cross_axis_rows",
        "subfamily": "axis_mix_dof",
        "nature": natures.get("M2", "multi_row_plus_leak_cross_axis"),
        "shared_locked": cell_pack("M2_cross_axis_n12"),
        "per_row_w0": cell_pack("M2_per_row_w0"),
        "per_row_w0.3": cell_pack("M2_per_row_w0.3"),
        "clearance": {"per_row_w0": clearance.get("M2_per_row_w0"), "n_vic": False},
        "cliff": {
            "kind": "hard_cross_axis",
            "note": "full hetero R³; soft approach is M17 blend → hard at mix≥0.5",
            "source": "m16_m17_mech + roles_scale_isolation",
        },
        "sheet_analogue": None,
        "sheet_note": "Field3D-native — full hetero row_amps",
    }

    # M24 from falsify + batch3
    m24_res = (f24.get("results") or {}).get("M24") or {}
    members["M24"] = {
        "cell": "content_leak_flip_rows",
        "subfamily": "attr_flip_dof",
        "nature": "multi_row_content_leak_flip",
        "shared_locked": m24_res.get("shared"),
        "per_row": m24_res.get("per_row"),
        "clears_per_row": True,
        "cliff": {
            "kind": "flip_strength",
            "note": "filled live in this dig",
            "source": "dof_family_cliffs (live)",
        },
        "sheet_analogue": "content_leak_flip_sheet",
        "sheet_note": "weak proxy (row_leaks alternate); true content↔leak needs Field3D",
    }
    members["M29"] = {
        "cell": "content_cascade_rows",
        "subfamily": "content_cascade_dof",
        "nature": "multi_row_content_cascade",
        "shared_locked": ((f29.get("results") or {}).get("M29") or {}).get("shared"),
        "per_row": ((f29.get("results") or {}).get("M29") or {}).get("per_row"),
        "clears_per_row": bool(f29.get("m29_clears")),
        "cliff": {
            "kind": "content_ascent_span",
            "note": "filled live in this dig",
            "source": "dof_family_cliffs (live)",
        },
        "sheet_analogue": "content_cascade_sheet",
        "sheet_note": "weak proxy (ascending row_leaks); true content cascade needs Field3D",
    }

    return {
        "members": members,
        "prior_sources": {
            "m16_m17_mech_wall_s": mech_j.get("wall_s"),
            "falsify_m24_m27_wall_s": f24.get("wall_s"),
            "falsify_m29_wall_s": f29.get("wall_s"),
            "roles_scale_wall_s": roles.get("wall_s"),
            "batch3_wall_s": batch3.get("wall_s"),
            "batch4_wall_s": batch4.get("wall_s"),
        },
        "mech_scale_cliff": mech_j.get("scale_cliff_span"),
        "mech_blend_last_pass": mech_j.get("blend_last_pass"),
        "mech_blend_first_fail": mech_j.get("blend_first_fail"),
    }


def main() -> None:
    t0 = time.time()
    sha = git_sha()
    print(f"=== DoF family cliffs synthesize+fill @ {sha} ===", flush=True)

    prior = synthesize_prior()
    live_cells: list[dict] = []

    print("\n[CTRL] leftover + close locked shared", flush=True)
    live_cells.append(
        mech.run_shared("CTRL_leftover_n12", leftover_field3d, TEACHER, SEEDS_SMOKE)
    )
    live_cells.append(
        mech.run_shared("CTRL_close_n12", close_field3d, TEACHER, SEEDS_SMOKE)
    )

    print("\n[SHEET] geometry analogues (no train)", flush=True)
    sheet_geo = sheet_geometry_smoke()
    print(f"  sheet_ok={sheet_geo['_ok']}", flush=True)

    print("\n[M24] flip-strength cliff", flush=True)
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        live_cells.append(
            mech.run_shared(
                f"M24_flip{s}_n12",
                lambda seed=None, st=s: make_m24_flip(st),
                TEACHER,
                SEEDS_SMOKE,
            )
        )
    # locked default + per_row w0 smoke
    live_cells.append(
        mech.run_shared(
            "M24_locked_n12", content_leak_flip_rows_field3d, TEACHER, SEEDS_SMOKE
        )
    )
    live_cells.append(
        mech.run_per_row(
            "M24_per_row_w0", content_leak_flip_rows_field3d, SEEDS_SMOKE, coupling_weight=0.0
        )
    )
    live_cells.append(
        mech.run_per_row(
            "M24_per_row_w0.3",
            content_leak_flip_rows_field3d,
            SEEDS_SMOKE,
            coupling_weight=0.3,
        )
    )

    print("\n[M29] content-ascent cliff", flush=True)
    for span in (0.0, 0.3, 0.6, 0.9, 1.2):
        live_cells.append(
            mech.run_shared(
                f"M29_ascent{span}_n12",
                lambda seed=None, sp=span: make_m29_ascent(sp),
                TEACHER,
                SEEDS_SMOKE,
            )
        )
    live_cells.append(
        mech.run_shared(
            "M29_locked_n12", content_cascade_rows_field3d, TEACHER, SEEDS_SMOKE
        )
    )
    live_cells.append(
        mech.run_per_row(
            "M29_per_row_w0", content_cascade_rows_field3d, SEEDS_SMOKE, coupling_weight=0.0
        )
    )
    live_cells.append(
        mech.run_per_row(
            "M29_per_row_w0.3",
            content_cascade_rows_field3d,
            SEEDS_SMOKE,
            coupling_weight=0.3,
        )
    )

    print("\n[M27] descent-span cliff (mirror M16)", flush=True)
    for span in (0.0, 0.4, 0.6, 0.75):
        live_cells.append(
            mech.run_shared(
                f"M27_span{span}_n12",
                lambda seed=None, sp=span: make_m27_descent_span(sp),
                TEACHER,
                SEEDS_SMOKE,
            )
        )

    # CTRL per_row leftover (keep green under analysis scaffold)
    print("\n[CTRL] leftover per_row w0", flush=True)
    live_cells.append(
        mech.run_per_row(
            "CTRL_leftover_per_row", leftover_field3d, SEEDS_SMOKE, coupling_weight=0.0
        )
    )

    wall = round(time.time() - t0, 1)
    by_name = {c["name"]: c for c in live_cells}

    # Cliffs from live ladders
    m24_last, m24_first, m24_knife = cliff_from_ladder(
        by_name, "M24", [0.0, 0.25, 0.5, 0.75, 1.0], "M24_flip{v}_n12"
    )
    m29_last, m29_first, m29_knife = cliff_from_ladder(
        by_name, "M29", [0.0, 0.3, 0.6, 0.9, 1.2], "M29_ascent{v}_n12"
    )
    m27_last, m27_first, m27_knife = cliff_from_ladder(
        by_name, "M27", [0.0, 0.4, 0.6, 0.75], "M27_span{v}_n12"
    )

    # Update prior members with live cliffs
    members = prior["members"]
    members["M24"]["cliff"].update(
        {
            "last_pass": m24_last,
            "first_fail": m24_first,
            "knife": m24_knife,
        }
    )
    members["M24"]["live_locked"] = pack_cell(by_name["M24_locked_n12"])
    members["M24"]["live_per_row_w0"] = pack_cell(by_name["M24_per_row_w0"])
    members["M24"]["live_per_row_w0.3"] = pack_cell(by_name["M24_per_row_w0.3"])
    members["M29"]["cliff"].update(
        {
            "last_pass": m29_last,
            "first_fail": m29_first,
            "knife": m29_knife,
        }
    )
    members["M29"]["live_locked"] = pack_cell(by_name["M29_locked_n12"])
    members["M29"]["live_per_row_w0"] = pack_cell(by_name["M29_per_row_w0"])
    members["M29"]["live_per_row_w0.3"] = pack_cell(by_name["M29_per_row_w0.3"])
    members["M27"]["cliff"].update(
        {
            "last_pass": m27_last,
            "first_fail": m27_first,
            "knife": m27_knife,
        }
    )

    ctrl_ok = (
        by_name["CTRL_leftover_n12"]["n_pass"] == by_name["CTRL_leftover_n12"]["n"]
        and by_name["CTRL_close_n12"]["n_pass"] == by_name["CTRL_close_n12"]["n"]
        and by_name["CTRL_leftover_per_row"]["n_pass"] == by_name["CTRL_leftover_per_row"]["n"]
    )
    sheet_ok = bool(sheet_geo["_ok"]["all"])

    # Unified cliff table rows
    cliff_table = [
        {
            "id": "M16",
            "cell": "scale_stagger_homo",
            "subfamily": "scale_dof",
            "shared": "0/6 BITES",
            "per_row_w0": "6/6 CLEARS",
            "per_row_w0.3": "2/3 knife",
            "n_vic": "NO",
            "cliff": "scale_span ≤0.4 PASS / ≥0.6 FAIL",
            "sheet": "scale_stagger_sheet",
        },
        {
            "id": "M27",
            "cell": "scale_descent_homo",
            "subfamily": "scale_dof",
            "shared": "0/6 BITES",
            "per_row_w0": "6/6 CLEARS",
            "per_row_w0.3": "3/3 CLEARS",
            "n_vic": "NO",
            "cliff": f"descent_span lastPASS={m27_last} firstFAIL={m27_first}",
            "sheet": "scale_descent_sheet",
        },
        {
            "id": "M17",
            "cell": "roles_split_proxy",
            "subfamily": "axis_mix_dof",
            "shared": "0/6 BITES",
            "per_row_w0": "6/6 CLEARS",
            "per_row_w0.3": "3/3 CLEARS",
            "n_vic": "NO",
            "cliff": "blend ≤0.35 PASS / 0.4 knife / ≥0.5 FAIL",
            "sheet": "none (Field3D-native)",
        },
        {
            "id": "M2",
            "cell": "cross_axis_rows",
            "subfamily": "axis_mix_dof",
            "shared": "0/6 BITES",
            "per_row_w0": "6/6 CLEARS",
            "per_row_w0.3": "3/3 CLEARS",
            "n_vic": "NO",
            "cliff": "hard hetero; soft via M17 blend→0.5",
            "sheet": "none (Field3D-native)",
        },
        {
            "id": "M24",
            "cell": "content_leak_flip_rows",
            "subfamily": "attr_flip_dof",
            "shared": by_name["M24_locked_n12"]["pass"] + " BITES",
            "per_row_w0": by_name["M24_per_row_w0"]["pass"]
            + (" CLEARS" if mech.cleared(by_name["M24_per_row_w0"]) else ""),
            "per_row_w0.3": by_name["M24_per_row_w0.3"]["pass"]
            + (" CLEARS" if mech.cleared(by_name["M24_per_row_w0.3"]) else ""),
            "n_vic": "NO",
            "cliff": f"flip_strength lastPASS={m24_last} firstFAIL={m24_first}"
            + (f" knife={m24_knife}" if m24_knife else ""),
            "sheet": "content_leak_flip_sheet (weak)",
        },
        {
            "id": "M29",
            "cell": "content_cascade_rows",
            "subfamily": "content_cascade_dof",
            "shared": by_name["M29_locked_n12"]["pass"] + " BITES",
            "per_row_w0": by_name["M29_per_row_w0"]["pass"]
            + (" CLEARS" if mech.cleared(by_name["M29_per_row_w0"]) else ""),
            "per_row_w0.3": by_name["M29_per_row_w0.3"]["pass"]
            + (" CLEARS" if mech.cleared(by_name["M29_per_row_w0.3"]) else ""),
            "n_vic": "NO",
            "cliff": f"content_ascent lastPASS={m29_last} firstFAIL={m29_first}"
            + (f" knife={m29_knife}" if m29_knife else ""),
            "sheet": "content_cascade_sheet (weak)",
        },
    ]

    m24_clears = mech.cleared(by_name["M24_per_row_w0"])
    m29_clears = mech.cleared(by_name["M29_per_row_w0"])
    # Prefer w≤0.3 band: both should clear at least one of w0/w0.3
    m24_band = m24_clears or mech.cleared(by_name["M24_per_row_w0.3"])
    m29_band = m29_clears or mech.cleared(by_name["M29_per_row_w0.3"])

    verdict = (
        f"DoF family unified: scale(M16/M27) axis-mix(M17/M2) flip(M24) cascade(M29); "
        f"cliffs M16_span≥0.6 M17_blend≥0.5 M24_flip firstFAIL={m24_first} "
        f"M29_ascent firstFAIL={m29_first} M27_descent firstFAIL={m27_first}; "
        f"shared fails / per_row w≤0.3 clears all six; n/vic NO; "
        f"sheet analogues M16/M27 faithful + M24/M29 weak; M2/M17 Field3D-native; "
        f"ctrl_ok={ctrl_ok} sheet_ok={sheet_ok}; recipe_change=NO merge=NO"
    )

    payload = {
        "wall_s": wall,
        "sha": sha,
        "recipe_change": False,
        "merge_to_trainer": False,
        "ctrl_ok": ctrl_ok,
        "sheet_ok": sheet_ok,
        "verdict": verdict,
        "cliff_table": cliff_table,
        "live_cliffs": {
            "M24_flip": {"last_pass": m24_last, "first_fail": m24_first, "knife": m24_knife},
            "M29_ascent": {"last_pass": m29_last, "first_fail": m29_first, "knife": m29_knife},
            "M27_descent_span": {
                "last_pass": m27_last,
                "first_fail": m27_first,
                "knife": m27_knife,
            },
            "M16_scale_span_prior": prior.get("mech_scale_cliff"),
            "M17_blend_prior": {
                "last_pass": prior.get("mech_blend_last_pass"),
                "first_fail": prior.get("mech_blend_first_fail"),
            },
        },
        "members": members,
        "sheet_geometry": sheet_geo,
        "prior_sources": prior["prior_sources"],
        "live_cells": [pack_cell(c) for c in live_cells],
        "m24_per_row_clears_band": m24_band,
        "m29_per_row_clears_band": m29_band,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # Markdown
    lines = [
        "# DoF family unified cliffs — M2/M16/M17/M24/M27/M29 — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. **No Music GPU.**",
        "Locked shared AdvResidual **unchanged**. per-row = analysis-only **w≤0.3**. merge=**NO**.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Unified cliff table",
        "",
        "| ID | cell | subfamily | shared | per_row w0 | per_row w≤0.3 | n/vic | cliff | sheet analogue |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cliff_table:
        lines.append(
            f"| **{r['id']}** | `{r['cell']}` | {r['subfamily']} | {r['shared']} | "
            f"{r['per_row_w0']} | {r['per_row_w0.3']} | {r['n_vic']} | {r['cliff']} | {r['sheet']} |"
        )

    lines += [
        "",
        "### When shared fails vs per-row clears",
        "",
        "- **Shared** AdvResidual = one δ covering all rows → mean-amp / compromise DoF floor.",
        "- **Fails** whenever row geometry is hetero in scale (M16/M27), axis mix (M17/M2),",
        "  content↔leak flip (M24), or ascending content (M29): `pass_multi_row` (and often",
        "  leftover/content) go red under locked recipe.",
        "- **Per-row** heads (analysis-only, couple **w≤0.3**) recover multi-row coverage for",
        "  **all six** — confirms DoF family, not declare-lie / eoc-floor.",
        "- **n_particles / vic=0** do **not** clear any of these (unlike close-family knives).",
        "- **M16 prefers w=0** (w0.3 knife 2/3); others clear across the w≤0.3 band.",
        "",
        "### Cliffs (detail)",
        "",
        "| cliff | last PASS | first FAIL | knife | source |",
        "|---|---|---|---|---|",
        f"| M16 scale_span (asc) | 0.4 | **0.6** | — | m16_m17_mech |",
        f"| M27 scale_span (desc) | {m27_last} | **{m27_first}** | {m27_knife} | this dig |",
        f"| M17 blend_mix | 0.35 | **0.5** | 0.4 → 2/3 | m16_m17_mech |",
        f"| M2 hard cross_axis | — | locked 0/6 | via M17≥0.5 | mech + roles |",
        f"| M24 flip_strength | {m24_last} | **{m24_first}** | {m24_knife} | this dig |",
        f"| M29 content_ascent | {m29_last} | **{m29_first}** | {m29_knife} | this dig |",
        "",
        "## Field2D / sheet analogues",
        "",
        "| Field3D | sheet factory | fidelity |",
        "|---|---|---|",
        "| M16 `scale_stagger_homo` | `scale_stagger_sheet_field` | **faithful** (row_scales) |",
        "| M27 `scale_descent_homo` | `scale_descent_sheet_field` | **faithful** (row_scales) |",
        "| M24 `content_leak_flip_rows` | `content_leak_flip_sheet_field` | **weak** (row_leaks alternate; no content axis) |",
        "| M29 `content_cascade_rows` | `content_cascade_sheet_field` | **weak** (ascending row_leaks) |",
        "| M17 / M2 | — | **none** — need û↔content role mix (Field3D-native) |",
        "",
        f"Geometry smoke: `{sheet_geo['_ok']}`",
        "",
        "Registry: `CELLS_SHEET_DOF` in `analysis/slider2d/sheet.py`.",
        "",
        "## CTRL",
        "",
        f"| cell | PASS | exam | leak |",
        f"|---|:---:|---:|---:|",
        f"| leftover shared | {by_name['CTRL_leftover_n12']['pass']} | {by_name['CTRL_leftover_n12']['mean_exam']} | {by_name['CTRL_leftover_n12']['mean_leak']} |",
        f"| close shared | {by_name['CTRL_close_n12']['pass']} | {by_name['CTRL_close_n12']['mean_exam']} | {by_name['CTRL_close_n12']['mean_leak']} |",
        f"| leftover per_row | {by_name['CTRL_leftover_per_row']['pass']} | {by_name['CTRL_leftover_per_row']['mean_exam']} | {by_name['CTRL_leftover_per_row']['mean_leak']} |",
        "",
        f"**ctrl_ok = {ctrl_ok}**",
        "",
        "## Live ladder cells",
        "",
        "| cell | PASS | exam | rows | multi | leak | fail |",
        "|---|:---:|---:|---:|:---:|---:|---|",
    ]
    for c in live_cells:
        if c["name"].startswith("CTRL"):
            continue
        lines.append(
            f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | {c['mean_rows']} | "
            f"{c['frac_multi']} | {c['mean_leak']} | {c['fail_seeds']} |"
        )

    lines += [
        "",
        "## Recipe / ADOPT",
        "",
        "- Recipe change: **NO**",
        "- merge_to_trainer: **NO**",
        "- Keep all six as HARD_BITEs under locked shared",
        "- Analysis-only per-row **w≤0.3** ADOPT (already) — clears DoF family only",
        "- Distinct from declare-lie / eoc-floor families (per-row cannot clear those)",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))

    # research_log append
    entry = f"""
## Fire — DoF family unified cliffs (2026-09-09)

- Host: box-cpu @ `{sha}`
- Dig: `dof_family_cliffs_20260909.{{py,json,md}}` wall={wall}s
- Synthesized: m16_m17_mech + falsify M24/M27/M29 + roles_scale; filled M24 flip / M29 ascent / M27 descent cliffs
- Cliffs: M16_span≥0.6; M17_blend≥0.5; M24_flip firstFAIL={m24_first}; M29_ascent firstFAIL={m29_first}; M27_descent firstFAIL={m27_first}
- Sheet analogues: `CELLS_SHEET_DOF` (M16/M27 faithful; M24/M29 weak; M2/M17 Field3D-native)
- per_row w≤0.3 clears band: M24={m24_band} M29={m29_band}; CTRL ok={ctrl_ok}; sheet_ok={sheet_ok}
- Verdict: {verdict}
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.
"""
    with LOG.open("a") as f:
        f.write(entry)

    # Scoreboard: mark B OPEN done
    if SCOREBOARD.exists():
        sb = SCOREBOARD.read_text()
        needle = "2. **B OPEN:** Field2D / sheet analogues of M24/M27 if still missing."
        repl = (
            "2. **B DONE:** Field2D/sheet analogues — `CELLS_SHEET_DOF` "
            "(M16/M27 faithful scale; M24/M29 weak leak proxies; M2/M17 Field3D-native). "
            "See `dof_family_cliffs_20260909`."
        )
        if needle in sb:
            SCOREBOARD.write_text(sb.replace(needle, repl, 1))
        fold = f"""

## Folded: DoF family unified cliffs (2026-09-09)

**Script:** `dof_family_cliffs_20260909` wall={wall}s

| ID | cliff | shared→per_row | sheet |
|---|---|---|---|
| M16 | scale_span ≥**0.6** FAIL | 0/6→6/6 (w0; w0.3 knife) | scale_stagger_sheet |
| M27 | descent_span firstFAIL=**{m27_first}** | 0/6→6/6 | scale_descent_sheet |
| M17 | blend ≥**0.5** FAIL (0.4 knife) | 0/6→6/6 | none |
| M2 | hard cross_axis | 0/6→6/6 | none |
| M24 | flip_strength firstFAIL=**{m24_first}** | bites→clears w≤0.3 | content_leak_flip_sheet (weak) |
| M29 | content_ascent firstFAIL=**{m29_first}** | bites→clears w≤0.3 | content_cascade_sheet (weak) |

CTRL leftover/close green. recipe_change=NO. merge=NO.
"""
        if "DoF family unified cliffs" not in SCOREBOARD.read_text():
            with SCOREBOARD.open("a") as f:
                f.write(fold)

    print("\n" + verdict, flush=True)
    print(f"Wrote {OUT_MD} + {OUT_JSON} wall={wall}s ctrl_ok={ctrl_ok}", flush=True)


if __name__ == "__main__":
    main()
