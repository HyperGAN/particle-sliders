"""Gates: Music metal instability is prompt-row geometry, testable in 2-D.

Translates the 2026-09-17 Music 3 particle-bridge finding (female calm,
metal brittle after the particle-forward fix) into CPU checks:

1. Live uni16 YAMLs: female = tiny attribute insert; metal = genre rewrite.
2. Toy deltas: attribute-insert stays aligned; genre-rewrite does not.
3. Short particle-bridge toy: rewrite is the harder cell to lock.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from analysis.slider2d.genre_rewrite import (
    ATTR_MIN_PAIRWISE,
    MUSIC_FEMALE_MIN_PAIRWISE,
    MUSIC_METAL_MIN_PAIRWISE,
    PROMPTS,
    REWRITE_MAX_PAIRWISE,
    alignment_stats,
    attribute_insert_deltas,
    genre_rewrite_deltas,
    load_uni16_edits,
    rewrite_risk,
    run_particle_bridge_toy,
)

# The uni16 prompt pack is generated on the music workstation (where the
# parent ``app`` package is importable) and never committed.
needs_uni16_pack = pytest.mark.skipif(
    importlib.util.find_spec("app") is None and not PROMPTS.is_dir(),
    reason="needs the music workstation's uni16_fresh3400_20260912 prompt pack",
)


@needs_uni16_pack
def test_live_female_is_attribute_insert_metal_is_genre_rewrite():
    female = load_uni16_edits("female")
    metal = load_uni16_edits("metal")
    assert len(female) == len(metal) == 4
    assert max(e.mid_growth for e in female) <= 16
    assert min(e.shared_prefix for e in female) >= 200
    assert min(e.mid_growth for e in metal) >= 80
    assert max(e.shared_prefix for e in metal) <= 40


def test_music_measured_floors_still_separate_the_axes():
    # Documented Music 3 prepared-target floors (seed-7 catalog).
    assert MUSIC_FEMALE_MIN_PAIRWISE >= ATTR_MIN_PAIRWISE
    assert MUSIC_METAL_MIN_PAIRWISE <= REWRITE_MAX_PAIRWISE


def test_toy_attribute_insert_stays_aligned():
    torch.manual_seed(7)
    deltas = attribute_insert_deltas()
    assert alignment_stats(deltas)["min_pairwise"] >= ATTR_MIN_PAIRWISE
    assert rewrite_risk(deltas)["flag"] is False


def test_toy_genre_rewrite_flags_misaligned_rows():
    torch.manual_seed(7)
    rewrite = genre_rewrite_deltas()
    attr = attribute_insert_deltas()
    risk = rewrite_risk(rewrite)
    assert risk["min_pairwise"] <= REWRITE_MAX_PAIRWISE
    assert risk["flag"] is True
    assert risk["min_pairwise"] < alignment_stats(attr)["min_pairwise"] - 0.1


def test_particle_bridge_toy_locks_worse_on_genre_rewrite():
    """Music symptom: shared residual cannot serve misaligned row deltas.

    Short toy budget — look at late cos lock and generator effort, not the
    early transient (attribute-insert can dip once while climbing).
    """
    attr = run_particle_bridge_toy(attribute_insert_deltas(), steps=120, seed=0)
    rewrite = run_particle_bridge_toy(genre_rewrite_deltas(), steps=120, seed=0)
    assert attr["alignment"]["min_pairwise"] >= ATTR_MIN_PAIRWISE
    assert rewrite["alignment"]["min_pairwise"] <= REWRITE_MAX_PAIRWISE
    worse_lock = rewrite["cos_late_median"] < attr["cos_late_median"] - 0.08
    harder_g = rewrite["g_adv_late_median"] > attr["g_adv_late_median"] * 1.15
    late_wander = rewrite["cos_late_std"] > attr["cos_late_std"] * 1.25
    assert worse_lock or harder_g or late_wander, f"attr={attr} rewrite={rewrite}"


@needs_uni16_pack
def test_uni16_prompt_pack_exists_for_geometry_gate():
    assert (PROMPTS / "female-train.yaml").is_file()
    assert (PROMPTS / "metal-train.yaml").is_file()
