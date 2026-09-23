"""Pin the provisional formulation overlay to the Hub golden record on main.

The exam module imports the YuE2 trainer, so this test reads ``V2_SPEC`` from
its source instead of importing it. CPU only. Architecture stays gmix.
``particle-gmix-1600-v2`` is only the current parameter overlay.
"""
import ast
from pathlib import Path

from particle_sliders.formulation import (
    ARCHITECTURE_SPEC_KEYS,
    CURRENT_FORMULATION,
    CURRENT_FORMULATION_PROVISIONAL,
    RESEARCH_EXAM_IS_PROPOSE_ONLY,
    particle_gmix_1600_v2,
    winning_formulation,
)

_EXAM = Path(__file__).resolve().parents[1] / "analysis" / "slider2d" / "yue2_gmix_v2_exam.py"


def _v2_spec():
    tree = ast.parse(_EXAM.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "V2_SPEC" for target in node.targets
        ):
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, str(_EXAM), "exec"), {"dict": dict}, namespace)
            return namespace["V2_SPEC"]
    raise AssertionError("V2_SPEC assignment missing from yue2_gmix_v2_exam.py")


def test_winning_stamp_matches_v2_spec():
    spec = _v2_spec()
    stamp = winning_formulation()
    assert CURRENT_FORMULATION is particle_gmix_1600_v2
    assert CURRENT_FORMULATION_PROVISIONAL is True
    assert stamp.formulation_provisional is True
    assert stamp.architecture_id == "gmix"
    assert stamp.formulation_id == "particle-gmix-1600-v2"
    assert set(stamp.spec) == set(spec) - {"propose_only"}
    for key, value in stamp.spec.items():
        assert value == spec[key], key
    for key in ARCHITECTURE_SPEC_KEYS:
        assert key not in stamp.formulation.parameters
        assert stamp.spec[key] == stamp.architecture[key]
    assert spec["propose_only"] is True
    assert RESEARCH_EXAM_IS_PROPOSE_ONLY is True
    assert stamp.spec["recipe_name"].startswith("anneal-routed-particle-error")
