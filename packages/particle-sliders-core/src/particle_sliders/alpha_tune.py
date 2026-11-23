"""Tune an exported RAW slider with Jev and save replayable inference settings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
from importlib.metadata import entry_points
from math import fsum, isfinite
from pathlib import Path
import re

from .alpha_search import search_alpha
from .alpha_tune_jev import JEV_MODEL, JevJudge
from .candidate_failure import FileRecord
from .execution.local import atomic_json
from .execution.request import read_json
from .strength_schedule import interval_strengths

SCHEMA = "particle-slider-alpha-tune-v1"


def _record(body, directory):
    record = FileRecord(str((directory / body["path"]).resolve()), body["bytes"], body["sha256"])
    record.verify()
    return record


def _load_definition(path):
    source = FileRecord.from_path(path)
    body = source.read_json()
    if not isinstance(body, dict) or body.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} slider definition")
    if body.get("selection") != "raw" or body.get("ema_decay", False) is not None:
        raise ValueError("alpha tuning requires selection=raw and ema_decay=null")
    if body.get("judge", "jev") != "jev":
        raise ValueError("alpha tuning uses Jev")
    for key in ("renderer", "concept"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            raise ValueError(f"missing {key}")

    checkpoint = _record(body["checkpoint"], path.parent)
    grids = body["profiles"]
    if not isinstance(grids, dict) or not grids or set(grids) - {"constant", "early", "late"}:
        raise ValueError("define native alpha grids by strength profile")
    for alphas in grids.values():
        if (not isinstance(alphas, list) or not alphas
                or any(type(alpha) not in (int, float) or not isfinite(alpha) for alpha in alphas)):
            raise ValueError("each profile needs a finite alpha grid")
    rounds = body.get("refinement_rounds", 1)
    if type(rounds) is not int or rounds < 0:
        raise ValueError("refinement_rounds must be a nonnegative integer")

    cases = {}
    for row in body["cases"]:
        key = row["id"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", key) or key in cases:
            raise ValueError("case IDs must be unique letters, digits, underscores, or hyphens")
        if row["split"] not in ("search", "reserved") or not isinstance(row["request"], dict):
            raise ValueError("case requires a native request and a search or reserved split")
        if not isinstance(row["preserve"], str) or not row["preserve"].strip():
            raise ValueError("case requires a description of the original content to preserve")
        cases[key] = row | {"off": _record(row["off"], path.parent)}
    if not any(row["split"] == "search" for row in cases.values()):
        raise ValueError("include at least one search case")
    fixtures = []
    for fixture in body.get("judge_fixtures", []):
        verdict = fixture["expected"]
        if verdict not in ("pass", "fail"):
            raise ValueError("judge fixture must expect pass or fail")
        fixtures.append((fixture["id"], _record(fixture["off"], path.parent),
                         _record(fixture["on"], path.parent), fixture["preserve"], verdict))
    return source, body, checkpoint, cases, fixtures


def _renderer(name):
    bindings = list(entry_points(group="particle_sliders.alpha_renderers", name=name))
    if len(bindings) != 1:
        raise ValueError(f"expected one installed alpha renderer {name!r}; found {len(bindings)}")
    return bindings[0].load()


def _render(renderer, row, gains, target):
    receipt = target.with_name("render-pin.json")
    if receipt.is_file():
        record = FileRecord(**read_json(receipt))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        record = renderer.render(row["request"], gains, target)
        if not isinstance(record, FileRecord):
            raise ValueError("native renderer must return a FileRecord")
    if Path(record.path).resolve() != target:
        raise ValueError("native renderer returned a different output path")
    record.verify()
    if not receipt.is_file():
        atomic_json(receipt, asdict(record))
    return record


def _evaluate(renderer, judge, root, cases, ids, alpha, profile, *, phase):
    gains = interval_strengths(alpha, renderer.interval_widths, profile=profile)
    folder = root / "renders" / profile / alpha.hex()
    evidence = {}
    observations = {}
    with ThreadPoolExecutor(3) as pool:
        pending = []
        for key in ids:
            row = cases[key]
            output = _render(renderer, row, gains, folder / key / "image.png")
            answer = pool.submit(judge.observe, row["off"], output, preserve=row["preserve"])
            pending.append((key, row["off"], output, answer))
        for key, off, output, answer in pending:
            assessment = answer.result()
            observations[key] = assessment.observation
            evidence[key] = dict(off=asdict(off), on=asdict(output), checks=assessment.checks,
                                 observation=asdict(assessment.observation))
    atomic_json(folder / f"{phase}.json", {
        "profile": profile, "alpha": alpha, "effective_strengths": gains, "cases": evidence,
    })
    return observations


def tune(path):
    path = Path(path).resolve()
    source, body, checkpoint, cases, fixtures = _load_definition(path)
    root = path.with_name(path.stem + ".alpha-tune")
    root.mkdir(exist_ok=True)
    with (root / "lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        request = root / "request.json"
        receipt = root / "tuning.json"
        if request.exists():
            if read_json(request) != asdict(source):
                raise ValueError(f"definition changed since this run started: {request}")
        else:
            atomic_json(request, asdict(source))
        if receipt.exists():
            saved = read_json(receipt)
            if saved.get("definition") != asdict(source) or saved.get("checkpoint") != asdict(checkpoint):
                raise ValueError("saved tuning belongs to a different definition or checkpoint")
            return receipt

        judge = JevJudge(root / "jev", body["concept"])
        checks = []
        for key, off, on, preserve, expected in fixtures:
            assessment = judge.observe(off, on, preserve=preserve)
            checks.append(dict(id=key, expected=expected, actual=assessment.observation.integrity,
                               checks=assessment.checks))
        if checks:
            atomic_json(root / "qualification.json", checks)
            if any(item["expected"] != item["actual"] for item in checks):
                raise ValueError("Jev did not qualify on the definition's edit fixtures")

        searches = {}
        candidates = []
        validation = []
        selected = []
        screen = [key for key, row in cases.items() if row["split"] == "search"]
        reserved = [key for key, row in cases.items() if row["split"] == "reserved"]
        open_renderer = _renderer(body["renderer"])
        with open_renderer(checkpoint, body.get("renderer_options", {})) as renderer:
            sampler = renderer.sampler
            sampler_path = root / "sampler.json"
            if sampler_path.exists() and read_json(sampler_path) != sampler:
                raise ValueError("the native sampler or renderer identity changed")
            atomic_json(sampler_path, sampler)
            for profile, grid in body["profiles"].items():
                def evaluate(alpha):
                    return _evaluate(renderer, judge, root, cases, screen, alpha, profile, phase="search")

                result = search_alpha(checkpoint=checkpoint, case_ids=screen, alphas=grid, evaluate=evaluate,
                                      refinement_rounds=body.get("refinement_rounds", 1))
                searches[profile] = asdict(result)
                candidates.extend((trial.mean_gain, profile, trial.alpha)
                                  for trial in result.trials if trial.status == "pass")
                atomic_json(root / "searches.json", searches)

            for mean_gain, profile, alpha in sorted(candidates, key=lambda item: -item[0]):
                if selected and mean_gain < selected[0]["search_gain"]:
                    break
                assessment = _evaluate(renderer, judge, root, cases, reserved, alpha, profile,
                                       phase="reserved") if reserved else {}
                statuses = [item.integrity for item in assessment.values()]
                status = "fail" if "fail" in statuses else "unmeasured" if "unmeasured" in statuses else "pass"
                validation.append(dict(profile=profile, alpha=alpha, status=status))
                atomic_json(root / "validation.json", validation)
                if status == "pass":
                    resolved = interval_strengths(alpha, renderer.interval_widths, profile=profile)
                    reserved_gain = fsum(item.on - item.off for item in assessment.values()) / len(reserved) if reserved else None
                    selected.append(dict(alpha=alpha, profile=profile, effective_strengths=resolved,
                                         step_gains=[gain / alpha if alpha else 1. for gain in resolved],
                                         search_gain=mean_gain, reserved_gain=reserved_gain))
        source.verify()
        checkpoint.verify()
        if not selected:
            raise ValueError("no measured alpha passed Jev on all declared cases")
        atomic_json(receipt, dict(schema="particle-slider-alpha-tuning-v1", judge="jev", judge_model=JEV_MODEL,
                    definition=asdict(source), checkpoint=asdict(checkpoint), sampler=sampler, selected=selected))
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alpha-tune", description=__doc__)
    parser.add_argument("slider_definition", type=Path)
    args = parser.parse_args(argv)
    try:
        print(tune(args.slider_definition), flush=True)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"alpha-tune: {error}\n")


if __name__ == "__main__":
    main()
