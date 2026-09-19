#!/usr/bin/env python3
"""Compile comparable Music 3 / YuE2 architecture winners.

Training geometry is a screen, never an audio-quality proxy.  A YuE2 winner
must also preserve held-out enjoyment and production within the release
tolerance on matched Off/On renders.  Missing audio metrics fail closed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPENDENCIES = ROOT.parent / '.cache/slider-quality/python'
if DEPENDENCIES.exists():
    sys.path.insert(0, str(DEPENDENCIES))

from slider_selection.features import AudioMeasurer, write_json

QUALITY_TOLERANCE = 0.2
WORST_WINDOW_TOLERANCE = 0.5


def training_rows(run: Path) -> list[dict]:
    rows = {}
    for path in sorted(run.glob('updates-from-*.jsonl'),
                       key=lambda p: int(p.stem.rsplit('-', 1)[1])):
        start = int(path.stem.rsplit('-', 1)[1])
        rows = {step: row for step, row in rows.items() if step <= start}
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                row = json.loads(line)
                rows[int(row['step'])] = row
    return [rows[key] for key in sorted(rows)]


def training_score(rows: list[dict], late_fraction: float = .3) -> dict:
    if len(rows) < 10:
        return dict(training_ready=False, training_reason='too_few_updates', updates=len(rows))
    late = rows[int(len(rows) * (1 - late_fraction)):]
    cos = [float(row['cos_pos']) for row in rows]
    late_cos = [float(row['cos_pos']) for row in late]
    gradients = [float(row['grad_norm']) for row in late]
    adversarial = [float(row['g_adv']) for row in late]
    values = cos + gradients + adversarial
    if not all(math.isfinite(value) for value in values):
        return dict(training_ready=False, training_reason='nonfinite', updates=len(rows))
    lock = stats.median(late_cos)
    floor = min(late_cos)
    volatility = stats.pstdev(late_cos) if len(late_cos) > 1 else 0.
    recovery = max(0., lock - min(cos))
    fitness = 2 * lock + .5 * floor + .75 * recovery - volatility
    return dict(training_ready=True, updates=len(rows), training_fitness=fitness,
                late_cos_median=lock, late_cos_min=floor,
                late_cos_std=volatility, recovery=recovery,
                gradient_median=stats.median(gradients),
                adversarial_median=stats.median(adversarial))


def _quality(audio: dict) -> dict:
    windows = audio['windows']
    return dict(enjoyment=stats.mean(float(w['aesthetics']['CE']) for w in windows),
                production=stats.mean(float(w['aesthetics']['PQ']) for w in windows),
                enjoyment_worst=min(float(w['aesthetics']['CE']) for w in windows),
                production_worst=min(float(w['aesthetics']['PQ']) for w in windows),
                concept=stats.mean(float(w['concept']['distorted']) for w in windows))


def audio_score(output: Path, measurer: AudioMeasurer) -> dict:
    pairs = []
    for off_path in sorted(output.glob('row-*-seed-*/off/audio.flac')):
        on_path = off_path.parents[1] / 'metal' / 'audio.flac'
        if not on_path.exists():
            continue
        off, on = _quality(measurer.measure(off_path)), _quality(measurer.measure(on_path))
        pairs.append({key: on[key] - off[key] for key in off})
    if not pairs:
        return dict(audio_ready=False, audio_reason='no_matched_off_on_pairs', pairs=0)
    aggregate = {key + '_delta': stats.mean(pair[key] for pair in pairs) for key in pairs[0]}
    aggregate.update({key + '_delta_min': min(pair[key] for pair in pairs) for key in pairs[0]})
    quality_pass = (
        aggregate['enjoyment_delta'] >= -QUALITY_TOLERANCE
        and aggregate['production_delta'] >= -QUALITY_TOLERANCE
        and aggregate['enjoyment_worst_delta_min'] >= -WORST_WINDOW_TOLERANCE
        and aggregate['production_worst_delta_min'] >= -WORST_WINDOW_TOLERANCE
    )
    return dict(audio_ready=True, pairs=len(pairs), quality_pass=quality_pass, **aggregate)


def rank(rows: list[dict]) -> list[dict]:
    """Fail closed, then prefer enjoyment, production, concept, train lock."""
    for row in rows:
        row['eligible'] = bool(row.get('training_ready') and row.get('audio_ready')
                               and row.get('quality_pass'))
    return sorted(rows, key=lambda row: (
        row['eligible'], row.get('enjoyment_delta', -math.inf),
        row.get('production_delta', -math.inf), row.get('concept_delta', -math.inf),
        row.get('training_fitness', -math.inf)), reverse=True)


def compile_board(search_root: Path, music3_root: Path | None,
                  cache: Path, device: str, measure_audio: bool) -> dict:
    rows = []
    measurer = AudioMeasurer(cache, device) if measure_audio else None
    for run in sorted((search_root / 'runs').glob('*')):
        if not run.is_dir():
            continue
        manifest = json.loads((run / 'manifest.json').read_text()) if (run / 'manifest.json').exists() else {}
        recipe = manifest.get('settings', {}).get('recipe', {})
        row = dict(model='yue2', candidate=run.name, critic=recipe.get('critic'),
                   critic_config=recipe.get('critic_config', {}), run=str(run))
        row.update(training_score(training_rows(run)))
        output = search_root / 'listens' / run.name
        if measurer is not None:
            row.update(audio_score(output, measurer))
        else:
            row.update(audio_ready=False, audio_reason='measurement_not_requested')
        rows.append(row)
    if music3_root and (music3_root / 'runs').exists():
        for run in sorted((music3_root / 'runs').glob('*')):
            if run.is_dir():
                row = dict(model='music3', candidate=run.name, critic=run.name.split('_', 1)[0],
                           critic_config={}, run=str(run), audio_ready=False,
                           audio_reason='cross_model_training_reference_only')
                row.update(training_score(training_rows(run)))
                rows.append(row)
    ranked = rank(rows)
    for index, row in enumerate(ranked, 1):
        row['rank'] = index
        row['winner'] = index == 1 and row['eligible']
    return dict(schema=1, policy=dict(quality_tolerance=QUALITY_TOLERANCE,
        worst_window_tolerance=WORST_WINDOW_TOLERANCE,
        order=['eligible','enjoyment_delta','production_delta','concept_delta','training_fitness'],
        note='Training geometry screens candidates; only measured held-out YuE2 audio can win.'),
        winner=next((row['candidate'] for row in ranked if row['winner']), None), rows=ranked)


def markdown(board: dict) -> str:
    lines = ['# Music architecture scoreboard', '',
        'A winner must preserve matched held-out enjoyment and production. Missing audio fails closed.', '',
        '| rank | model | candidate | critic | eligible | enjoyment Δ | production Δ | concept Δ | train |',
        '|---:|---|---|---|:---:|---:|---:|---:|---:|']
    def value(row, key):
        return '—' if row.get(key) is None else f"{row[key]:.3f}"
    for row in board['rows']:
        lines.append(f"| {row['rank']} | {row['model']} | {row['candidate']} | {row.get('critic') or '—'} | "
            f"{'yes' if row['eligible'] else 'no'} | {value(row,'enjoyment_delta')} | "
            f"{value(row,'production_delta')} | {value(row,'concept_delta')} | "
            f"{value(row,'training_fitness')} |")
    lines += ['', f"Winner: **{board['winner']}**" if board['winner'] else 'Winner: **pending audio-qualified candidate**', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search_root', type=Path, required=True)
    parser.add_argument('--music3_root', type=Path)
    parser.add_argument('--cache', type=Path, default=ROOT / 'eval/architecture-scoreboard-cache')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--measure_audio', action='store_true')
    args = parser.parse_args()
    board = compile_board(args.search_root, args.music3_root, args.cache,
                          args.device, args.measure_audio)
    write_json(args.search_root / 'scoreboard.json', board)
    (args.search_root / 'scoreboard.md').write_text(markdown(board))
    print(json.dumps(dict(winner=board['winner'], candidates=len(board['rows']))))


if __name__ == '__main__':
    main()
