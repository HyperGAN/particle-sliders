"""Build the repeatable development-game specification from existing evidence.

This creates fixtures and reference scorecards, not an executable GPU evaluator.
Existing outputs must match exactly; changing the game requires a new version.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis/reward_search_20260908'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_digest(value):
    # Match reward_sliders.specs.digest, which includes JSON separator spaces.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_immutable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Existing game artifact changed: {path}; create a new game version')
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def score(cases, label):
    by_family = {}
    scores, deltas, comparisons = [], [], []
    for case in cases:
        controls = case['controls']
        value = controls[label]['ce']
        delta = value - controls['off']['ce']
        by_family.setdefault(case['family']['family'], []).append(delta)
        scores.append(value)
        deltas.append(delta)
        comparisons.append(value - controls['v1-original']['ce'])
    family_means = {k: sum(v) / len(v) for k, v in by_family.items()}
    return dict(cases=len(cases), wins_vs_off=sum(x > 0 for x in deltas),
                meaningful_wins_vs_off=sum(x >= .02 for x in deltas),
                mean_ce=sum(scores)/len(scores),
                equal_family_ce_gain=sum(family_means.values())/len(family_means),
                worst_ce_delta=min(deltas),
                wins_vs_original=sum(x > 0 for x in comparisons),
                mean_gain_vs_original=sum(comparisons)/len(comparisons),
                family_deltas=family_means)


def main():
    manifest = read(SOURCE / 'manifest.json')
    families = {f['family']: f for f in manifest['families'] if f['split'] == 'dev'}
    labels = {'off': 'off', 'v1-original': 'v1-m1', 'v1-half': 'v1-m0.5'}
    cases = {}
    for family_id, family in sorted(families.items()):
        for seed in (5519, 6637):
            case_id = f'{family_id}-s{seed}'
            controls = {}
            for label, arm in labels.items():
                path = SOURCE / 'stages/v1-strength/observations' / f'{case_id}-{arm}.json'
                row = read(path)
                if row['status'] != 'complete' or not row['reward']['valid']:
                    raise ValueError('Invalid reference observation: ' + str(path))
                if sha(row['audio']) != row['audio_sha256']:
                    raise ValueError('Reference WAV changed: ' + row['audio'])
                if row['family'] != family_id or row['seed'] != seed:
                    raise ValueError('Reference identity mismatch')
                if row['provenance']['family_sha256'] != source_digest(family):
                    raise ValueError('Reference prompt or family metadata changed')
                if row['cell_hash'] != source_digest({k: family[k] for k in ('caption', 'lyrics', 'style_multipliers')}):
                    raise ValueError('Reference comparison cell changed')
                windows = row['reward']['windows']
                if [(w['start_s'], w['end_s']) for w in windows] != [(0., 10.), (10., 20.)]:
                    raise ValueError('Reference uses different score windows')
                if abs(sum(w['axes']['CE'] for w in windows) / 2 - row['reward']['scalar']) > 1e-9:
                    raise ValueError('Reference score aggregation changed')
                controls[label] = dict(observation=str(path), observation_sha256=sha(path),
                    audio=row['audio'], audio_sha256=row['audio_sha256'],
                    ce=row['reward']['scalar'], arm=row['arm'],
                    provenance=row['provenance'], reward_spec_sha256=row['reward']['reward_spec_sha256'])
            gpus = {value['provenance']['physical_gpu'] for value in controls.values()}
            if len(gpus) != 1:
                raise ValueError('Controls were not rendered on one matched physical GPU')
            cases[case_id] = dict(id=case_id, family=family, seed=seed,
                physical_gpu=gpus.pop(), style_components=manifest['style_components'][family_id], controls=controls)
    # Development cases are already exposed. This order is now fixed, including
    # balanced voices, both seeds, mixed styles, and an early instrumental dance case.
    first = ['search-dev-16-s5519', 'search-dev-17-s6637',
             'search-dev-22-s5519', 'search-dev-19-s6637']
    first_eight = first + ['search-dev-18-s5519', 'search-dev-20-s5519',
                          'search-dev-21-s6637', 'search-dev-23-s6637']
    order = first_eight + [key for key in sorted(cases) if key not in first_eight]
    assert len(order) == len(set(order)) == 16
    assert len({cases[key]['family']['voice'] for key in first}) == 4
    assert len({cases[key]['family']['family'] for key in first_eight}) == 8
    game = dict(schema='music-reward-slider-game-spec-v1', game_id='reward-slider-game-v1',
        created_date='2026-09-08', purpose='repeatable adaptive development benchmark; not untouched validation',
        status='fixture specification ready; evaluator/controller CLI must be implemented from the handoff',
        source_manifest=str(SOURCE/'manifest.json'), source_manifest_sha256=sha(SOURCE/'manifest.json'),
        historical_12_of_16_is_a_different_cohort=True,
        reward_spec=manifest['reward_spec'], sampler=manifest['sampler'],
        duration_seconds=20.4, scored_seconds=20., host_energy_max=manifest['host_energy_max'],
        controls=list(labels), original_reference=manifest['search']['reference'],
        default='off', cases=[cases[key] for key in order],
        stages={str(n):order[:n] for n in (4,8,16)},
        rules=dict(win='candidate CE > matched Off CE; exact ties are not wins',
            meaningful_win='candidate CE - Off CE >= .02',
            family_weight='equal independent family weight; two windows are not separate examples',
            stage4=dict(min_wins=3,min_mean_ce_gain=.02,min_worst_ce_delta=-.50),
            stage8=dict(min_wins=6,min_mean_ce_gain=.05,min_worst_ce_delta=-.50),
            stage16=dict(min_wins=13,min_mean_ce_gain=.10,min_worst_ce_delta=-.50,
                         min_mean_gain_vs_original=.05,min_wins_vs_original=10),
            validity='every declared comparison valid; model failures are non-wins and fail advancement',
            leaderboard='separate stages; full valid 16-case cards rank by gate pass, wins vs Off, worst delta, mean gain vs Off, mean gain vs original',
            weak_candidate='record rejection and try a different candidate/method; do not stop the whole search',
            thresholds='operational defaults for this new game; never revise in place after seeing candidate scores'),
        heldout=dict(included=False,minimum_families=8,seeds_per_family=2,
            controls=['off','v1-original'],normal_new_audio_ceiling=48,
            use='only after a 16-case development pass; fresh families and seeds, frozen before rendering',
            repeated_attempts='track all confirmation attempts and predeclare multiplicity handling or independent replication'))
    references = dict(game_id=game['game_id'], benchmark_sha256=digest(game),
        comparisons_are_development=True,
        stages={str(n):{label:score([cases[key] for key in order[:n]],label) for label in labels} for n in (4,8,16)})
    write_immutable(HERE/'benchmark.json',game)
    write_immutable(HERE/'reference-scorecards.json',references)
    write_immutable(HERE/'audit.json',dict(passed=True,cases=16,families=8,reference_observations=48,
        reference_audio_hashes_verified=True,matched_physical_gpu=True,
        new_audio_generated=0,benchmark_sha256=sha(HERE/'benchmark.json'),
        reference_scorecards_sha256=sha(HERE/'reference-scorecards.json')))
    print('Verified 16 development cases and 48 cached reference WAVs. No new audio generated.')


if __name__ == '__main__':
    main()
