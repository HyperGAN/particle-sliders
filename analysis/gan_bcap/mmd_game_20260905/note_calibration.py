"""Write complete measured findings for a single strength-calibration round."""
import argparse
import json
from pathlib import Path
import re
import statistics as stats

from .game import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--next-experiment', required=True)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    folder = home/'rounds'/args.round
    state = json.loads((home/'state.json').read_text())
    card = json.loads((folder/'round.json').read_text())
    candidates = [e for e in state['entries'].values() if e['round'] == args.round and e['family'] == 'calibration']
    if len(candidates) != 1:
        raise ValueError('Expected one declared calibration candidate')
    candidate = candidates[0]
    parent = state['entries'][card['parent']]
    mmd = state['entries'][card['initial_champions']['mmd']]
    training = Path(candidate['weights']).parent
    manifest = json.loads((training/'manifest.json').read_text())
    completion = json.loads((training/'completion.json').read_text())
    if completion['attempted_updates'] != 0 or completion['tensor_audit'] != 'passed':
        raise ValueError('Expected an audited construction without optimizer updates')
    protocol = json.loads((home/'protocol.json').read_text())
    row_by_fixture = {digest([protocol['prompts_sha256'], row, seed, protocol['duration']]): row
        for row in protocol['rows'] for seed in protocol['seeds']}
    reference = {r['fixture']:r for r in parent['records']}
    clips = '\n'.join(f"| {row_by_fixture[r['fixture']]} | {r['seed']} | {r['heuristic_score']:.7f} | {reference[r['fixture']]['heuristic_score']:.7f} | "
        f"{r['candidate']['lyric_diagnostics']['recall']:.4f} | {reference[r['fixture']]['candidate']['lyric_diagnostics']['recall']:.4f} | "
        f"{r['candidate']['production']:.3f} | {reference[r['fixture']]['candidate']['production']:.3f} |" for r in candidate['records'])
    raw = []
    for key in ('concept', 'enjoyment', 'production'):
        raw.append(f"| {key} | {stats.mean(r['candidate'][key] for r in candidate['records']):.6f} | {stats.mean(r['candidate'][key] for r in parent['records']):.6f} |")
    for key in ('phrase_accuracy', 'recall'):
        raw.append(f"| ASR {key} | {stats.mean(r['candidate']['lyric_diagnostics'][key] for r in candidate['records']):.6f} | {stats.mean(r['candidate']['lyric_diagnostics'][key] for r in parent['records']):.6f} |")
    wins = sum(r['heuristic_score'] > reference[r['fixture']]['heuristic_score'] for r in candidate['records'])
    lyric_losses = sum(r['candidate']['lyric_diagnostics']['recall'] < reference[r['fixture']]['candidate']['lyric_diagnostics']['recall'] for r in candidate['records'])
    gain = candidate['score']-card['initial_best_scores']['overall']
    execution = [json.loads(line) for line in (home/'execution.jsonl').read_text().splitlines()]
    times = {Path(r['log']).name:r['seconds'] for r in execution if str(folder) in r['log'] and r.get('returncode') == 0}
    test_count = int(re.search(r'(\d+) passed', (folder/'validation.txt').read_text()).group(1))
    plan = json.loads((folder/'confirmation/plan.json').read_text())
    confirmation_path = folder/'confirmation/summary.json'
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text())
        if confirmation['status'] != 'complete':
            raise ValueError('Required confirmation must be complete')
        names = {e['weights']:e['id'] for e in state['entries'].values()}
        ranking = '\n'.join(f"| {names.get(r['checkpoint'], Path(r['checkpoint']).stem)} | {r['heuristic_score']:.7f} | {r['minimum_clip_score']:.7f} |" for r in confirmation['ranking'])
        extra = [r for r in confirmation['records'] if r['checkpoint']['path'] == candidate['weights']]
        refs = {(r['row'],r['seed']):r for r in confirmation['records'] if r['checkpoint']['path'] == parent['weights']}
        extra_gain = stats.mean(r['heuristic_score']-refs[(r['row'],r['seed'])]['heuristic_score'] for r in extra)
        extra_wins = sum(r['heuristic_score'] > refs[(r['row'],r['seed'])]['heuristic_score'] for r in extra)
        extra_lyric_losses = sum(r['candidate']['lyric_diagnostics']['recall'] < refs[(r['row'],r['seed'])]['candidate']['lyric_diagnostics']['recall'] for r in extra)
        extra_clips = '\n'.join(f"| {r['row']} | {r['seed']} | {r['heuristic_score']:.7f} | {refs[(r['row'],r['seed'])]['heuristic_score']:.7f} | "
            f"{r['candidate']['lyric_diagnostics']['recall']:.4f} | {refs[(r['row'],r['seed'])]['candidate']['lyric_diagnostics']['recall']:.4f} | "
            f"{r['candidate']['production']:.3f} | {refs[(r['row'],r['seed'])]['candidate']['production']:.3f} |" for r in extra)
        substantial = gain >= plan['substantial_development_gain'] and extra_gain > 0
        confirmation_text = f'''The new-leader trigger fired and the predeclared confirmation completed on rows {plan['rows']}, seeds {plan['seeds']}, up to {plan['duration']:g} seconds. These were fresh conditions and seeds for this game search. Results stay outside the development leaderboard.

| Entry | Added-fixture mean | Worst clip |
| --- | ---: | ---: |
{ranking}

The candidate wins {extra_wins}/4 added scores against its parent, with mean gain {extra_gain:+.7f}. Supplied-word recall is lower on {extra_lyric_losses}/4 added clips. The declared numerical substantial-gain screen is {'met' if substantial else 'not met'}; individual lyric and quality regressions still matter. Confirmation took {times['driver.log']:.3f} seconds including rendering, scoring and diversity diagnostics.

| Row | Seed | Candidate score | Parent score | Candidate recall | Parent recall | Candidate production | Parent production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{extra_clips}

[Additional listening comparison](confirmation/index.html) · [All confirmation results and diversity diagnostics](confirmation/summary.json). Within-condition cross-seed chroma, spectral-flux rhythm and content-embedding distances are saved alongside production, lyrics and high-frequency power. These full-mix proxies and two seeds per condition cannot establish mode coverage; larger distances can reflect errors. No human listening verdict is claimed.'''
    else:
        result = json.loads((folder/'confirmation/outcome.json').read_text())
        if result['status'] != 'not_triggered':
            raise ValueError('Required confirmation is unfinished')
        confirmation_text = f'''The new-overall-leader trigger did not fire. Confirmation/outcome.json records this. The predeclared additional comparison was rows {plan['rows']}, seeds {plan['seeds']}, up to {plan['duration']:g} seconds against the parent and the additional strong reference. It was not rendered. This candidate's extra-prompt generalization and musical diversity remain unmeasured.'''
    outcome = 'set a new development lead' if candidate['eligible'] and gain > 1e-9 else 'did not beat the starting overall leader'
    text = f'''# {args.round}: strength {manifest['factor']:g} {outcome}

The candidate scored **{candidate['score']:.7f}**, {gain:+.7f} versus the starting leader {parent['score']:.7f}. MMD/adaptive remains a separate method record at {mmd['score']:.7f}. This candidate is explicitly family `calibration`. It wins {wins}/4 development scores against its parent; worst score {candidate['worst_score']:.7f}; eligible {candidate['eligible']}.

[Listening comparison](index.html) · [All development metrics](comparison.json) · [Exact derivation]({training.name}/manifest.json) · [Saved-state and source audit](state-and-source-audit.json).

## Hypothesis, construction and compute

{card['hypothesis']} This is one preset candidate at factor {manifest['factor']:g}, declared before rendering. It is not a sweep or an additional learned update. The preceding float32 and coverage MMD candidates improved fixed teacher-fit losses but scored worse on audio. The strongest measured baseline GAN already exceeded the positive-caption concept proxy on all four development clips, motivating a modest gain test before further teacher distillation.

The parent is `{parent['id']}`. All {manifest['alpha_buffer_count']} alpha buffers and sidecar alpha changed from {manifest['alpha_before']:g} to {manifest['alpha_after']:g}, while every trainable matrix stayed bitwise identical. The renderer still uses the locked scale +1 and unit_scale1. No optimizer was instantiated or updated; the complete parent state, its critic and moments are archived in parent-state.pt. The derived state stores the new buffers and exact recipe under schema derived-strength-1. It is reconstructable calibration state, not a claim of resumed GAN training.

Actual construction time was {completion['total_seconds']:.3f} seconds, with zero training updates and zero training time. Audio rendering/scoring took {times['evaluation.log']:.3f} seconds. All model work ran serially on physical GPU 1 in the existing environment. {test_count} launch checks passed, covering the game keeper, explicit family outcomes, unchanged MMD champion after a non-MMD win, actual native-LoRA gain parity in bf16/float32, and confirmation audio identity. Construction also reloaded the real safetensors and verified all trained tensors. The parent weights/state, prompts, source, factor, metadata and final checkpoint hashes are retained. No requirements, production package or catalog changed.

The only game-keeper changes add the user-authorized method labels and permit declared zero-update construction rounds. Both round outcome and MMD championship keep method identities explicit. The frozen score, four evaluation sources, fixtures, normalization, component weights, strength setting and eligibility checks are unchanged; hashes are archived under bookkeeping-provenance.

## Individual audio tradeoffs

| Row | Seed | Candidate score | Parent score | Candidate recall | Parent recall | Candidate production | Parent production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{clips}

| Raw mean | Candidate | Parent |
| --- | ---: | ---: |
{chr(10).join(raw)}

Supplied-word recall is lower on {lyric_losses}/4 development clips. The mean excess-high-frequency penalty component is {candidate['components']['artifacts']:.7f}; that metric alone does not establish artifact-free audio. Each exact off and positive-caption control is shared byte for byte. These four repeated development fixtures and the heuristic score are not a human preference verdict.

## Separate confirmation and next experiment

{confirmation_text}

**Next experiment:** {args.next_experiment}
'''
    (folder/'notes.md').write_text(text)
    (folder/'next-experiment.txt').write_text(args.next_experiment+'\n')
    print(folder/'notes.md')


if __name__ == '__main__':
    main()
