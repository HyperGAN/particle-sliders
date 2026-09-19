"""Report the completed frozen active-history challenge from retained evidence."""
import json
from pathlib import Path
import statistics as stats

from .game import digest


def main():
    home = Path(__file__).resolve().parent
    folder = home/'rounds/round-0003'
    training = folder/'coverage-candidate'
    state = json.loads((home/'state.json').read_text())
    card = json.loads((folder/'round.json').read_text())
    candidate = state['entries']['round0003_coverage']
    parent = state['entries'][card['parent']]
    leader = state['entries'][card['initial_champions']['overall']]
    manifest = json.loads((training/'manifest.json').read_text())
    completion = json.loads((training/'completion.json').read_text())
    audit = json.loads((folder/'state-and-precision-audit.json').read_text())
    initial = json.loads((training/f'evaluation-{manifest["source_step"]}.json').read_text())
    final = json.loads((training/f'evaluation-{completion["step"]}.json').read_text())
    records = [json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()]
    execution = [json.loads(line) for line in (home/'execution.jsonl').read_text().splitlines()]
    times = {Path(row['log']).name: row['seconds'] for row in execution
             if str(folder) in row['log'] and row.get('returncode') == 0}
    protocol = json.loads((home/'protocol.json').read_text())
    fixture_rows = {digest([protocol['prompts_sha256'], row, seed, protocol['duration']]): row
                    for row in protocol['rows'] for seed in protocol['seeds']}
    parent_records = {row['fixture']: row for row in parent['records']}
    leader_records = {row['fixture']: row for row in leader['records']}
    clips = '\n'.join(f"| {fixture_rows[r['fixture']]} | {r['seed']} | {r['heuristic_score']:.7f} | "
        f"{parent_records[r['fixture']]['heuristic_score']:.7f} | {leader_records[r['fixture']]['heuristic_score']:.7f} | "
        f"{r['candidate']['lyric_diagnostics']['recall']:.4f} | {parent_records[r['fixture']]['candidate']['lyric_diagnostics']['recall']:.4f} |"
        for r in candidate['records'])
    means = []
    for key in ('concept', 'enjoyment', 'production'):
        means.append(f"| {key} | {stats.mean(r['candidate'][key] for r in candidate['records']):.6f} | "
                     f"{stats.mean(r['candidate'][key] for r in parent['records']):.6f} | {stats.mean(r['candidate'][key] for r in leader['records']):.6f} |")
    for key in ('phrase_accuracy', 'recall'):
        means.append(f"| ASR {key} | {stats.mean(r['candidate']['lyric_diagnostics'][key] for r in candidate['records']):.6f} | "
                     f"{stats.mean(r['candidate']['lyric_diagnostics'][key] for r in parent['records']):.6f} | {stats.mean(r['candidate']['lyric_diagnostics'][key] for r in leader['records']):.6f} |")
    group_rows = []
    for kind in initial['train_groups']:
        a, b = initial['train_groups'][kind], final['train_groups'][kind]
        group_rows.append(f'| {kind} | {a:.7f} | {b:.7f} | {(1-b/a)*100:.2f}% |')
    a, b = stats.mean(r['loss'] for r in initial['heldout']), stats.mean(r['loss'] for r in final['heldout'])
    group_rows.append(f'| Reserved prompt histories | {a:.7f} | {b:.7f} | {(1-b/a)*100:.2f}% |')
    branch_rows = []
    for name, ix in (('Original', slice(0, 4)), ('Active parent', slice(4, 8))):
        for branch, label in enumerate(('conditional', 'unconditional')):
            start = stats.mean(r['branch_continuation_rmse'][branch] for r in initial['train'][ix])
            end = stats.mean(r['branch_continuation_rmse'][branch] for r in final['train'][ix])
            branch_rows.append(f'| {name}, {label} | {start:.7f} | {end:.7f} |')
    wins = sum(r['heuristic_score'] > parent_records[r['fixture']]['heuristic_score'] for r in candidate['records'])
    delta_mmd, delta_leader = candidate['score']-parent['score'], candidate['score']-leader['score']
    outcome = ('won the overall development lead' if delta_leader > 1e-9 else
               'improved the MMD personal best' if delta_mmd > 1e-9 else 'lost the audio challenge')
    confirmation_path = folder/'confirmation/summary.json'
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text())
        if confirmation['status'] != 'complete':
            raise ValueError('Do not report an unfinished required confirmation as complete')
        names = {e['weights']: e['id'] for e in state['entries'].values()}
        ranking = '\n'.join(f"| {names.get(r['checkpoint'], Path(r['checkpoint']).stem)} | {r['heuristic_score']:.7f} | {r['minimum_clip_score']:.7f} |" for r in confirmation['ranking'])
        extra = [r for r in confirmation['records'] if r['checkpoint']['path'] == candidate['weights']]
        reference = {(r['row'], r['seed']): r for r in confirmation['records'] if r['checkpoint']['path'] == leader['weights']}
        extra_gain = stats.mean(r['heuristic_score']-reference[(r['row'], r['seed'])]['heuristic_score'] for r in extra)
        extra_wins = sum(r['heuristic_score'] > reference[(r['row'], r['seed'])]['heuristic_score'] for r in extra)
        lyric_losses = sum(r['candidate']['lyric_diagnostics']['recall'] < reference[(r['row'], r['seed'])]['candidate']['lyric_diagnostics']['recall'] for r in extra)
        extra_clips = '\n'.join(f"| {r['row']} | {r['seed']} | {r['heuristic_score']:.7f} | {reference[(r['row'], r['seed'])]['heuristic_score']:.7f} | "
            f"{r['candidate']['lyric_diagnostics']['recall']:.4f} | {reference[(r['row'], r['seed'])]['candidate']['lyric_diagnostics']['recall']:.4f} | "
            f"{r['candidate']['production']:.3f} | {reference[(r['row'], r['seed'])]['candidate']['production']:.3f} |" for r in extra)
        substantial = delta_leader >= .15 and extra_gain > 0
        confirmation_text = f'''The new-record trigger fired. The predeclared confirmation is complete: rows 2/3, seeds 101/303, up to 30 seconds, against both baseline 660 and FM-cap 1950. These results remain outside the development leaderboard. Matching reference/control WAVs were reused only after checking the completed source experiment, full render specification, frozen sources, checkpoint identity and every copied audio hash.

| Entry | Additional-fixture mean | Worst clip |
| --- | ---: | ---: |
{ranking}

Against the starting leader, the candidate wins {extra_wins}/4 added scores with mean gain {extra_gain:+.7f}. Supplied-word recall is lower on {lyric_losses}/4 clips. The predeclared numerical substantial-gain screen (development gain at least0.15 plus positive added-fixture mean gain) is {'met' if substantial else 'not met'}; that screen does not override individual lyric/quality regressions or substitute for listening.

| Row | Seed | Candidate score | Reference score | Candidate recall | Reference recall | Candidate production | Reference production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{extra_clips}

[Confirmation listening page](confirmation/index.html) · [Per-clip results and diversity diagnostics](confirmation/summary.json). The diagnostics include lyric errors, production, high-frequency power and within-condition cross-seed chroma, rhythm and content-embedding distances. Two seeds per condition and full-mix proxies do not establish mode coverage or a human listening preference.'''
    else:
        trigger = json.loads((folder/'confirmation/outcome.json').read_text())
        if trigger['status'] != 'not_triggered':
            raise ValueError('Required confirmation is not finished')
        confirmation_text = '''The new-record trigger did not fire. The predeclared additional test was rows 2/3, seeds 101/303, up to 30 seconds, against baseline 660 and FM-cap 1950. `confirmation/outcome.json` records why it was not run. This candidate's extra-prompt generalization and cross-seed musical diversity remain unmeasured. Different files, a falling hidden loss and successfully collecting nonempty histories establish neither.'''
    next_test = ('Test one predeclared 1.25 strength calibration of the strongest measured checkpoint, with zero optimizer updates, '
        'separate calibration-family bookkeeping, unchanged scale +1 in the frozen renderer, and exact alpha-buffer/sidecar provenance. '
        'Use fresh confirmation rows 4/5 and seeds 515/727 at 30 seconds for any new overall record. '
        'Baseline 660 already exceeds its positive-caption reference on all four development concept scores, so teacher matching is not assumed to be the best next route. '
        'Keep the readout-aware distillation prototype as a later experiment; it has CPU checks but no GPU result.')
    findings = f'''# Frozen-history coverage: measured training findings

Fixed loss **{audit['loss_before']:.9f} → {audit['loss_after']:.9f}**, a {100*audit['objective_decrease_fraction']:.2f}% drop. Accepted {completion['accepted']}/{completion['attempted_updates']} attempted updates, using {completion['objective_evaluations']} trial objective evaluations. Net parameter movement from the parent: L2 {audit['final_parameter_l2_from_parent']:.9f}.

| Data group | Initial loss | Final loss | Decrease |
| --- | ---: | ---: | ---: |
{chr(10).join(group_rows)}

| Continuation branch | Initial RMSE | Final RMSE |
| --- | ---: | ---: |
{chr(10).join(branch_rows)}

All four added parent histories contained 250 frames. They were sampled at +1 with seeds 17/18/19/20 before optimization, using the existing bf16 CFG/top-k/RVQ feedback sampler. Parent parameters stayed bitwise unchanged during collection. Original neutral-history inputs and float32 teachers, including the diagnostic rows, remain bitwise identical to round 2's archived tensors. Calibration is now {manifest['scale']:.12f}, computed over all eight histories; raw losses cannot be ranked across changed calibrations. Within this run, the targets, inputs, calibration and objective remain fixed.

The memory-efficient finite-difference probe frees each completed graph before constructing the next; it preserves the objective and was checked alongside the earlier implementation. Float32 repeated forwards and backward-enabled loss agree. BF16 direction failures remain recorded in the precision files. [Exact state, source and history audit](state-and-precision-audit.json).
'''
    notes = f'''# Round 0003: active-history coverage {outcome}

The candidate scored **{candidate['score']:.7f}**, {delta_mmd:+.7f} versus retained MMD/adaptive and {delta_leader:+.7f} versus the round's starting overall leader. It beat its MMD parent on {wins}/4 development clips. Its worst clip scored {candidate['worst_score']:.7f}; eligibility is {candidate['eligible']}. A reduction in training loss alone is insufficient to call this a better slider.

[Listening comparison](index.html) · [All components and transcripts](comparison.json) · [Training findings](training-findings.md) · [Source/state audit](state-and-precision-audit.json).

## Hypothesis and recipe

The test asked whether training only on neutral histories undercovers histories encountered with the slider active. Starting from retained MMD720, it added one fixed parent-generated +1 history per training prompt and trained on the equal mixture of four original and four added histories. Prompts, both CFG branches, rank/alpha 8, all 4096 hidden coordinates, five RBF bandwidths, LR 0.0005, AdamW betas(0,0.999), epsilon 1e-8, no weight decay and the 150-attempt budget match the round 2 recipe. Base bf16 values were widened to float32 with TF32 disabled. Teachers and calibration were recomputed and frozen for the expanded data. Fresh Adam moments are explicit; the source optimizer and RNG are preserved in the initial archive. Starting checkpoint/state outputs, zero-scale identity, unchanged parent sampling weights and original-data parity all passed.

This is paired hidden MMD on a frozen set of histories. It introduces neither a learned discriminator nor independent conditional particles and a fake-fake distributional term. The sampling measure is fixed before optimization; the result does not establish a full trajectory objective or an anti-collapse guarantee.

## Actual compute and preservation

The trainer accepted {completion['accepted']}/{completion['attempted_updates']} updates, with {completion['objective_evaluations']} trial evaluations, {sum(not r['line_search']['accepted'] for r in records)} rejected whole updates and {completion['gradient_fallback_acceptances']} gradient fallback acceptances. The stop status is `{completion['status']}`. Update-loop time was {completion['training_seconds']:.3f} seconds; enclosing trainer time {times['training.log']:.3f} seconds; audio evaluation {times['evaluation.log']:.3f} seconds. Peak allocated GPU memory was {audit['peak_cuda_gib']:.3f} GiB. Eight histories approximately double each full-batch evaluation's work relative to round 2. Every model job used physical GPU 1 serially and the existing environment; no requirements were installed.

39 launch checks passed. The exact checkpoint and complete state agree bitwise. Initial and final weights, moments, RNG, source snapshots, both prepared datasets, histories, every proposal/rejection, intermediate snapshots and all audio measurements are retained. Separate CPU tests cover the proposed next strength-calibration and policy-distillation utilities; neither is a result of this training run. No production package or catalog was changed.

{findings.split(chr(10),2)[2]}

## Locked audio result

| Row | Seed | Coverage candidate | MMD/adaptive | Starting leader | Candidate recall | Parent recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{clips}

| Raw mean | Coverage candidate | MMD/adaptive | Starting leader |
| --- | ---: | ---: | ---: |
{chr(10).join(means)}

The judge stays render-heuristic-v2 on rows 0/1, seeds 7/23, 20 seconds, strength +1 and the same bf16 inference. Off and positive-caption controls are byte-identical across the compared checkpoints. Component weights, eligibility, normalization, descriptions and source code were not retuned. These repeated fixtures are development data; the heuristic ranking is not a new human listening verdict. Review the per-clip lyric and quality changes alongside the mean.

## Confirmation and next experiment

{confirmation_text}

**Next experiment:** {next_test}
'''
    (folder/'training-findings.md').write_text(findings)
    (folder/'notes.md').write_text(notes)
    (folder/'next-experiment.txt').write_text(next_test+'\n')
    print(folder/'notes.md')


if __name__ == '__main__':
    main()
