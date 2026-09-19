"""Write measured round-two conclusions after the additional opponent registers."""
import json
from pathlib import Path
import statistics


NEXT = ('From retained MMD 720, test fixed history coverage: collect one extra seeded history '
        'per training prompt with that frozen parent active at +1, archive it before optimization, '
        'and mix it equally with the original neutral histories. Recompute and freeze float32 '
        'teachers and calibration, use fresh AdamW, and train one 150-attempt paired-MMD candidate. '
        'Keep prompts, rank, kernel and locked audio judge unchanged; measure original/added-history '
        'losses separately. This is a frozen-history surrogate, not full trajectory-distribution MMD.')


def main():
    home = Path(__file__).resolve().parent
    folder = home/'rounds/round-0002'
    state = json.loads((home/'state.json').read_text())
    candidate = state['entries']['round0002_float32']
    parent = state['entries']['mmd_seed_720']
    leader = state['entries']['gan_fm_cap_1950']
    opponent = state['entries']['gan_v2_baseline']
    records = candidate['records']
    completion = json.loads((folder/'float32-candidate/completion.json').read_text())
    entries = sorted(state['entries'].values(), key=lambda item: -item['score'])
    ranking = '\n'.join(f"| {e['id']} | {e['score']:.7f} | {e['worst_score']:.7f} |" for e in entries)
    by_parent = {r['fixture']: r for r in parent['records']}
    by_leader = {r['fixture']: r for r in leader['records']}
    protocol = json.loads((home/'protocol.json').read_text())
    from .game import digest
    row_by_fixture = {digest([protocol['prompts_sha256'], row, seed, protocol['duration']]): row
                      for row in protocol['rows'] for seed in protocol['seeds']}
    clips = '\n'.join(f"| {row_by_fixture[r['fixture']]} | {r['seed']} | {r['heuristic_score']:.7f} | "
                      f"{by_parent[r['fixture']]['heuristic_score']:.7f} | {by_leader[r['fixture']]['heuristic_score']:.7f} | "
                      f"{r['heuristic_score']-by_parent[r['fixture']]['heuristic_score']:+.7f} |" for r in records)
    raw = '\n'.join(f"| {key} | {statistics.mean(r['candidate'][key] for r in records):.6f} | "
                    f"{statistics.mean(r['candidate'][key] for r in parent['records']):.6f} | "
                    f"{statistics.mean(r['candidate'][key] for r in leader['records']):.6f} |"
                    for key in ('concept', 'enjoyment', 'production'))
    execution = [json.loads(line) for line in (home/'execution.jsonl').read_text().splitlines()]
    times = {Path(record['log']).name: record['seconds'] for record in execution
             if str(folder) in record['log'] and record['returncode'] == 0}
    inventory = json.loads((home/'opponents.json').read_text())['opponents']
    pending = sum(row['id'] not in state['entries'] for row in inventory)
    confirmation_path = folder/'confirmation-baseline/summary.json'
    baseline_confirmation = ('The newly benchmarked baseline 660 independently triggered a new-leader '
        'confirmation against FM-cap 1950. Its additional prompts, seeds and duration were declared '
        'before those renders in confirmation-baseline/plan.json. That confirmation is currently '
        '**unfinished**; the 0.002407 development lead is provisional.')
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text())
        groups = {entry['id']: [r for r in confirmation['records'] if r['checkpoint']['path'] == entry['weights']]
                  for entry in (opponent, leader)}
        rows = '\n'.join(f"| {name} | {statistics.mean(r['heuristic_score'] for r in group):.7f} | "
                         f"{min(r['heuristic_score'] for r in group):.7f} | "
                         f"{statistics.mean(r['candidate']['lyric_diagnostics']['recall'] for r in group):.4f} |"
                         for name, group in groups.items())
        reference_by_fixture = {(r['row'], r['seed']): r for r in groups[leader['id']]}
        wins = sum(r['heuristic_score'] > reference_by_fixture[(r['row'], r['seed'])]['heuristic_score']
                   for r in groups[opponent['id']])
        gain = statistics.mean(r['heuristic_score'] for r in groups[opponent['id']])-statistics.mean(
            r['heuristic_score'] for r in groups[leader['id']])
        confirmation_clips = '\n'.join(f"| {r['row']} | {r['seed']} | {r['heuristic_score']:.7f} | "
            f"{reference_by_fixture[(r['row'], r['seed'])]['heuristic_score']:.7f} | "
            f"{r['candidate']['lyric_diagnostics']['recall']:.4f} | "
            f"{reference_by_fixture[(r['row'], r['seed'])]['candidate']['lyric_diagnostics']['recall']:.4f} |"
            for r in groups[opponent['id']])
        baseline_confirmation = f'''The newly measured baseline 660 also became the overall development leader, by only 0.002407 over FM-cap 1950. A separate confirmation was therefore declared before extra renders: rows 2/3, seeds 101/303, 30 seconds, against FM-cap 1950. It is complete in [the raw confirmation and diversity report](confirmation-baseline/summary.json), outside the leaderboard. Both checkpoints used the exact same additional fixtures and controls.

| Reference checkpoint | Additional-fixture mean ↑ | Worst clip | Mean supplied-word recall |
| --- | ---: | ---: | ---: |
{rows}

Baseline 660 wins {wins}/4 additional clips, with a mean advantage of {gain:+.7f}. The confirmation took {times['driver.log']:.3f} seconds including rendering, scoring and diversity diagnostics. Its results support the new reference on this small additional set, while showing an individual lyric regression on row 3/seed 101.

| Row | Seed | Baseline 660 score | FM-cap 1950 score | Baseline recall | FM-cap recall |
| --- | ---: | ---: | ---: | ---: | ---: |
{confirmation_clips}

Within-condition cross-seed chroma, spectral-flux rhythm and speech-model embedding distances are retained for both references, alongside production, lyric recall and high-frequency power. These full-mix proxies and two samples per condition do not establish full mode coverage or a human listening preference. They do not measure the losing float32 candidate's diversity. No triggered confirmation is left unfinished.'''
    notes = f'''# Round 0002: float32 restored descent, but the audio challenge lost

The candidate scored **{candidate['score']:.7f}**, below retained MMD/adaptive by **{-candidate['score']+parent['score']:.7f}** and the round's starting overall leader, FM-cap 1950, by **{-candidate['score']+leader['score']:.7f}**. It lost three clips and improved one. Its worst clip was {candidate['worst_score']:.7f}, versus the parent's {parent['worst_score']:.7f}. This round did not produce a substantially better slider. The current MMD champion is `{state['champions']['mmd']}` and overall leader is `{state['champions']['overall']}`.

[Listening comparison](index.html) · [All per-clip components](comparison.json) · [Precision and optimization findings](training-findings.md) · [State/source audit](state-and-precision-audit.json) · [Training chart](training.svg).

| Entry | Mean score ↑ | Worst clip |
| --- | ---: | ---: |
{ranking}

The extra inventoried opponent `gan_v2_baseline` (baseline 660) was measured on the same locked board and scored {opponent['score']:.7f}. It retains its own exact weights, renders and score reports. {pending} historical opponents remain unmeasured on this board; the result does not rank them.

## Hypothesis, recipe and actual optimization

The hypothesis was that reduced-precision arithmetic obstructed local descent at MMD 720. The parent is also exactly the round-0001 adaptive checkpoint; the earlier subjective preference for adaptive is preserved without claiming those identical weights differ. We verified exact starting weights, outputs, loss, zero-scale identity and RNG. Unchanged bf16 evaluations were deterministic and matched gradient-enabled forwards, yet all four tested negative-gradient step lengths increased loss. With float32 teacher/student arithmetic, three smaller lengths decreased loss and local finite differences closely matched autograd.

The new training uses the existing bf16 base values widened to float32, TF32 disabled, all 4096 hidden coordinates, the same five RBF bandwidths, rank/alpha 8, four unchanged training prompts, both CFG branches and every original frozen neutral-history value. Teachers were recomputed once in float32 and frozen. Calibration changed from 0.2667504187 to 0.2654571645; raw objectives across these precisions do not rank models. AdamW explicitly starts with fresh moments, LR 0.0005, betas (0, 0.999), epsilon 1e-8 and no weight decay. Parent moments are archived. Proposal fractions adapt, with transactional rejection and a negative-gradient fallback; the predeclared stop after three stalled attempts never fired.

All **150 attempted updates were accepted**, using 203 full-batch trial evaluations and 53 rejected intermediate proposals. There were no rejected whole updates and no fallback acceptances. The fixed float32 objective decreased **0.8574034423 → 0.7942526191 (7.37%)**; reserved-prompt loss decreased **1.1001039743 → 1.0892243385 (0.99%)**. Final parameter movement from the parent was L2 3.694962252, and saved Adam counters equal 150. A loss-only improvement failed this audio challenge; these results do not establish convergence or justify more identical training.

The update loop took {completion['training_seconds']:.3f} seconds ({completion['training_seconds']/60:.2f} minutes), and the enclosing successful trainer took {times['training.log']:.3f} seconds. Candidate rendering/scoring took {times['evaluation.log']:.3f} seconds; the additional opponent took {times['opponent-evaluation.log']:.3f} seconds. Peak allocated training GPU memory was 44.716 GiB. Slow disk loading dominated parts of the render time. One sequential cache-warming read pass is retained in cache-prefetch.json; it changed no model files, source, render settings or arithmetic. All model jobs ran serially on physical GPU 1 with the existing minimax-music3 interpreter. No requirements were installed.

34 relevant tests passed. The scored safetensors and saved full-state network are bitwise identical; original history prefix, optimizer, CPU/CUDA RNG, fixed data, every rejected trial, intermediate checkpoints, all model clips, components and transcripts, source snapshots and base-model/tokenizer hashes are retained. The saved states contain the complete information for float32 continuation; its loader must select float32 and reuse prepared.pt with restored moments, not rerun the bf16-to-float32 transition or silently reset optimization. A corrected pre-import syntax error preceded all model loading and training; both invocations remain logged. Nothing was published or promoted to the production catalog.

## Audio evidence and individual failures

| Prompt row | Seed | Float32 candidate | MMD/adaptive | FM-cap 1950 | Change vs MMD |
| --- | ---: | ---: | ---: | ---: | ---: |
{clips}

| Raw mean | Float32 candidate | MMD/adaptive | FM-cap 1950 |
| --- | ---: | ---: | ---: |
{raw}

Compared with MMD/adaptive, mean concept margin, enjoyment and production all fell. Mean supplied-word recall increased from 0.7981 to 0.8558, but phrase accuracy fell from 0.9300 to 0.9137. The combined lyric component improved slightly while remaining below the exact off controls. Row 1/seed 23 improved recall from 0.4231 to 0.9615 and its score rose by 0.7083. Row 1/seed 7 instead lost recall from 1.0 to 0.6154 and its score fell by 1.0178. Better average recall does not erase that failure. The candidate was eligible and incurred no excess-high-frequency penalty; this is not an assertion that its audio has no artifacts.

The judge is unchanged render-heuristic-v2 on rows 0/1, seeds 7/23, 20 seconds, strength +1 and bf16 inference, with byte-identical shared controls. Coefficients, normalization, descriptions, prompts, seeds and failure eligibility were not retuned. These four clips are repeated development fixtures and a heuristic comparison, not a new human listening verdict.

## Confirmation, diversity and next experiment

A 30-second comparison of the float32 candidate against FM-cap 1950 on rows 2/3 and seeds 101/303 was predeclared before training. Its trigger was a new MMD personal best or overall lead. The candidate did not trigger it, so confirmation/outcome.json records `not_triggered`. The float32 candidate's new-prompt generalization and cross-seed musical diversity remain unmeasured. Distinct WAVs and a falling hidden loss do not establish either. No discriminator, independent conditional particles or fake-fake kernel term was introduced; paired hidden matching retains its known limitations.

{baseline_confirmation}

**Next experiment:** {NEXT}

This next test targets a plausible mismatch between the neutral histories used for optimization and histories encountered with the slider active. The current results do not prove that this mismatch caused the regression. They show that repairing descent alone was insufficient, so another identical continuation is not the justified next round. Preserve the MMD/adaptive champion while testing the new coverage, and require the same audio judge plus triggered additional-fixture checks for any new record.
'''
    (folder/'notes.md').write_text(notes)
    (folder/'next-experiment.txt').write_text(NEXT+'\n')
    print(folder/'notes.md')


if __name__ == '__main__':
    main()
