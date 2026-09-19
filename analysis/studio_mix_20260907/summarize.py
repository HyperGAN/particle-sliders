"""Export transparent per-concept diagnostics, without a composite winner score."""
import csv
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent


def main():
    spec = json.loads((WORK/'screen.json').read_text())
    renders = json.loads((WORK/'renders.json').read_text())
    measured = json.loads((WORK/'measurements.json').read_text())
    pairs = [('female','pop'), ('country','indie-rock'), ('house','acoustic-folk')]
    complete = sum(r['status'] == 'complete' for r in renders['records'].values())
    lines = ['# Studio mix listening results', '',
             f'Render status: **{renders["status"]}** ({complete}/25). Measurement status: **{measured["status"]}**.', '',
             '[Listen in the studio](http://localhost:7860/studio-mix-20260907/) · [Protocol](README.md) · [Exact jobs](screen.json)', '',
             'The table compares the same 50/50 mix at three energies, with the same caption, lyric sheet and seed.',
             'Concept columns are CLAP margins against fixed descriptions. Compare changes within each column; different concepts do not share an audible strength scale.',
             'Phrase match is a Whisper transcript diagnostic for the first 20 seconds. Low values can reflect late vocals or ASR failure as well as wrong words. PQ and CE are model predictions, not listener preference.',
             'All excerpts use their original amplitude. The measurement models use separate RMS-normalized copies; the source recordings are preserved.', '']
    lines += ['Each of these six checkpoints lists a recommended multiplier range of 0–1. An equal pair at studio energy 2 already applies multiplier 1 to each adapter; energy 2.8 applies 1.4 to each. A solo at energy 2 applies multiplier 2. The reduced norm of a blend relative to a solo at energy 2 therefore does not by itself establish that the blend is too weak.', '']
    rows = []
    for job in spec['jobs']:
        record = renders['records'].get(job['id'], {})
        m = measured['records'].get(job['id'], {})
        row = dict(id=job['id'], label=job['label'], render_status=record.get('status','pending'),
                   measurement_status=m.get('status','pending'), energy=job['energy']['language_model'],
                   audio=record.get('audio',''))
        if record.get('status') == 'complete':
            row.update(record['inspection'])
        if m.get('status') == 'complete':
            row.update({f'concept_{k}':v for k,v in m['concepts'].items()})
            row.update({f'aesthetics_{k}':v for k,v in m['aesthetics'].items()})
            row.update({f'lyric_{k}':v for k,v in m['lyric_diagnostics'].items()})
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (WORK/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for a,b in pairs:
        lines += [f'## {a} + {b}', '',
                  f'| Energy | {a} margin | {b} margin | Phrase match | PQ | CE |',
                  '|---:|---:|---:|---:|---:|---:|']
        for job in spec['jobs']:
            if {s['id'] for s in job['sliders']} != {a,b}:
                continue
            m = measured['records'].get(job['id'], {})
            if m.get('status') != 'complete':
                lines.append(f'| {job["energy"]["language_model"]:g} | pending | | | | |')
                continue
            vals = [job['energy']['language_model'], m['concepts'][a], m['concepts'][b],
                    m['lyric_diagnostics']['phrase_accuracy'], m['aesthetics']['PQ'], m['aesthetics']['CE']]
            lines.append('| '+' | '.join(f'{v:.3f}' for v in vals)+' |')
        lines.append('')
    feedback_path = WORK/'listening-feedback.json'
    if feedback_path.exists():
        feedback = json.loads(feedback_path.read_text())['entries'][-1]
        if feedback['screen_sha256'] != renders['screen_sha256']:
            raise ValueError('Listening feedback belongs to a different screen')
        lines += ['## User listening result', '',
                  'The user preferred energy **2.8 in all three pairs**. These preferences take precedence over the automatic diagnostics above.', '',
                  '| Pair | Best to worst energy | Qualification |',
                  '|---|---|---|']
        for ranking in feedback['rankings']:
            order = ', '.join(f'{r["energy"]:g}' for r in ranking['best_to_worst'])
            note = 'Top two were close' if ranking['top_two_close'] else 'No closeness stated'
            lines.append(f'| {" + ".join(ranking["sliders"])} | {order} | {note} |')
        lines += ['', 'Next confirmation: energies 2.0 and 2.8 on reserved row 3 / seed 303 for all three pairs (six renders). No studio default is changed by these one-fixture rankings.', '']
    lines += ['## Scope', '',
              'One fixture, one seed, three authored pairs. The study does not establish a new studio default or test all sixteen adapters in combination.',
              'No confirmation runs on row 3 / seed 303 are included. Those remain reserved until a listening preference or clear failure hypothesis selects the next comparison.', '',
              'Raw render records: `renders.json`. Model measurements and transcripts: `measurements.json`. All individual results: `results.csv`.', '']
    (WORK/'results.md').write_text('\n'.join(lines))
    print(f'Summarized {complete} renders and {sum(r["status"] == "complete" for r in measured["records"].values())} measurements.')


if __name__ == '__main__':
    main()
