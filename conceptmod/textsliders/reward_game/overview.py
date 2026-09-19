"""Combined read-only evidence view for compatible development games.

This writes a derived page and JSON report. It never scores or renders audio,
changes a scientific decision, or promotes a development winner.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time

from .core import read, write, digest, sha, scorecard, IntegrityError, ROOT


CHILDREN = ('acoustic-v1', 'joint-v1', 'ff-v1', 'block-v1')
SHARED_FIELDS = ('reward_spec', 'sampler', 'duration_seconds', 'scored_seconds',
                 'controls', 'original_reference', 'cases', 'stages', 'rules')


def comparison_signature(game):
    return digest({k: game[k] for k in SHARED_FIELDS})


def verified_card(game, card):
    if card['benchmark_sha256'] != digest(game):
        raise IntegrityError('Scorecard belongs to a different benchmark')
    observations = {}
    for row in card['rows']:
        if row['case'] in observations:
            raise IntegrityError('Duplicate scorecard case')
        observations[row['case']] = dict(status='complete',
            reward=dict(valid=True, scalar=row['ce']), audio=row.get('audio'))
    rebuilt = scorecard(game, observations, card['stage'])
    for key in ('comparisons', 'advance', 'valid_cases', 'scheduled_cases', 'rows'):
        if rebuilt[key] != card[key]:
            raise IntegrityError('Scorecard values do not reproduce: '+key)
    return card


def ranking(row):
    card = row['card']; off = card['comparisons']['off']
    original = card['comparisons']['v1-original']
    number = lambda x: -float('inf') if x is None else x
    return (card['advance'], off['wins'], number(off['worst_delta']),
            number(off['equal_family_gain']), number(original['equal_family_gain']))


def candidate_label(candidate):
    if candidate.get('mode') == 'off' or not candidate.get('path'):
        return candidate.get('mode', 'Off')
    path = Path(candidate['path'])
    return path.parent.name if path.stem == 'candidate' else path.stem


def snapshot(home):
    home = Path(home).resolve(); root_record = read(home/'game.json')
    root_game = read(root_record['spec'])
    if digest(root_game) != root_record['benchmark_sha256']:
        raise IntegrityError('Root game changed')
    signature = comparison_signature(root_game)
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), confirmed=False,
        full_development=[], early_development=[], incomplete=[], fresh=[],
        pending_attempts=[], excluded=[], source_hashes={})
    for game_home in [home]+[home/c for c in CHILDREN if (home/c/'game.json').exists()]:
        name = 'language-model' if game_home == home else game_home.name
        record = read(game_home/'game.json'); game = read(record['spec'])
        if (digest(game) != record['benchmark_sha256'] or comparison_signature(game) != signature
                or (game_home != home and Path(record.get('parent_home', '/')).resolve() != home)):
            result['excluded'].append(dict(game=name, reason='Different frozen cases, controls, reward, sampler or gates'))
            continue
        off = game_home/'audit/off-pcm-v1/result.json'
        if not off.exists() or not read(off)['passed']:
            result['excluded'].append(dict(game=name, reason='Off compatibility audit missing or failed'))
            continue
        result['source_hashes'][str(Path(record['spec']))] = sha(record['spec'])
        page = read(game_home/'listening-page.json').get('path') if (game_home/'listening-page.json').exists() else None
        for folder in sorted((game_home/'evaluations').glob('*')):
            path = folder/'scorecard.json'
            if not path.exists():
                if (folder/'initial.json').exists():
                    result['incomplete'].append(dict(game=name, run=folder.name))
                continue
            try:
                card = verified_card(game, read(path))
            except (IntegrityError, KeyError, ValueError) as exc:
                result['excluded'].append(dict(game=name, run=folder.name, reason=str(exc)))
                continue
            candidate = read(folder/'candidate.json')
            row = dict(game=name, label=candidate_label(candidate), candidate=candidate,
                       card=card, scorecard=str(path), scorecard_sha256=sha(path), listening_page=page)
            if card['valid_cases'] != card['scheduled_cases']:
                result['incomplete'].append(row)
            elif card['stage'] == 16:
                result['full_development'].append(row)
            else:
                result['early_development'].append(row)
        for folder in sorted((game_home/'confirmation').glob('*')):
            path = folder/'scorecard.json'
            if not path.exists():
                continue
            card = read(path); protocol = read(folder/'protocol.json')
            if card['protocol_sha256'] != digest(protocol):
                result['excluded'].append(dict(game=name, batch=folder.name, reason='Fresh scorecard/protocol mismatch'))
                continue
            result['fresh'].append(dict(game=name, batch=folder.name, card=card,
                candidate_sha256=protocol['candidate']['weights_sha256'],
                protocol_sha256=digest(protocol), scorecard_sha256=sha(path)))
        for folder in sorted((game_home/'attempts').glob('attempt-*')):
            if (folder/'decision.json').exists():
                continue
            status = read(folder/'status.json'); recipe = read(folder/'recipe.json')
            training_status = recipe.get('training_status')
            training = read(training_status) if training_status and Path(training_status).exists() else {}
            result['pending_attempts'].append(dict(game=name, attempt=folder.name,
                state=status['state'], updates=training.get('actual_updates', 0),
                target_updates=recipe.get('expected_updates'), training_state=training.get('state'),
                pending_batch_cases=training.get('completed_cases_in_pending_batch'),
                cases_per_batch=training.get('cases_per_batch'),
                elapsed_seconds=max(0., time.time()-status.get('started_unix', time.time())),
                error=status.get('error')))
    result['full_development'].sort(key=ranking, reverse=True)
    result['early_development'].sort(key=lambda row: (row['card']['stage'], ranking(row)), reverse=True)
    # Only an explicitly packaged, ledger-recorded research completion counts.
    status = read(home/'status.json') if (home/'status.json').exists() else {}
    if status.get('research_complete') and status.get('state') == 'confirmed':
        evidence = Path(status['package'])/'evidence.json'
        events = [json.loads(line) for line in (home/'ledger.jsonl').read_text().splitlines() if line]
        result['confirmed'] = evidence.exists() and any(
            e.get('kind') == 'research_confirmed' and e.get('package') == status['package']
            and e.get('evidence_sha256') == sha(evidence) for e in events)
        if result['confirmed']:
            result['confirmed_package'] = status['package']
    return result


def render(result, listen_root):
    e = html.escape
    def link(path, label):
        if path:
            try:
                relative = Path(path).resolve().relative_to(listen_root.resolve())
                return f'<a href="../{e(str(relative))}">{e(label)}</a>'
            except ValueError:
                pass
        return e(label)
    fmt = lambda value: '—' if value is None else f'{value:+.3f}'
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="refresh" content="60"><title>Reward slider results</title>',
        '<style>body{font:16px/1.5 system-ui;margin:2rem;max-width:1100px;background:#15171c;color:#e7e9ef}a{color:#8fc3ff}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{text-align:left;padding:.6rem;border-bottom:1px solid #414653}code{overflow-wrap:anywhere}small{color:#b6bbc6}.scroll{overflow-x:auto}</style>',
        '<h1>Reward slider results</h1>',
        '<p>'+('A confirmed research package is available.' if result['confirmed'] else
        '<strong>No confirmed improving incumbent.</strong> Off remains eligible. Passing development still requires fresh confirmation, independent replication, preservation and composition checks.')+'</p>',
        '<p><small>Updated '+e(result['updated_utc'])+'</small></p>']
    if result['pending_attempts']:
        parts += ['<h2>Current work</h2><ul>']
        for row in result['pending_attempts']:
            steps = '' if row['target_updates'] is None else f" · {row['updates']}/{row['target_updates']} optimizer updates"
            if row.get('pending_batch_cases') is not None:
                count = str(row['pending_batch_cases'])
                if row.get('cases_per_batch'):
                    count += f"/{row['cases_per_batch']}"
                label = 'case' if count == '1' else 'cases'
                steps += f" · {count} {label} accumulated for the next update"
            elapsed = f" · {row['elapsed_seconds']/60:.0f} min elapsed"
            parts += [f"<li>{e(row['game'])}: {e(row['state'])}{e(steps+elapsed)} <small>{e(row['attempt'])}</small></li>"]
        parts += ['</ul>']
    for title, key in [('Completed development', 'full_development'), ('Early development screens', 'early_development')]:
        parts += [f'<h2>{title}</h2>']
        if key == 'full_development':
            parts += ['<p>Same exposed cases and controls. Ranking: gate pass, wins, worst delta, mean gain, then mean against the original. Gains are equal-family CE differences. Meaningful wins are at least +0.02 CE.</p>']
        else:
            parts += ['<p>Small screens retain their actual denominators. These are not full development results.</p>']
        parts += ['<div class="scroll"><table><tr><th>Candidate / listen</th><th>Gate</th><th>Wins vs Off</th><th>Meaningful</th><th>Mean</th><th>Worst</th><th>Mean vs original</th></tr>']
        for row in result[key]:
            c = row['card']; off = c['comparisons']['off']; original = c['comparisons']['v1-original']
            label = link(row['listening_page'], row['label'])
            note = ' · forced full diagnostic' if c.get('forced_stage') else ''
            parts += [f"<tr><td>{label}<br><small>{e(row['game'])} · multiplier {row['candidate'].get('multiplier',0)}{e(note)}</small></td><td>{'Pass' if c['advance'] else 'Reject'}</td><td>{off['wins']}/{c['stage']}</td><td>{off['meaningful_wins']}/{c['stage']}</td><td>{fmt(off['equal_family_gain'])}</td><td>{fmt(off['worst_delta'])}</td><td>{fmt(original['equal_family_gain'])}</td></tr>"]
        parts += ['</table></div>']
    parts += ['<h2>Fresh confirmation attempts</h2><p>Each batch uses different families and seeds. These rows are separate confirmation outcomes, not matched rankings. All failed outcomes are retained.</p>',
        '<div class="scroll"><table><tr><th>Batch / listen</th><th>Valid</th><th>Wins vs Off</th><th>Mean vs Off</th><th>95% Off interval</th><th>95% original interval</th><th>Batch gate</th></tr>']
    pages = [(p, p.read_text()) for p in listen_root.glob('*fresh*/index.html')]
    for row in result['fresh']:
        c = row['card']; off = c['comparisons']['off']; original = c['comparisons']['original']
        page = next((str(p) for p,text in pages if f"Fresh batch {row['batch']}:" in text), None)
        def interval(values):
            return 'unavailable' if values is None else f"[{fmt(values['low'])}, {fmt(values['high'])}]"
        parts += [f"<tr><td>{link(page,row['batch'])}</td><td>{c['valid_cases']}/{c['scheduled']}</td><td>{off['wins']}/{c['valid_cases']}</td><td>{fmt(off['equal_family_gain'])}</td><td>{interval(off['interval'])}</td><td>{interval(original['interval'])}</td><td>{'Pass; further checks required' if c['batch_pass'] else 'Not passed'}</td></tr>"]
    parts += ['</table></div>']
    if result['incomplete']:
        parts += ['<p>Incomplete development runs: '+str(len(result['incomplete']))+'. No full result is inferred.</p>']
    if result['excluded']:
        parts += ['<h2>Excluded from combined comparison</h2><pre>'+e(json.dumps(result['excluded'],indent=2))+'</pre>']
    parts += ['<p><small>First-20-second CE research. Natural full-song completion remains separate. This page creates no audio and changes no production settings.</small></p></html>']
    return '\n'.join(parts)


def publish(home, folder):
    folder = Path(folder).resolve(); listen_root = ROOT/'eval/listen'
    if folder.parent != listen_root.resolve():
        raise ValueError('Overview must be one folder under eval/listen')
    result = snapshot(home); folder.mkdir(parents=True, exist_ok=True)
    write(folder/'overview.json', result)
    temporary = folder/'index.html.tmp'; temporary.write_text(render(result, listen_root))
    temporary.replace(folder/'index.html')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--home', required=True)
    parser.add_argument('--folder', default=str(ROOT/'eval/listen/reward-game-current'))
    parser.add_argument('--watch', action='store_true'); args = parser.parse_args()
    while True:
        result = publish(args.home, args.folder)
        print(json.dumps(dict(updated_utc=result['updated_utc'], full=len(result['full_development']),
                              fresh=len(result['fresh']), excluded=result['excluded'])), flush=True)
        if not args.watch:
            break
        time.sleep(30)
