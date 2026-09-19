"""Diagnostic tables, without turning embedding/ASR scores into a winner."""
import csv
import json
from common import WORK, PAIRS, sha, write


def main():
    spec=json.loads((WORK/'screen.json').read_text())
    renders=json.loads((WORK/'renders.json').read_text())
    measured=json.loads((WORK/'measurements.json').read_text())
    feedback_path=WORK/'listening-feedback.json'
    feedback=json.loads(feedback_path.read_text()) if feedback_path.exists() else None
    if feedback:
        assert feedback['screen_sha256']==sha(WORK/'screen.json')
        assert feedback['blind_key_sha256']==sha(WORK/'blind-key.json')
    labels=dict(linear='Ordinary E2.8',ties='TIES',knots_ties='KnOTS-TIES',linear_e2='Ordinary E2 reference')
    rows=[]
    for j in spec['jobs']:
        rec=renders['records'].get(j['id'],{});m=measured['records'].get(j['id'],{})
        if rec.get('status')!='complete' or m.get('status')!='complete':continue
        a,b=j['pair']
        rows.append(dict(id=j['id'],fixture=j['fixture'],pair='+'.join(j['pair']),method=j['method'],
                         seconds=rec['inspection']['seconds'],clipped_fraction=rec['inspection']['clipped_fraction'],
                         concept_a=m['concepts'][a],concept_b=m['concepts'][b],
                         **m['aesthetics'],**m['lyric_diagnostics']))
    if rows:
        with (WORK/'results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    geometry=[]
    for pair in PAIRS:
        for method in ('ties','knots_ties'):
            p=WORK/'artifacts'/'+'.join(pair)/method/'manifest.json'
            modules=json.loads(p.read_text())['modules']
            total=sum(m['reference_norm']**2 for m in modules)
            cosine=sum(m['reference_norm']*m['matched_norm']*m['cosine_to_linear'] for m in modules)/total
            geometry.append(dict(pair='+'.join(pair),method=method,cosine_to_linear=cosine,
                max_norm_relative_error=max(abs(m['matched_norm']/m['reference_norm']-1) for m in modules),
                normalization_scale_min=min(m['scale'] for m in modules),normalization_scale_max=max(m['scale'] for m in modules),
                modules=len(modules)))
    write(WORK/'geometry-summary.json',geometry)
    lines=['# Merger study diagnostics','',
           ('Human listening preferences are recorded for all six comparisons and take precedence over the proxy scores below. No production default has changed.' if feedback else
            'Human listening preferences for this comparison are pending. The earlier gain study favored ordinary energy 2.8 for all three pairs. No production default has changed.'),'',
           f"Recordings: {sum(r['status']=='complete' for r in renders['records'].values())}/24 complete. Six are verified reuse; eighteen are new. Full audio and exact first-20-second excerpts are available in the gallery.",'',
           '## Geometry','',
           'Each of the 144 projection norms matches ordinary energy 2.8. The cosine below measures update direction; it is not musical similarity or quality.','',
           '| Pair | Method | Cosine to ordinary | Maximum relative norm error |','|---|---|---:|---:|']
    if feedback:
        listening=['## User listening results','',
            '[Exact feedback and decoded key](listening-feedback.json). Rankings are best to worst; the close and uncertain judgments are retained.','',
            '| Song | Pair | Ranking | Qualification |','|---|---|---|---|']
        for rec in feedback['records']:
            ranking=' > '.join(labels[r['method']] for r in rec['ranking_best_to_worst'])
            qualification=' '.join(rec['qualifications']) or 'None stated'
            listening.append(f"| {rec['fixture']} | {' + '.join(rec['pair'])} | {ranking} | {qualification} |")
        listening+=['',
            'Ordinary E2.8 ranks first in three groups, KnOTS-TIES in two, and TIES in one. The TIES win is the explicitly uncertain female + pop confirmation group. TIES ranks last in four groups. Counts summarize order only; they do not measure how large the preference was.','',
            'KnOTS-TIES beats ordinary mixing for female + pop on both songs, although the confirmation ranking was hard to distinguish. Ordinary wins country + indie rock twice; on confirmation it is close to TIES, with KnOTS-TIES well behind. House + acoustic folk splits between KnOTS-TIES and ordinary, with TIES last on both songs.','',
            'Recommendation: retain ordinary full-delta addition as the studio default. KnOTS-TIES merits a focused follow-up for female + pop and a possible optional mode; it does not justify a universal replacement or automatic pair-based routing. The current TIES setting is a lower priority given its four last places and uncertain lone win.','',
            'The user did not rank the energy-2 references. These results select among merger methods at matched parameter strength; they do not independently confirm energy 2.8 over energy 2 on the second song.','']
        index=lines.index('## Geometry')
        lines[index:index]=listening
    if (WORK/'audit.json').exists() and json.loads((WORK/'audit.json').read_text())['status']=='passed':
        lines[2:2]=['[Final audit](audit.json): passed all 24 recordings, 24 measurements and 48 audio URLs. [Listen in the studio](http://192.168.1.90:7860/studio-mergers-20260907/).','']
    for g in geometry:lines.append(f"| {g['pair']} | {g['method']} | {g['cosine_to_linear']:.4f} | {g['max_norm_relative_error']:.2g} |")
    lines+=['','## Audio diagnostics','',
            'The CSV includes CLAP concept margins, aesthetics, lyric recognition and signal checks. These proxies can disagree with listeners. ASR recall against the complete lyric sheet is also duration-dependent; an instrumental opening is not automatically a vocal failure. No overall score or automated winner is computed.','',
            'Interpret listening group by group, then compare the same pair on the second song. A single 30% pruning setting does not test every possible TIES/KnOTS variant.']
    if len(rows)==24:
        lines+=['', 'Observed proxy tradeoffs: ordinary E2.8 has higher country and indie-rock CLAP margins than either merger on both songs. For female + pop, KnOTS-TIES has higher PQ/CE predictions and pop margins than ordinary E2.8 on both songs, with lower female margins. House + acoustic folk changes are mixed across songs and metrics. These patterns motivate listening comparisons; they do not establish a preferred merger or a universal replacement.']
    lines+=['', 'Concept columns are CLAP margins against fixed descriptions; compare within each concept column. PQ and CE are aesthetics-model predictions. Phrase match is a first-20-second ASR diagnostic, not a human intelligibility rating.']
    for fixture in ('familiar','confirmation'):
        for pair in PAIRS:
            selected=[r for r in rows if r['fixture']==fixture and r['pair']=='+'.join(pair)]
            if not selected:continue
            lines+=['',f"### {fixture}: {' + '.join(pair)}",'',
                f'| Method | {pair[0]} | {pair[1]} | PQ | CE | Phrase match |',
                '|---|---:|---:|---:|---:|---:|']
            for r in selected:
                lines.append(f"| {labels[r['method']]} | {r['concept_a']:.3f} | {r['concept_b']:.3f} | {r['PQ']:.3f} | {r['CE']:.3f} | {r['phrase_accuracy']:.3f} |")
    (WORK/'results.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {len(rows)} diagnostic rows and six geometry summaries.')


if __name__=='__main__':main()
