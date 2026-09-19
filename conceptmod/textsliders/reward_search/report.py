"""A readable progress and decision record for the scored follow-up."""
from pathlib import Path
from collections import Counter

from ..reward_sliders.specs import read_json
from . import queue


def report(run):
    run=Path(run);m=read_json(run/'manifest.json');state=read_json(run/'status.json')
    lines=['# Reward CE search','',f"Current stage: **{state['stage']}**.",'',
           'The user authorized continued score-driven experiments on both GPUs and deprioritized '
           'natural song completion. Primary quality is mean CE across two disjoint ten-second '
           'windows, using one RMS 0.1 copy of the first 20 seconds. Raw audio remains unchanged.','',
           'The new study uses 24 training families (eight verified reused families and sixteen new ones), '
           'eight development families, four causal-check families and eight untouched final families. '
           'The original step-600 LoRA at multiplier 1 remains a final comparison.','',
           '[Frozen protocol](manifest.json) · [Queue status](status.json) · [Prompt validation](prompt-audit.json)','',
           '| Bundle state | Count |','|---|---:|']
    for name,count in queue.counts(run).items():lines.append(f'| {name} | {count} |')
    lines+=['','Each bundle keeps a prompt/seed and every compared arm on the same physical GPU. '
            'GPU 1 starts immediately; GPU 0 joins after the studio queue drains. Playback stays available '
            'while studio generation is paused. The resource manager restores the studio afterward.','']
    for filename,label in [('v1-selection.json','Existing LoRA strength'),('teacher-selection.json','Activation teacher'),
                           ('student-selection.json','New student selection')]:
        path=run/filename
        if not path.exists():continue
        result=read_json(path)
        lines += [f'## {label}','',f"Selected: **{result['selected']['name']}**.",'',
                  '| Arm | Development mean CE |','|---|---:|']
        for name,value in result['mean_ce'].items():
            lines.append(f"| {name} | {'invalid' if value is None else f'{value:.4f}'} |")
        lines.append('')
    if (run/'causal-result.json').exists():
        gate=read_json(run/'causal-result.json')
        lines+=['## Causal check','',f"Training gate: **{'passed' if gate['passed'] else 'closed'}**.",'']
        for key in ('positive_vs_off','positive_vs_random','reversed_vs_off'):
            row=gate[key];lines.append(f"{key}: mean CE change {row['mean_delta']}, seed win rate {row['seed_win_rate']}, "
                                      f"family bootstrap interval {row['family_bootstrap_95']}.")
        lines.append('')
    training=list((run/'students').glob('*/status.json'))
    if training:
        lines+=['## Training','','| Recipe | Stage | Updates |','|---|---|---:|']
        for path in sorted(training):
            value=read_json(path);lines.append(f"| {path.parent.name} | {value['stage']} | {value.get('step',0)} |")
        lines+=['','Both recipes use rank 8 and alpha 8, with full preceding history retained for windows '
                'throughout the first 500 feedback positions. Inference checkpoints contain ordinary LoRA weights; '
                'critics and optimizer states remain training artifacts.','']
    if (run/'results.json').exists():
        result=read_json(run/'results.json')
        lines+=['## Final comparison','',f"Decision: **{result['decision']}**.",'',
                '| Comparison | Valid pairs | Mean CE change | Seed wins | Family bootstrap 95% interval |',
                '|---|---:|---:|---:|---|']
        comparisons={'Original v1 versus Off':result['original_v1_vs_off']}
        for name,group in result['comparisons'].items():
            comparisons[name+' versus Off']=group['vs_off']
            comparisons[name+' versus original v1']=group['vs_original_v1']
        for name,row in comparisons.items():
            mean='unavailable' if row['mean_delta'] is None else f"{row['mean_delta']:+.4f}"
            wins='unavailable' if row['seed_win_rate'] is None else f"{row['seed_win_rate']:.1%}"
            lines.append(f"| {name} | {row['valid_pairs']} | {mean} | {wins} | {row['family_bootstrap_95']} |")
        lines+=['','The final candidate was selected before these new families were rendered. '
                'The result applies to fixed-length excerpts; it does not establish full-song intent preservation. '
                '[Pair statistics and selected checkpoint](results.json).','']
    if (run/'audit/tests.json').exists():lines+=['[Implementation tests](audit/tests.json)','']
    if (run/'audit/listening-page.json').exists():
        page=read_json(run/'audit/listening-page.json')
        lines+=[f"[Listen to all 32 final comparison sets]({page['url']}).",'']
    if (run/'audit/final-integrity.json').exists():
        lines+=['[Final evidence and artifact audit](audit/final-integrity.json)','']
    if (run/'intent/results.json').exists():
        intent=read_json(run/'intent/results.json')
        lines+=['## Intent diagnostics','',
                f"Valid diagnostic observations: {intent['valid_diagnostics']}/{intent['total']}.",'',
                '| Comparison | Within reference tolerances | Reported issues |','|---|---|---:|']
        for name,value in intent['comparisons'].items():
            lines.append(f"| {name} | {'yes' if value['passed'] else 'no'} | {len(value['issues'])} |")
        lines+=['',intent['limitation']+'. '
                'Voice, lyric, style and diversity measurements do not change CE-based selection. '
                '[Pair details and diagnostics](intent/results.json).','']
    if (run/'amendments').exists():
        lines+=['## Engineering amendments','',
                'The initial worker passed a list to a single-adapter API. The error occurred before any '
                'adapter-treated generation. The sole completed Off render was retained and reused; only '
                'unattempted adapter arms resumed after the wrapper repair. Scoring and fixtures were unchanged. '
                '[Archived source, manifest and scheduler error](amendments/0002-single-adapter-component/record.json).','']
        if (run/'amendments/0003-transient-unit-restart/record.json').exists():
            lines+=['A separate orchestration correction recreates expired transient systemd units and starts '
                    'failed units without discarding their definitions. No renders occurred during the failed '
                    'restart. [Restart correction](amendments/0003-transient-unit-restart/record.json).','']
        if (run/'amendments/0004-studio-transient-restart/record.json').exists():
            lines+=['The studio service also used a transient definition. It was restored after the queue '
                    'drained, with the same launch command and previous GPU-0 default. The handoff now uses '
                    'an atomic restart, and a persistent unit preserves its definition. Playback and Keep '
                    'settings were restored before GPU 0 research began. '
                    '[Studio handoff recovery](amendments/0004-studio-transient-restart/record.json).','']
    lines+=['Production slider registration is unchanged. No checkpoint is promoted automatically.','',
            '```bash','systemctl --user status music-reward-search-campaign-20260908.service',
            'journalctl --user -u music-reward-search-campaign-20260908.service -n 30 --no-pager','```','']
    temporary=run/'README.md.tmp';temporary.write_text('\n'.join(lines));temporary.replace(run/'README.md')


if __name__=='__main__':
    import argparse
    from .setup import DEFAULT_RUN
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();report(args.run_dir)
