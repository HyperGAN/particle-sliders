"""Summarize all completed objective-audit runs, including failures."""
from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    gaussians=[]
    for completion in sorted((HERE/'runs').glob('*/completion.json')):
        folder=completion.parent
        args=json.loads((folder/'invocation.json').read_text())
        progress=[json.loads(line) for line in (folder/'progress.jsonl').read_text().splitlines()]
        gaussians.append(dict(name=folder.name,args=args,final=progress[-1],history=progress))
    spans=json.loads((HERE/'span-results.json').read_text())['runs'] if (HERE/'span-results.json').exists() else []
    live=[]
    for folder in [ROOT/'models/conditional-energy-research-20260905',ROOT/'models/conditional-energy-cfg-research-20260905-attempt2',ROOT/'models/conditional-mmd-cfg-research-20260905']:
        if not (folder/'completion.json').exists():continue
        complete=json.loads((folder/'completion.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        evaluations=[json.loads(p.read_text()) for p in sorted(folder.glob('evaluation-*.json'))]
        history=[json.loads(s) for s in (folder/'train.jsonl').read_text().splitlines()]
        live.append(dict(name=folder.name,complete=complete,manifest=manifest,evaluations=evaluations,
            shortened=sum(h['line_search']['accepted'] and h['line_search']['fraction']<1 for h in history),
            rejected=sum(not h['line_search']['accepted'] for h in history),
            max_step=max(h['parameter_step'] for h in history),
            fixed_objective_monotone=all(b['line_search']['loss']<=a['line_search']['loss']+1e-7 for a,b in zip(history,history[1:])) if manifest['arguments'].get('fixed_kernel') else None,
            all_accepted_decreased=all(h['line_search']['loss']<h['loss_before'] for h in history if h['line_search']['accepted'])))
    occupancy=json.loads((HERE/'mode-occupancy.json').read_text()) if (HERE/'mode-occupancy.json').exists() else None
    occupancy_rows={r['run']:r for r in occupancy['runs']} if occupancy else {}
    summary=dict(gaussians=gaussians,span_games=spans,live=live,mode_occupancy=occupancy)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Objective experiments — complete results','',
        'These are research screens, not a universal stability result. Every completed run is included. '
        'Gaussian HQ and core width are distribution metrics; saved-span error and live teacher error are not audio-quality scores.','',
        '| Gaussian run | Steps | Batch | Schedule | EMA modes | EMA HQ | EMA core width / truth | EMA mode TV | Live HQ |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in gaussians:
        a,f=row['args'],row['final'];m=f['ema']
        tv=f"{occupancy_rows[row['name']]['ema']['mode_total_variation']:.4f}" if row['name'] in occupancy_rows else 'unmeasured'
        lines.append(f"| {row['name']} | {f['step']} | {a['batch']} | {a['schedule']} | {m['modes']} | {m['hq']:.4f} | {m['core']['per_mode_core_ratio']:.3f} | {tv} | {f['live']['hq']:.4f} |")
    lines+=['','Mode TV is total-variation error from uniform nearest-mode probabilities, including tail samples. '
        'At 20,000 independent uniform assignments its simulated sampling reference averages .0281. '
        'The initial `energy-seed7` run hit an audit-argument error at final evaluation. '
        'It has no completed checkpoint and was rerun as `energy-seed7-repaired-audit`; its intermediate log is retained. '
        'The constant batch-32 seed-23 stress arms jointly change batch, schedule, seed and horizon. '
        'The later seed-7 cap controls isolate removing decay (live HQ .9828→.2433) and changing batch 256→32 (EMA HQ .9876→.2292). '
        'At fixed MMD batch 1024, removing decay changes EMA HQ .8224→.3576 and live HQ .8288→.1133.',
        '', '| Saved-span game | Seeds | Mean relative error at 1200 | Range |', '| --- | ---: | ---: | --- |']
    for arm in sorted({r['arm'] for r in spans}):
        values=[r['final']['nrmse'] for r in spans if r['arm']==arm]
        lines.append(f'| {arm} | {len(values)} | {statistics.mean(values):.4f} | {min(values):.4f}–{max(values):.4f} |')
    lines+=['','The span canary optimizes free hidden values from a 64-coordinate slice of archived states. '
        'It uses fixed source-600 outputs as critic context, not the actual neutral hidden context. '
        'It is a falsification fixture, not a LoRA capacity or audio benchmark. '
        'A discriminator loss near log(2) with large target error is not successful matching.','',
        '| Real LoRA continuation | Teacher branches | Imported G moments | Train error before → after | Heldout error before → after | Shortened / rejected proposals | Maximum parameter step |',
        '| --- | --- | --- | --- | --- | ---: | ---: |']
    for row in live:
        first,last=row['evaluations'][0],row['evaluations'][-1]
        def error(e,group):return statistics.mean(r['relative_error'] for r in e[group])
        manifest=row['manifest']
        lines.append(f"| {row['name']} | {', '.join(manifest.get('branches',['conditional']))} | {'no' if manifest['arguments'].get('fresh_optimizer') else 'yes'} | {error(first,'train'):.4f} → {error(last,'train'):.4f} | {error(first,'heldout'):.4f} → {error(last,'heldout'):.4f} | {row['shortened']} / {row['rejected']} | {row['max_step']:.4f} |")
    lines+=['','All live runs start from the same source-600 LoRA. The second energy trial changes both branch coverage '
        'and optimizer initialization; it is a combined implementation test, not an isolated CFG ablation. '
        'The fixed-MMD trial uses the same fresh moments and both branches as the second energy trial. '
        'The first CFG attempt was stopped before training to move source capture ahead of model I/O. '
        'The two heldout prompts are diagnostic development fixtures. No run here establishes free-running convergence.','']
    (HERE/'results.md').write_text('\n'.join(lines))
    if gaussians:
        fig,axes=plt.subplots(1,2,figsize=(12,4.4))
        for row in gaussians:
            if row['name'] not in ['bcap-seed7','r1r2-seed7','learned-energy-seed7','mmd-seed7','mmd-batch1024-extended-seed7','energy-seed7-repaired-audit']:continue
            axes[0].plot([x['step'] for x in row['history']], [x['ema']['hq'] for x in row['history']],label=row['name'].replace('-seed7',''))
        axes[0].set(xlabel='Optimizer updates',ylabel='High-quality fraction (EMA)',ylim=(0,1.03),title='Gaussian fidelity versus training budget')
        axes[0].legend(fontsize=7)
        if live:
            for row in live:
                for group,style in [('train','-'),('heldout','--')]:
                    axes[1].plot([x['step'] for x in row['evaluations']],
                        [statistics.mean(r['relative_error'] for r in x[group]) for x in row['evaluations']],style,
                        label=('MMD, both' if 'mmd' in row['name'] else ('energy, both' if 'cfg' in row['name'] else 'energy, conditional'))+' / '+group)
            axes[1].set(xlabel='Source updates + research updates',ylabel='Relative full-state error',title='Real LoRA, teacher-forced histories')
            axes[1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=160);fig.savefig(HERE/'comparison.svg');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        by_name={r['name']:r for r in gaussians}
        for ax,title,pair in zip(axes,['Cap objective, batch 256','Fixed MMD, batch 1024'],[
            ('bcap-seed7','bcap-constant-seed7'),('mmd-batch1024-seed7','mmd-batch1024-constant-seed7')]):
            for name,color,label in zip(pair,['#176a9b','#b54825'],['decay','constant LR']):
                if name not in by_name:continue
                rows=by_name[name]['history']
                for kind,style in [('live','-'),('ema','--')]:
                    ax.plot([r['step'] for r in rows],[r[kind]['hq'] for r in rows],style,
                        color=color,label=f'{label}, {kind}')
            ax.set(title=title,xlabel='Optimizer updates',ylabel='High-quality fraction',ylim=(0,1.03))
            ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(HERE/'optimizer-comparison.png',dpi=160)
        fig.savefig(HERE/'optimizer-comparison.svg');plt.close(fig)


if __name__=='__main__':main()
