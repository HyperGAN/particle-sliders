#!/usr/bin/env python3
"""Summarize intervention training and telemetry without assigning audio quality."""
import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--max-step',type=int,help='Limit diagnostics to a saved checkpoint before an intentional stop')
    args=p.parse_args()
    reports=[];fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    bounds=set()
    for index,folder in enumerate(args.runs):
        color=plt.get_cmap('tab10')(index%10)
        path=folder/f'{folder.name}_train.jsonl'
        if not path.exists(): continue
        rows=read_rows(path)
        telemetry_path=folder/f'{folder.name}_telemetry.jsonl'
        telemetry=read_rows(telemetry_path) if telemetry_path.exists() else []
        if args.max_step is not None:
            rows=[r for r in rows if r['step']<=args.max_step]
            telemetry=[r for r in telemetry if r['step']<=args.max_step]
        selected=[r for r in rows if r['step']>600]
        if not selected: continue
        means={k:statistics.mean(r[k] for r in selected[-30:]) for k in
               ['loss','pperc','cos_pos','mag_ratio','g_adv','fm','d_pen','grad_norm','edrift_p']}
        record=dict(run=folder.name,completed_updates=rows[-1]['step'],new_updates=len(selected),
                    last_window=len(selected[-30:]),last_window_means=means,
                    logs_finite=all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(float,int))))
        if telemetry:
            record['telemetry']=dict(steps=len(telemetry),cap_active_rows=sum(r['fm_cap_active_rows'] for r in telemetry),
                                     measured_rows=sum(len(r['fm_rows']) for r in telemetry),
                                     last_window_update_norm=statistics.mean(r['parameter_update_norm'] for r in telemetry[-30:]),
                                     max_update_norm=max(r['parameter_update_norm'] for r in telemetry),
                                     minimum_fm_factor=min(r['factor'] for t in telemetry for r in t['fm_rows']))
        step_limit_path=folder/f'{folder.name}_telemetry_step_limit.jsonl'
        if step_limit_path.exists():
            limits=read_rows(step_limit_path)
            if args.max_step is not None:
                limits=[r for r in limits if r['step']<=args.max_step]
            if limits:
                record['parameter_step_limit']=dict(measured_updates=len(limits),
                    active_updates=sum(r['factor']<1 for r in limits),
                    largest_proposed_norm=max(r['proposed_update_norm'] for r in limits),
                    largest_applied_norm=max(r['actual_update_norm'] for r in limits),
                    minimum_factor=min(r['factor'] for r in limits))
        manifest_path=folder/'experiment.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            configured_bound=manifest.get('intervention',{}).get('parameter_step_limit',{}).get('maximum')
            if configured_bound is not None:
                bounds.add(configured_bound)
            if 'last_saved_updates' in manifest:
                record['last_saved_updates']=manifest['last_saved_updates']
                record['intentional_stop']=manifest.get('status')
        reports.append(record)
        x=[r['step'] for r in selected]
        axes[0,0].plot(x,[r['cos_pos'] for r in selected],label=folder.name,linewidth=1,color=color)
        axes[0,1].plot(x,[r['fm'] for r in selected],label=folder.name,linewidth=1,color=color)
        axes[1,0].plot(x,[r['mag_ratio'] for r in selected],label=folder.name,linewidth=1,color=color)
        if telemetry:
            axes[1,1].plot([r['step'] for r in telemetry],[r['parameter_update_norm'] for r in telemetry],label=folder.name,linewidth=1,color=color)
    for bound in sorted(bounds):
        axes[1,1].axhline(bound,color='black',linestyle=':',linewidth=1,label=f'Configured update bound {bound:g}')
    for ax,title in zip(axes.flat,['Training hidden-shift alignment','Feature-matching loss',
                                  'Hidden-shift norm / teacher norm','Actual LoRA parameter-update norm']):
        ax.set_title(title);ax.set_xlabel('Completed updates');ax.grid(alpha=.2)
        ax.axvspan(830,900,color='gray',alpha=.08)
        ax.legend(fontsize=6)
    axes[0,1].set_yscale('log');axes[1,1].set_yscale('log')
    fig.suptitle('Stability experiments — training diagnostics do not establish listening preference')
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.png'),dpi=150);plt.close(fig)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(dict(runs=reports,
        note='Gray interval marks the historical late failure. Reported losses do not rank audio quality.'),indent=2,allow_nan=False)+'\n')
    for row in reports:
        print(row['run'],row['completed_updates'],'last30 cos',round(row['last_window_means']['cos_pos'],4),
              'magnitude',round(row['last_window_means']['mag_ratio'],4),'FM',round(row['last_window_means']['fm'],4),
              'FM cap rows',row.get('telemetry',{}).get('cap_active_rows'),
              'bounded parameter updates',row.get('parameter_step_limit',{}).get('active_updates'))


if __name__=='__main__':main()
