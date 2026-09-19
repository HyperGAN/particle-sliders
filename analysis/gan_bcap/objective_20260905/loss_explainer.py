"""Plot the recorded MMD objective, error and actual optimizer steps.

No model loading or new training. All run curves come from preserved logs;
the kernel-response figure is an explicitly labeled analytic illustration.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'explainer'
RUN=ROOT/'models/conditional-mmd-cfg-research-20260905'
BLUE='#176ca4'
PURPLE='#8152a4'
RED='#b73836'


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((RUN/'manifest.json').read_text())
    rows=[json.loads(s) for s in (RUN/'train.jsonl').read_text().splitlines()]
    evaluations=[json.loads(p.read_text()) for p in sorted(RUN.glob('evaluation-*.json'))]
    origin=manifest['source_step']
    x=np.arange(len(rows)+1)
    loss=np.array([rows[0]['loss_before']]+[r['line_search']['loss'] for r in rows])
    ex=np.array([e['step']-origin for e in evaluations])
    def metric(group,key):
        return np.array([statistics.mean(r[key] for r in e[group]) for e in evaluations])
    heldout=metric('heldout','loss')
    accepted=np.array([r['line_search']['accepted'] for r in rows])
    rejected_x=x[1:][~accepted]
    steps=np.array([r['parameter_step'] for r in rows])
    assert np.isfinite(loss).all() and np.all(np.diff(loss)<=0)
    assert np.allclose(loss[ex],metric('train','loss'),atol=1e-8,rtol=0)
    assert all(abs(r['loss_before']-loss[i])<1e-8 for i,r in enumerate(rows))
    assert np.all(steps[~accepted]==0)
    stats=dict(start_loss=float(loss[0]),end_loss=float(loss[-1]),
        loss_reduction_percent=float(100*(1-loss[-1]/loss[0])),
        heldout_start=float(heldout[0]),heldout_end=float(heldout[-1]),
        accepted=int(accepted.sum()),rejected=int((~accepted).sum()),
        shortened=sum(r['line_search']['accepted'] and r['line_search']['fraction']<1 for r in rows),
        first_rejection_attempt=int(rejected_x[0]),maximum_loss_increase=float(np.diff(loss).max()),
        reduction_in_first_30_fraction=float((loss[0]-loss[30])/(loss[0]-loss[-1])),
        last_30_loss_reduction_percent=float(100*(1-loss[-1]/loss[90])),
        train_relative_error=metric('train','relative_error').tolist(),
        heldout_relative_error=metric('heldout','relative_error').tolist())
    data=dict(summary=stats,origin=origin,loss=loss.tolist(),rows=rows,
        evaluations=evaluations,scale=manifest['scale'],bandwidths=manifest['kernel_bandwidths'])
    inputs=[RUN/'manifest.json',RUN/'train.jsonl',*sorted(RUN.glob('evaluation-*.json')),Path(__file__)]
    inputs += [HERE/'runs'/name/'progress.jsonl' for name in ('bcap-seed7','mmd-batch1024-extended-seed7')]
    data['sources']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (OUT/'loss-steps.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['attempt','source_step','loss_before','loss_after','accepted','fraction_of_adam_proposal','parameter_step_l2','gradient_l2'])
        writer.writerow([0,origin,loss[0],loss[0],'initial',0,0,''])
        for i,r in enumerate(rows,1):
            s=r['line_search']
            writer.writerow([i,r['step'],r['loss_before'],s['loss'],s['accepted'],s['fraction'],r['parameter_step'],r['gradient_norm']])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
        'axes.labelcolor':'#283542','text.color':'#192833','axes.edgecolor':'#9eacb6',
        'grid.color':'#dde3e8','grid.linewidth':.7,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.6))
    fig.suptitle('Fixed MMD on the real LoRA',fontsize=20,x=.06,ha='left',y=.985)
    fig.text(.06,.945,'120 proposal attempts from source checkpoint 600 · fixed training data · both guidance branches',fontsize=11,color='#526574')
    ax=axes[0,0]
    ax.plot(x,loss,color=BLUE,lw=2.7,label='Train: measured every attempt')
    ax.plot(ex,heldout,'o--',color=PURPLE,lw=1.8,ms=5,label='Two heldout prompts: five evaluations')
    ax.scatter(rejected_x,loss[rejected_x],marker='x',color=RED,s=24,zorder=4,label='Rejected proposal: weights restored')
    ax.annotate(f'{loss[0]:.4f}',(0,loss[0]),xytext=(5,9),textcoords='offset points',color=BLUE,weight='bold')
    ax.annotate(f'{loss[-1]:.4f}',(120,loss[-1]),xytext=(-5,-20),ha='right',textcoords='offset points',color=BLUE,weight='bold')
    ax.set(title='A. The monotonically non-increasing number',ylabel='Conditional MMD loss (0 = exact match)',ylim=(0,1.36))
    ax.legend(loc='lower left',fontsize=8.4,frameon=False)
    ax.text(.97,.43,'24.5% lower training loss',transform=ax.transAxes,ha='right',color=BLUE,fontsize=11,weight='bold')
    ax=axes[0,1]
    ax.plot(x,loss,color=BLUE,lw=2.2)
    ax.scatter(rejected_x,loss[rejected_x],marker='x',color=RED,s=34,zorder=4)
    ax.set(title='B. Late-run zoom: many flat steps',xlim=(80,121),ylim=(loss[-1]-.0008,loss[80]+.001),ylabel='The same training loss; expanded vertical scale')
    ax.text(.97,.93,f"First rejection: attempt {stats['first_rejection_attempt']}\nLast 30 attempts: only {stats['last_30_loss_reduction_percent']:.2f}% lower loss",transform=ax.transAxes,ha='right',va='top',fontsize=10)
    ax.ticklabel_format(axis='y',useOffset=False,style='plain')
    ax=axes[1,0]
    for group,color,label in [('train',BLUE,'Train'),('heldout',PURPLE,'Heldout')]:
        vals=metric(group,'relative_error')
        ax.plot(ex,vals,'o-',color=color,lw=2,label=label)
        ax.annotate(f'{vals[-1]:.4f}',(120,vals[-1]),xytext=(-4,8 if group=='heldout' else -19),ha='right',textcoords='offset points',color=color,weight='bold')
    ax.set(title='C. A different number: direct hidden-state error',ylabel='Mean relative error to teacher delta',ylim=(0,1.05))
    ax.axhline(1.,color='#889aa8',lw=1,ls=':',zorder=0)
    ax.text(.97,.945,'No slider = 1.0',transform=ax.transAxes,ha='right',fontsize=8.7,color='#526574')
    ax.legend(loc='lower left',frameon=False,fontsize=9)
    ax.text(.04,.42,'Measured separately; its decrease is not enforced.',transform=ax.transAxes,fontsize=9,color='#526574')
    ax=axes[1,1]
    ax.plot(x[1:],steps,color=BLUE,lw=1.1,alpha=.6)
    ax.scatter(x[1:][accepted],steps[accepted],color=BLUE,s=15,label='Accepted step')
    ax.scatter(rejected_x,steps[~accepted],color=RED,marker='x',s=32,label='Rejected: zero step')
    ax.set(title='D. Actual movement of the LoRA parameters',ylabel='L2 norm of the parameter change',ylim=(-.075,max(steps)*1.15))
    ax.text(.96,.91,'99 accepted · 21 rejected\n89 accepted steps were shortened',ha='right',va='top',transform=ax.transAxes,fontsize=10)
    ax.legend(loc='center right',frameon=False,fontsize=9)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.8)
        ax.set_xlabel('MMD proposal attempt (0 = original update 600)')
        if ax is not axes[0,1]:ax.set_xlim(0,123)
    fig.text(.06,.025,'Raw recorded values; no curve smoothing. Heldout lines connect five samples. Parameter distance does not measure audio change.',fontsize=9,color='#526574')
    fig.tight_layout(rect=[.02,.05,1,.925],h_pad=2.5,w_pad=3)
    for suffix in ('png','svg','pdf'):fig.savefig(OUT/f'loss-curves.{suffix}',dpi=180)
    plt.close(fig)

    r=np.geomspace(1e-4,100,1600)
    widths=np.array(manifest['kernel_bandwidths'])
    components=-2*np.expm1(-r[None,:]**2/(2*widths[:,None]**2))
    gradient=2*r[None,:]/widths[:,None]**2*np.exp(-r[None,:]**2/(2*widths[:,None]**2))
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7))
    palette=['#246da2','#289f92','#7c9d41','#c1882e','#9660a2']
    for i,(width,color) in enumerate(zip(widths,palette)):
        axes[0].plot(r,components[i],color=color,lw=1.3,alpha=.8,label=f'b = {width:g}')
        axes[1].plot(r,gradient[i],color=color,lw=1.2,alpha=.65)
    axes[0].plot(r,components.mean(0),color='#17242f',lw=3,label='Actual loss: average of five')
    axes[1].plot(r,gradient.mean(0),color='#17242f',lw=3,label='Actual gradient magnitude')
    axes[0].set(title='Each distance scale asks: how close is this pair?',ylabel='Paired loss contribution',ylim=(-.04,2.06))
    axes[1].set(title='The pull is strongest around each scale',ylabel='Derivative of paired loss with respect to r',ylim=(-.5,42))
    axes[0].legend(frameon=False,fontsize=8.5,loc='upper left')
    axes[1].legend(frameon=False,fontsize=9,loc='upper right')
    for ax in axes:
        ax.set_xscale('log');ax.set_xlim(.0001,100)
        ax.set_xlabel('r = hidden RMS mismatch / frozen teacher-delta scale')
        ax.grid(axis='y',alpha=.7)
    fig.text(.05,.01,'Analytic illustration, not a training trace. Exact match: loss = 0, gradient = 0. Very distant pairs: loss approaches 2, gradient approaches 0.',fontsize=9,color='#526574')
    fig.tight_layout(rect=[0,.07,1,1])
    for suffix in ('png','svg','pdf'):fig.savefig(OUT/f'kernel-intuition.{suffix}',dpi=180)
    plt.close(fig)
    gaussian_traces()
    write_html(data)
    print(json.dumps(stats,indent=2))


def gaussian_traces():
    cap=[json.loads(s) for s in (HERE/'runs/bcap-seed7/progress.jsonl').read_text().splitlines()]
    mmd=[json.loads(s) for s in (HERE/'runs/mmd-batch1024-extended-seed7/progress.jsonl').read_text().splitlines()]
    cap=[r for r in cap if r['losses']]
    mmd=[r for r in mmd if r['losses']]
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    x=[r['step'] for r in cap]
    axes[0].plot(x,[r['losses']['g']+r['losses']['vicreg'] for r in cap],'o-',color=BLUE,label='G + movable-prior objective')
    axes[0].plot(x,[r['losses']['d'] for r in cap],'o-',color=PURPLE,label='D loss including cap')
    axes[0].axhline(np.log(2),color='#8e9da8',ls=':',lw=1,label='log(2): logistic reference only')
    axes[0].set(title='Original Gaussian cap control',ylabel='Logged minibatch objective',ylim=(.55,.9))
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].plot([r['step'] for r in mmd],[r['losses']['g'] for r in mmd],'o--',color=BLUE)
    axes[1].axhline(0,color='#8e9da8',lw=1)
    axes[1].set(title='Gaussian fixed MMD: four logged minibatches',ylabel='Unbiased minibatch MMD estimate')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[1].text(.98,.96,'Can be negative from sample noise.\nNot the deterministic paired music loss.',transform=axes[1].transAxes,ha='right',va='top',fontsize=9,color='#526574')
    for ax in axes:ax.set_xlabel('Gaussian optimizer update');ax.grid(axis='y',alpha=.7)
    fig.text(.045,.01,'Sparse recorded samples, not dense loss histories. Values across these panels and the music trial use different objectives or estimators.',fontsize=9,color='#526574')
    fig.tight_layout(rect=[0,.07,1,1])
    for suffix in ('png','svg','pdf'):fig.savefig(OUT/f'gaussian-losses.{suffix}',dpi=180)
    plt.close(fig)


def write_html(data):
    template='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What the custom MMD loss measures</title>
<style>body{font:16px system-ui;color:#1c2e3a;background:#f6f8fa;max-width:1160px;margin:32px auto;padding:0 22px}h1{font-size:32px}h2{font-size:22px}p{max-width:930px;line-height:1.65}a{color:#176ca4}.card{background:white;border:1px solid #d7e0e7;border-radius:12px;padding:22px;margin:22px 0}img,svg{width:100%;height:auto}input[type=range]{width:100%}.readout{font-variant-numeric:tabular-nums;line-height:1.8}.bars{display:grid;grid-template-columns:80px 1fr 75px;gap:9px;align-items:center;margin-top:14px}.bar{height:12px;background:#e9eff3;border-radius:4px;overflow:hidden}.fill{height:100%;background:#176ca4}.muted{color:#576e7d;font-size:14px}button{background:white;border:1px solid #a1b3bf;border-radius:5px;padding:7px 14px;color:#193746;cursor:pointer}code{font-size:14px}</style>
<h1>What the custom MMD loss measures</h1>
<p>This is the fixed-kernel, both-guidance-branches LoRA trial from checkpoint 600. The blue training loss is the monotonically non-increasing number: 1.1608 → 0.8767. The optimizer explicitly rejects increases. It is a hidden-state matching score, not a probability or an audio-quality rating.</p>
<div class="card"><h2>Inspect an actual training attempt</h2><svg id="trace" viewBox="0 0 1000 290" role="img" aria-label="Recorded training loss with movable cursor"></svg><label for="attempt">Proposal attempt</label><input id="attempt" type="range" min="0" max="120" value="120"><div id="steptext" class="readout" aria-live="polite"></div><p class="muted">Every point comes from the saved log. No smoothing. Rejections keep both weights and optimizer moments at their previous values.</p></div>
<div class="card"><h2>Five fixed distance scales</h2><p>At each prompt/history/position/branch, compare the slider hidden vector with its paired teacher vector. Compute normalized RMS distance r. Each scale b contributes <code>2 × (1 − exp(−r² / (2b²)))</code>; average the five contributions. The run averages this result over all paired conditions.</p><label for="distance">Move one hypothetical pair closer or farther apart (logarithmic slider)</label><input id="distance" type="range" min="-4" max="2" step="0.01" value="-0.5"><button id="match">Exact match: r = 0</button><div id="distanceText" class="readout" aria-live="polite"></div><div id="bars" class="bars"></div><p class="muted">This illustration represents one pair. The actual batch contains many different distances, so its average loss cannot be inverted to recover one RMS error. A loss of 0.8767 does not mean 87.67% wrong.</p></div>
<div class="card"><h2>The recorded curves</h2><img src="loss-curves.png" alt="Training and heldout MMD, late-run loss zoom, direct hidden error and actual parameter steps"><a href="loss-curves.svg">Vector graphic</a> · <a href="loss-curves.pdf">PDF</a> · <a href="loss-steps.csv">Raw CSV</a></div>
<div class="card"><h2>How the distance scales behave</h2><img src="kernel-intuition.png" alt="Analytic loss and gradient against normalized hidden mismatch"><p>Small scales respond to fine mismatches; broad scales continue responding to larger mismatches. Near an exact match the loss behaves locally like squared error. Far beyond every scale, both the loss and its derivative flatten.</p></div>
<p>In the Gaussian MMD experiment, entire random clouds are compared: matching real/fake points is rewarded and excess fake/fake similarity is penalized. In this deterministic music trial there is one output per condition, so the within-condition self terms are constants and the same MMD becomes the paired regression curve above. Different prompts and time positions are not treated as interchangeable samples.</p>
<p>Training uses four prompt rows and 250 cached history frames, with two additional prompts for diagnostics. New generated histories are not covered by this finite training set. Neither monotonically decreasing training loss nor these heldout hidden scores guarantees better free-running audio.</p>
<p><a href="gaussian-losses.png">Recorded Gaussian loss samples</a> · <a href="../README.md">Full audit</a> · <a href="../audio-results.md">Audio diagnostics</a> · <a href="data.json">Values and source hashes</a> · <a href="https://jmlr.org/papers/v13/gretton12a.html">MMD reference</a></p>
<script id="data" type="application/json">__DATA__</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent);
const svg=document.getElementById('trace');
const sx=x=>62+x/120*886, sy=y=>242-(y-.85)/.4*206;
const path=d.loss.map((y,i)=>(i?'L':'M')+sx(i).toFixed(2)+','+sy(y).toFixed(2)).join(' ');
let grid='';for(const y of [.9,1,1.1,1.2])grid+=`<line x1="62" x2="948" y1="${sy(y)}" y2="${sy(y)}" stroke="#dbe4eb"/><text x="48" y="${sy(y)+5}" text-anchor="end" font-size="14" fill="#526574">${y.toFixed(1)}</text>`;
for(const x of [0,30,60,90,120])grid+=`<text x="${sx(x)}" y="270" text-anchor="middle" font-size="14" fill="#526574">${x}</text>`;
svg.innerHTML=grid+`<text x="62" y="20" font-size="15" fill="#526574">Training MMD loss (vertical axis zoomed)</text><path d="${path}" fill="none" stroke="#176ca4" stroke-width="3"/><line id="cursor" y1="33" y2="242" stroke="#8d4d89" stroke-dasharray="4 4"/><circle id="dot" r="5" fill="#8d4d89"/>`;
function step(){const i=+document.getElementById('attempt').value,y=d.loss[i],r=i?d.rows[i-1]:null;document.getElementById('cursor').setAttribute('x1',sx(i));document.getElementById('cursor').setAttribute('x2',sx(i));document.getElementById('dot').setAttribute('cx',sx(i));document.getElementById('dot').setAttribute('cy',sy(y));document.getElementById('steptext').innerHTML=`<b>Attempt ${i} / source update ${d.origin+i}</b> · training loss <b>${y.toFixed(8)}</b>`+(r?`<br>${r.line_search.accepted?'Accepted':'Rejected; restored previous state'} · retained ${(100*r.line_search.fraction).toFixed(3)}% of Adam proposal · actual parameter step ${r.parameter_step.toFixed(6)}<br>Loss before ${r.loss_before.toFixed(8)} → after ${y.toFixed(8)}`:'<br>Starting weights before any MMD update.');}
function distance(r){let total=0,bars='';for(const b of d.bandwidths){const value=-2*Math.expm1(-r*r/(2*b*b));total+=value;bars+=`<div>b = ${b}</div><div class="bar"><div class="fill" style="width:${value/2*100}%"></div></div><div>${value.toFixed(4)}</div>`;}document.getElementById('bars').innerHTML=bars;document.getElementById('distanceText').innerHTML=`Normalized mismatch r = <b>${r.toFixed(5)}</b> · raw hidden RMS mismatch ${(r*d.scale).toFixed(5)}<br>Average paired loss = <b>${(total/d.bandwidths.length).toFixed(6)}</b> (0 = exact match; upper limit 2)`;}
document.getElementById('attempt').addEventListener('input',step);document.getElementById('distance').addEventListener('input',e=>distance(10**(+e.target.value)));document.getElementById('match').addEventListener('click',()=>distance(0));step();distance(10**(-.5));
</script>'''
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(data,allow_nan=False)))


if __name__=='__main__':main()
