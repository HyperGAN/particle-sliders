"""A deterministic predictor facing two valid targets: mean versus one mode.

This is an analytic counterexample, not a measurement of the music model.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def kernel(x,y,widths):
    r2=(x[:,None]-y[None,:])**2
    return sum(np.exp(-r2/(2*b*b)) for b in widths)/len(widths)


def main():
    x=np.linspace(-2,2,8001)
    targets=np.array([-1.,1.])
    widths=[.03,.1,.3,1.,3.]
    mse=((x[:,None]-targets[None,:])**2).mean(1)
    broad=2*(1-kernel(x,targets,[3.]).mean(1))
    current=2*(1-kernel(x,targets,widths).mean(1))
    def mmd(fake):
        return float(kernel(fake,fake,widths).mean()+kernel(targets,targets,widths).mean()-2*kernel(fake,targets,widths).mean())
    output=dict(description='Two equally likely targets for the SAME student input. Single-output regression cannot represent both modes.',
        mse_minimizer=float(x[mse.argmin()]),broad_kernel_minimizer=float(x[broad.argmin()]),
        current_kernel_single_output_minimizers=x[np.isclose(current,current.min(),atol=1e-8,rtol=0)].tolist(),
        full_distribution_mmd_two_correct_particles=mmd(targets),
        full_distribution_mmd_two_mean_particles=mmd(np.zeros(2)),
        conditional_paired_loss_with_distinct_inputs_and_correct_targets=0.,
        interpretation='Broad-kernel regression can average; narrow-kernel regression can select one mode. Full MMD with multiple fake and real samples retains the within-fake term. Distinct observable inputs with deterministic teachers are a different case.')
    assert abs(output['mse_minimizer'])<1e-8 and abs(output['broad_kernel_minimizer'])<1e-8
    assert all(abs(v)>.9 for v in output['current_kernel_single_output_minimizers'])
    assert abs(mmd(targets))<1e-12 and mmd(np.zeros(2))>.1
    (HERE/'averaging-probe.json').write_text(json.dumps(output,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12.8,4))
    for ax,values,title in zip(axes,[mse,broad,current],['MSE: chooses the mean','Broad MMD kernel: also chooses the mean','Current kernel mixture: chooses one mode']):
        ax.plot(x,values,color='#176ca4',lw=2)
        for value in targets:ax.axvline(value,color='#a3afb8',ls=':',lw=1)
        indices=np.flatnonzero(np.isclose(values,values.min(),atol=1e-8,rtol=0))
        ax.scatter(x[indices],values[indices],s=25,color='#b84628',zorder=5)
        ax.set(title=title,xlabel='One deterministic prediction',ylabel='Expected paired loss')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two valid targets at −1 and +1 for the same input',fontsize=16)
    fig.text(.05,.01,'Analytic example, not a music result. One deterministic output cannot cover both choices; matching a distribution requires a distribution of outputs.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94])
    fig.savefig(HERE/'averaging-probe.png',dpi=170);fig.savefig(HERE/'averaging-probe.svg');plt.close(fig)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
