"""Check mode probabilities separately from mode coverage and core fidelity."""
from pathlib import Path
import json
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    results=[]
    for folder in sorted((HERE/'runs').iterdir()):
        if not (folder/'completion.json').exists():continue
        values={}
        for name in ('ema','live'):
            samples=torch.load(folder/f'{name}_samples.pt',weights_only=True).numpy()
            # Exact nearest center for the rectangular Cartesian grid;
            # include tail samples instead of silently conditioning on HQ.
            nearest=np.clip(np.rint(samples+4.5),0,9).astype(int)
            counts=np.bincount(nearest[:,0]*10+nearest[:,1],minlength=100)
            probabilities=counts/counts.sum()
            values[name]=dict(mode_total_variation=float(.5*np.abs(probabilities-.01).sum()),
                minimum_assigned=int(counts.min()),maximum_assigned=int(counts.max()),counts=counts.tolist())
        results.append(dict(run=folder.name,**values))
    counts=np.random.default_rng(20260905).multinomial(20000,np.full(100,.01),size=1000)
    noise=.5*np.abs(counts/20000-.01).sum(1)
    result=dict(runs=results,uniform_sampling_reference=dict(draws=20000,replicates=1000,
        mean=float(noise.mean()),q05=float(np.quantile(noise,.05)),q95=float(np.quantile(noise,.95))),
        interpretation='Nearest-mode occupancy over all saved samples, including tails. Sampling reference is diagnostic, not a checkpoint-selection threshold.')
    (HERE/'mode-occupancy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['uniform_sampling_reference']))


if __name__=='__main__':main()
