"""Add explicit failure rates and two diversity views to frozen audio scores.

Does not modify the original heuristic or its cache. All seeds and failures
are retained; an unvalidated judge cannot become a passed quality gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders.gan_v2.data import sha,digest
from conceptmod.textsliders.gan_v2.metrics import compare_diversity,clip_diagnostics,quality_decision


class Embeddings:
    def __init__(self,cache):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.model=self.processor=None

    def load(self):
        from transformers import ClapModel,ClapProcessor
        from slider_selection.features import CLAP
        self.processor=ClapProcessor.from_pretrained(CLAP[0],revision=CLAP[1],local_files_only=True)
        self.model=ClapModel.from_pretrained(CLAP[0],revision=CLAP[1],local_files_only=True).eval().requires_grad_(False)

    def measure(self,path,expected):
        import numpy as np
        import soundfile as sf
        import torch
        from torchaudio.functional import resample
        from slider_selection.features import CLAP,windows
        if sha(path)!=expected:raise ValueError('Audio changed after its quality measurement')
        provenance=dict(version=1,clap=CLAP,implementation=sha(__file__),normalization='mono RMS .1',device='cpu')
        cache=self.cache/f'{expected}-{digest(provenance)[:16]}.json'
        if cache.exists():return json.loads(cache.read_text())
        audio,rate=sf.read(path,dtype='float32',always_2d=True)
        if not audio.size or not np.isfinite(audio).all():raise ValueError('Invalid audio')
        mono=torch.from_numpy(audio.mean(1))
        mono=mono*(.1/max(float(mono.double().square().mean().sqrt()),1e-12))
        if self.model is None:self.load()
        features=[];durations=[]
        with torch.inference_mode():
            # Nonoverlapping windows; the last short window has proportional
            # weight. No stochastic long-clip cropping and no discarded tail.
            for start in range(0,len(mono),10*rate):
                segment=mono[start:start+10*rate]
                if not len(segment):continue
                wave=resample(segment,rate,48000).numpy()
                inputs=self.processor.feature_extractor(wave,sampling_rate=48000,return_tensors='pt')
                value=self.model.get_audio_features(**inputs)
                value=value.pooler_output if hasattr(value,'pooler_output') else value
                features.append(value[0].float().numpy());durations.append(len(segment))
            embedding=np.average(features,axis=0,weights=durations)
            # An independent, fixed DSP view emphasizes pitch-class balance
            # and onset timing. It remains a proxy, not a voice-isolated view.
            signal=resample(mono,rate,12000)
            if len(signal)<1024:signal=torch.nn.functional.pad(signal,(0,1024-len(signal)))
            power=torch.stft(signal,n_fft=1024,hop_length=256,window=torch.hann_window(1024),return_complex=True).abs().square()
            chroma=torch.zeros(12,power.shape[1]);freq=torch.fft.rfftfreq(1024,1/12000)
            valid=(freq>=55)&(freq<=2000)
            pitch=(69+12*torch.log2(freq[valid]/440)).round().long()%12
            chroma.index_add_(0,pitch,power[valid])
            chroma=chroma/(chroma.sum(0,keepdim=True)+1e-12)
            flux=(power.sqrt()[:,1:]-power.sqrt()[:,:-1]).clamp_min(0).mean(0)
            flux=flux-flux.mean()
            lags=range(5,65,3)
            autocorrelation=torch.stack([(flux[:-lag]*flux[lag:]).mean()/(flux.square().mean()+1e-12)
                                          if len(flux)>lag else torch.tensor(0.) for lag in lags])
            rhythm=torch.cat([chroma.mean(1),chroma.std(1,unbiased=False),autocorrelation]).numpy()
        result=dict(audio_sha256=expected,provenance=provenance,clap=embedding.tolist(),rhythm_chroma=rhythm.tolist())
        temporary=cache.with_suffix('.tmp');temporary.write_text(json.dumps(result,allow_nan=False));temporary.replace(cache)
        return result


def evaluate(reports,output,cache,*,minimum_prompts=8):
    from collections import defaultdict
    from analysis.gan_bcap.autonomous_audio import RULE
    extractor=Embeddings(cache)
    by_candidate=defaultdict(list)
    seen=set()
    for path in reports:
        report=json.loads(Path(path).read_text())
        if report.get('status')!='complete' or report['rule']!=RULE:raise ValueError('Expected complete fixed-rule audio scores')
        for row in report['records']:
            identity=(row['checkpoint']['sha256'],row['fixture'])
            if identity in seen:raise ValueError('A prompt/seed fixture was counted twice')
            seen.add(identity)
            spec_path=Path(row['baseline']['audio']).parent.parent/'render_spec.json'
            spec=json.loads(spec_path.read_text())
            if sha(spec['prompts'])!=spec['prompts_sha256']:raise ValueError('Evaluation prompt changed')
            # Same prompt remains one group across files, EMA topology batches,
            # and generation seeds. Never use score-file count as sample size.
            prompt=digest([spec['prompts_sha256'],spec['row'],spec['duration']])
            measured={key:extractor.measure(row[key]['audio'],row[key]['sha256'])
                      for key in ('candidate','baseline','positive_reference')}
            by_candidate[row['checkpoint']['path']].append(dict(prompt=prompt,seed=row['seed'],
                score=row['heuristic_score'],diagnostics=clip_diagnostics(row['candidate'],row['baseline'],row['positive_reference']),
                embeddings=measured,fixture=row['fixture']))
    result=dict(version=1,status='measuring',source_sha256=sha(__file__),scores=[dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in reports],candidates={})
    # Candidate selection requires the same observed fixture set.
    fixture_sets=[{row['fixture'] for row in rows} for rows in by_candidate.values()]
    if any(s!=fixture_sets[0] for s in fixture_sets):raise ValueError('Candidates have unmatched evaluation fixtures')
    for checkpoint,rows in by_candidate.items():
        groups=defaultdict(list)
        for row in rows:groups[row['prompt']].append(row)
        diversity={}
        for prompt,examples in groups.items():
            examples.sort(key=lambda r:r['seed'])
            if len({r['seed'] for r in examples})!=len(examples):raise ValueError('Duplicate sampling seed')
            views={name:compare_diversity(*[[r['embeddings'][side][name] for r in examples]
                           for side in ('candidate','baseline','positive_reference')]) for name in ('clap','rhythm_chroma')}
            statuses=[v['status'] for v in views.values()]
            status=('insufficient_evidence' if 'insufficient_evidence' in statuses else
                    'collapse_suspected' if all(s=='collapse_suspected' for s in statuses) else 'no_collapse_detected')
            diversity[prompt]=dict(status=status,views=views,
                view_disagreement=len(set(statuses))>1,
                scope='Consensus gross-collapse screen across semantic and rhythm/chroma views; a single-view alarm remains visible.')
        quality=quality_decision(rows,diversity,minimum_prompts=minimum_prompts)
        import numpy as np
        scores=np.asarray([r['score'] for r in rows])
        result['candidates'][checkpoint]=dict(quality=quality,diversity=diversity,
            mean_score=float(scores.mean()),score_quantile_10=float(np.quantile(scores,.1)),worst_score=float(scores.min()),
            clips=[{k:v for k,v in r.items() if k!='embeddings'} for r in rows])
        Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    result['status']='complete'
    temporary=Path(output).with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(output)
    return result


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'analysis/gan_bcap/v2_20260905/audio_embeddings')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    evaluate(args.scores,args.output,args.cache)
