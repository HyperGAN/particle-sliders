"""Fixed rendered-audio heuristic for the user-authorized automatic campaign.

This is a suggestion rule, not a learned or validated overall quality metric.
All components and every sampled clip remain visible in the output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / '.cache/slider-quality/python'))
from slider_selection import features as F
from scripts.lm_score import WhisperBackend

# Judge descriptions are fixed independently of candidate weights and captions.
PAIRS = [
    ('gender', 'male', 'a feminine sounding lead singing voice', 'a masculine sounding lead singing voice'),
    ('energy', 'quiet', 'forceful drums and energetic singing', 'restrained drums and gentle quiet singing'),
    ('tempo', 'slow', 'a fast musical tempo', 'a slow musical tempo'),
    ('distortion', 'clean', 'distorted electric guitars and saturated sound', 'clean guitar tones and undistorted sound'),
    ('live', 'studio', 'a live room with audience sounds and microphone bleed', 'an isolated studio recording without audience or microphone bleed'),
    ('grit', 'smooth', 'a rough raspy gritty singing voice', 'a smooth clean singing voice'),
    ('joy', 'somber', 'joyful happy expressive singing', 'somber downcast expressive singing'),
    ('hurt', 'numb', 'anguished pained emotional singing', 'even emotionally restrained singing'),
    ('tender', 'fierce', 'gentle vulnerable singing with held vowels', 'forceful tense singing with clipped consonants'),
    ('yearn', 'settled', 'yearning reaching expressive vocal phrases', 'settled relaxed vocal phrases with a sense of resolution'),
    ('sexy', 'plain', 'intimate close microphone singing with breath and relaxed late phrasing', 'direct distant microphone singing with firm on beat phrasing'),
    ('triphop', 'pop', 'a slow dark groove with dusty breakbeats and rounded bass', 'bright polished pop with crisp drums and glossy synthesizers'),
    ('rhyme', 'prose', 'strong rhythmic emphasis on rhyming line endings', 'loose conversational phrasing without emphasized rhyming endings'),
    ('breath', None, 'breathy singing with audible inhales and mouth air', 'clear firm singing without audible breath'),
    ('rapslow', None, 'rhythmic spoken rap vocals', 'sustained melodic singing'),
]
CONCEPTS = {}
for positive, negative, a, b in PAIRS:
    CONCEPTS[positive] = ('a song with ' + a, 'a song with ' + b)
    if negative:
        CONCEPTS[negative] = ('a song with ' + b, 'a song with ' + a)

RULE = dict(
    version='render-heuristic-v2',
    weights=dict(concept=.4, enjoyment=.2, production=.2, lyrics=.2, artifacts=.2),
    scales=dict(concept=.05, enjoyment=.5, production=.5, lyrics=.25, artifacts=.02),
    component_clip=3.,
    reference='Each candidate compared with its exact prompt/seed slider-off render.',
    aggregation='Audio windows weighted by their share of covered time, then equal weight per prompt/seed. Overlapping tail windows do not count the ending twice.',
    lyric_component='Mean of transcript phrase accuracy and supplied-word recall; ASR errors remain possible.',
    perceptual_normalization='CLAP and aesthetics receive a separate float32 measurement copy at stereo RMS 0.1. Original audio is preserved and supplies levels, transcripts and noise fractions.',
    artifact_component='Penalty only: 14kHz-and-above power fraction exceeding both the slider-off and positive-caption reference by more than 0.002; scale 0.02. Intended reference brightness is allowed.',
    limitation='Heuristic coefficients were fixed before this scoring pass. This is not a validated quality metric or a guarantee of audible concept strength.',
)


def score_components(candidate, baseline, positive_reference=None):
    components = {k: max(-RULE['component_clip'], min(RULE['component_clip'],
                    (candidate[k] - baseline[k]) / RULE['scales'][k]))
                  for k in RULE['weights'] if k!='artifacts'}
    reference=positive_reference or baseline
    excess=max(0.,candidate.get('hf14k_fraction',0.)-max(baseline.get('hf14k_fraction',0.),
               reference.get('hf14k_fraction',0.))-.002)
    components['artifacts']=-min(RULE['component_clip'],excess/RULE['scales']['artifacts'])
    return sum(RULE['weights'][k] * v for k, v in components.items()), components


def coverage_weights(windows):
    boundaries=sorted({v for w in windows for v in (w['start_s'],w['end_s'])})
    shares=[0.]*len(windows)
    for left,right in zip(boundaries,boundaries[1:]):
        active=[i for i,w in enumerate(windows) if w['start_s']<=left and w['end_s']>=right]
        for i in active:shares[i]+=(right-left)/len(active)
    total=sum(shares)
    if not total:raise ValueError('No measured audio time')
    return [v/total for v in shares]


class Judge:
    def __init__(self):
        F.CONCEPTS = CONCEPTS
        self.audio = F.AudioMeasurer(ROOT/'analysis/gan_bcap/autonomous_audio_cache', 'cpu')
        self.asr = WhisperBackend(ROOT/'analysis/gan_bcap/asr_cpu_fp32_cache', device='cpu')

    def measure(self, audio, sheet, concept, duration):
        import numpy as np
        import soundfile as sf
        audio=Path(audio)
        original_sha=F.file_hash(audio)
        data,rate=sf.read(audio,dtype='float32',always_2d=True)
        if not data.size or not np.isfinite(data).all():raise ValueError('Empty or nonfinite original audio')
        original_rms=float(np.sqrt(np.mean(data.astype('float64')**2)))
        normalized=ROOT/'analysis/gan_bcap/normalized_measurement_audio'/f'{original_sha}-rms010-f32.wav'
        if not normalized.exists():
            normalized.parent.mkdir(parents=True,exist_ok=True)
            temp=normalized.with_suffix(f'.{os.getpid()}.tmp.wav')
            sf.write(temp,data*(.1/max(original_rms,1e-12)),rate,subtype='FLOAT')
            temp.replace(normalized)
        measured = self.audio.measure(normalized)
        asr = self.asr.measure(audio, duration)
        lyric = F.lyric_features(sheet, asr['text'])
        shares=coverage_weights(measured['windows'])
        def average(f):return sum(a*f(w) for a,w in zip(shares,measured['windows']))
        return dict(
            audio=str(audio.resolve()), sha256=original_sha,
            perceptual_measurement_sha256=measured['sha256'],
            concept=average(lambda w:w['concept'][concept]),
            enjoyment=average(lambda w:w['aesthetics']['CE']),
            production=average(lambda w:w['aesthetics']['PQ']),
            lyrics=(lyric['phrase_accuracy']+lyric['recall'])/2,
            lyric_diagnostics=lyric, transcript=asr['text'],
            rms=original_rms, duration=len(data)/rate,
            clipped_fraction=float(np.mean(np.abs(data)>=.999)),tail_ratio=asr['tail_ratio'],
            hf14k_fraction=F.fullband_features(audio)['hf14k_fraction'],
        )

    def folder(self, folder, concept):
        import yaml
        folder=Path(folder)
        spec=json.loads((folder/'render_spec.json').read_text())
        prompts=Path(spec['prompts'])
        if F.file_hash(prompts)!=spec['prompts_sha256']:
            raise ValueError('Prompt changed since rendering')
        rows=yaml.safe_load(prompts.read_text())
        if isinstance(rows,dict): rows=rows['rows']
        sheet=rows[spec['row']]['lyrics']
        first=Path(spec['checkpoints'][0]['path']).stem
        results=[]
        for seed in spec['seeds']:
            sub=folder/f'{first}-s{seed}'
            base=self.measure(sub/'01_slider_neutral_base_zero.wav', sheet, concept, spec['duration'])
            ref=self.measure(next(sub.glob('03_REF_prompt_*_no_slider.wav')), sheet, concept, spec['duration'])
            for checkpoint in spec['checkpoints']:
                weights=Path(checkpoint['path'])
                if F.file_hash(weights)!=checkpoint['sha256']:
                    raise ValueError('Checkpoint changed since rendering')
                clip=next((folder/f'{weights.stem}-s{seed}').glob('02_slider_*_plus1.wav'))
                candidate=self.measure(clip,sheet,concept,spec['duration'])
                score, components=score_components(candidate,base,ref)
                eligible=candidate['rms'] >= .02*max(base['rms'],1e-12)
                results.append(dict(checkpoint=checkpoint, seed=seed,
                    fixture=F.digest([spec['prompts_sha256'],spec['row'],seed,spec['duration']]),
                    candidate=candidate, baseline=base, positive_reference=ref,
                    heuristic_score=score, components=components,
                    eligible=eligible, exclusion=None if eligible else 'near-silent relative to its control'))
                print(weights.stem, seed, 'heuristic',round(score,4),flush=True)
        return results


def summarize(records):
    grouped={}
    for row in records:
        grouped.setdefault(row['checkpoint']['path'],[]).append(row)
    fixture_sets=[{r['fixture'] for r in rows} for rows in grouped.values()]
    if any(s!=fixture_sets[0] for s in fixture_sets):
        raise ValueError('Candidates do not have the same prompt/seed fixtures')
    if any(len(rows)!=len({r['fixture'] for r in rows}) for rows in grouped.values()):
        raise ValueError('Repeated prompt/seed would count a clip twice')
    summaries=[]
    for path,rows in grouped.items():
        summaries.append(dict(checkpoint=path, steps=rows[0]['checkpoint']['steps'],
            examples=len(rows), eligible=all(r['eligible'] for r in rows),
            heuristic_score=statistics.mean(r['heuristic_score'] for r in rows),
            minimum_clip_score=min(r['heuristic_score'] for r in rows),
            components={k:statistics.mean(r['components'][k] for r in rows) for k in RULE['weights']}))
    return sorted(summaries,key=lambda r:(not r['eligible'],-r['heuristic_score'],r['steps']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folders',type=Path,nargs='+',required=True)
    p.add_argument('--concept',choices=CONCEPTS,default='gender')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    judge=Judge();records=[]
    for folder in args.folders:
        records.extend(judge.folder(folder,args.concept))
        F.write_json(args.output,dict(status='measuring',rule=RULE,concept=args.concept,records=records))
    F.write_json(args.output,dict(status='complete',rule=RULE,concept=args.concept,
        source_sha256=F.file_hash(Path(__file__)),concepts=CONCEPTS,
        records=records,ranking=summarize(records)))


if __name__=='__main__':main()
