"""Validate the rendered comparisons and expose frozen transcript diagnostics."""
from pathlib import Path
import hashlib
import html
import json
import os

import numpy as np
import soundfile as sf

from slider_selection.features import lyric_features

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GALLERY=ROOT/'eval/listen/gan-objective-20260905'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows=[]
    for path in [HERE/'audio-asr.json',HERE/'audio-mmd-asr.json']:
        if not path.exists():continue
        blob=json.loads(path.read_text())
        for row in blob['records']:
            if sha(Path(row['audio']))!=row['sha256']:raise ValueError('Audio changed after transcription')
            row=dict(row,lyrics=lyric_features(blob['lyrics'],row['transcript']))
            if not any(r['sha256']==row['sha256'] and r['role']==row['role'] and r['checkpoint']==row['checkpoint'] for r in rows):rows.append(row)
    validation=[]
    for folder in [GALLERY,ROOT/'eval/listen/gan-objective-mmd-20260905']:
        if not (folder/'README.md').exists():continue
        spec=json.loads((folder/'render_spec.json').read_text())
        if (folder/'reuse.json').exists():
            for entry in json.loads((folder/'reuse.json').read_text())['copies']:
                if sha(Path(entry['source']))!=entry['sha256'] or sha(Path(entry['destination']))!=entry['sha256']:
                    raise ValueError('Reused control or source audio changed')
        controls={}
        for ck in spec['checkpoints']:
            if sha(Path(ck['path']))!=ck['sha256']:raise ValueError('Checkpoint changed after render')
            for seed in spec['seeds']:
                leaf=folder/f"{Path(ck['path']).stem}-s{seed}"
                meta=json.loads((leaf/'checkpoint.json').read_text())
                if meta['sha256']!=ck['sha256']:raise ValueError('Render sidecar/checkpoint mismatch')
                for audio in sorted(leaf.glob('*.wav')):
                    data,sr=sf.read(audio,dtype='float32',always_2d=True)
                    if not data.size or not np.isfinite(data).all():raise ValueError(f'Invalid audio {audio}')
                    digest=sha(audio)
                    if audio.name!='02_slider_Female_plus1.wav':
                        key=(seed,audio.name)
                        if key in controls and controls[key]!=digest:raise ValueError('Unmatched control audio')
                        controls[key]=digest
                    validation.append(dict(path=str(audio),sha256=digest,seconds=len(data)/sr,
                        rms=float(np.sqrt(np.mean(data.astype(np.float64)**2))),peak=float(np.abs(data).max())))
    (HERE/'audio-validation.json').write_text(json.dumps(dict(files=validation,asr=rows),indent=2,allow_nan=False)+'\n')
    names={
        'smoke-steps600-s7-20260904_last':'Reference · 600',
        'conditional-energy-research-20260905_step720':'Energy · conditional branch',
        'conditional-energy-cfg-research-20260905-attempt2_step720':'Energy · both branches',
        'conditional-mmd-cfg-research-20260905_step720':'Fixed MMD · both branches'}
    metrics_path=HERE/'audio-frozen-metrics.json'
    metrics={r['sha256']:r for r in json.loads(metrics_path.read_text())['records']} if metrics_path.exists() else {}
    table=['# Matched audio diagnostics','',
        'Every first clip is retained. These fixed transcription and audio-model scores are diagnostics on two development seeds; no thresholds were fitted here. '
        'CLAP margin is feminine-voice descriptor minus masculine-voice descriptor; it is not a probability. PQ is the frozen production-quality estimate.',
        '', '| Candidate / control | Seed | ASR precision | Phrase match | Sheet coverage | Voice margin | PQ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(rows,key=lambda r:(r['seed'],r['role'],(r['checkpoint'] or {}).get('path',''))):
        stem=Path((row['checkpoint'] or {}).get('path','')).stem
        label=names.get(stem,stem) if row['role']=='slider_plus1' else row['role']
        values=row['lyrics'];measure=metrics.get(row['sha256'])
        audio_values=f"{measure['female_margin']:.4f} | {measure['production_quality']:.3f}" if measure else 'pending | pending'
        table.append(f"| {label} | {row['seed']} | {values['precision']:.3f} | {values['phrase_accuracy']:.3f} | {values['recall']:.3f} | {audio_values} |")
    table+=['','The 20-second cap does not test natural endings or long-form stability. '
        'A correct transcript does not certify the intended vocal change, and a higher voice margin does not certify lyric fidelity.',
        '', '[Listen to all candidates](../../../eval/listen/gan-objective-20260905/index.html)', '']
    (HERE/'audio-results.md').write_text('\n'.join(table))
    def player(path):
        return '<audio controls preload="none" src="'+html.escape(os.path.relpath(path,GALLERY))+'"></audio>'
    output=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Slider objective comparison</title>',
        '<style>body{background:#111820;color:#e7edf3;font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}a{color:#97d5ff}table{border-collapse:collapse;width:100%}td,th{padding:18px 12px;border-bottom:1px solid #34414e;text-align:left}audio{width:100%;min-width:220px}small{display:block;color:#bdcad6;max-width:340px;margin-top:9px}h1{font-size:30px}p{line-height:1.6}summary{cursor:pointer}</style>',
        '<h1>Slider objective comparison</h1><p>Same prompt and lyrics, 20 seconds, scale +1. Compare vocal character, lyric order, extra words and artifacts. Every first render is retained.</p>',
        '<p><a href="../../../analysis/gan_bcap/objective_20260905/README.md">Technical audit</a> · <a href="../../../analysis/gan_bcap/objective_20260905/results.md">All experiments</a> · <a href="../../../analysis/gan_bcap/objective_20260905/audio-results.md">Audio diagnostics</a></p>',
        '<table><tr><th>Checkpoint</th><th>Seed 7</th><th>Seed 23</th></tr>']
    for stem,name in names.items():
        folder=ROOT/'eval/listen/gan-objective-mmd-20260905' if stem.startswith('conditional-mmd') else GALLERY
        paths=[folder/f'{stem}-s{seed}'/'02_slider_Female_plus1.wav' for seed in (7,23)]
        if not all(p.exists() for p in paths):continue
        output.append('<tr><th>'+html.escape(name)+'</th>')
        for seed,path in zip((7,23),paths):
            matching=[r for r in rows if r['role']=='slider_plus1' and r['seed']==seed and Path(r['audio']).parent.name==path.parent.name]
            output.append('<td>'+player(path))
            if matching:
                r=matching[0];v=r['lyrics']
                output.append(f"<small>ASR word match {v['precision']:.0%} · phrase match {v['phrase_accuracy']:.0%} · sheet coverage {v['recall']:.0%}</small><details><summary>Transcript</summary><small>{html.escape(r['transcript'])}</small></details>")
            output.append('</td>')
        output.append('</tr>')
    first='smoke-steps600-s7-20260904_last'
    for name,filename in [('Slider off','01_slider_neutral_base_zero.wav'),('Positive-caption teacher','03_REF_prompt_Female_no_slider.wav')]:
        output.append('<tr><th>'+name+'</th>'+''.join('<td>'+player(GALLERY/f'{first}-s{s}'/filename)+'</td>' for s in (7,23))+'</tr>')
    output+=['</table><p>ASR can misread singing. Word match, phrase match and sheet coverage are separate diagnostics; slower phrasing can reduce coverage. These two seeds do not establish musical quality or long-run stability.</p>']
    (GALLERY/'index.html').write_text('\n'.join(output))
    print('Validated',len(validation),'WAV files;',len(rows),'ASR records')


if __name__=='__main__':main()
