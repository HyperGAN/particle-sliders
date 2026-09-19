#!/usr/bin/env python3
"""Prepare the verified experimental GitHub assets without changing the Space."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT/'eval/listen/yue2-lora-distill-20260917'
REPO=ROOT.parent/'.cache/yue2-github-release'
EXPERIMENT=ROOT/'models/yue2-lora-distill-20260917'
URL='https://github.com/mikkel/yue2-concept-sliders/releases/download/distilled-rank8-20260917/'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    summary=json.loads((PAGE/'summary.json').read_text())
    verified=json.loads((PAGE/'verification.json').read_text())
    assert summary['completed']==16 and verified['complete'] and verified['audio_clips']==128
    destination=REPO/'validation/distillation'
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(PAGE/'summary.json',REPO/'validation/distillation-summary.json')
    shutil.copy2(PAGE/'verification.json',REPO/'validation/distillation.json')
    for row in summary['sliders']:
        shutil.copy2(EXPERIMENT/row['concept']/'refined-evaluation.json',destination/(row['concept']+'.json'))
    report=REPO/'DISTILLATION.md'
    text=report.read_text().split('\n## Final measurements')[0]
    text+='\n## Final measurements\n\nAll 16 distillations and 128 diagnostic recordings completed on both GPUs. '
    text+='Every converted file loaded all 56 fused attention patches through the real ComfyUI LoRA loader. '
    text+='The native files contain only ordinary LoRA tensors.\n\n'
    text+='Values below average the music-region measurements across two held-out prompts, '
    text+='base and teacher token trajectories, and strengths 0.5 and 1. Lower relative MSE is better; '
    text+='zero is exact agreement, and one is the error from leaving the slider off. '
    text+='These are teacher-fidelity diagnostics, not percentages of perceptual quality.\n\n'
    text+='| Control | Relative steering MSE | Steering cosine |\n|---|---:|---:|\n'
    for row in summary['sliders']:
        text+=f'| {row["concept"]} | {row["music_relative_mse"]:.4f} | {row["music_cosine"]:.4f} |\n'
    text+='\n[Aggregated measurements](validation/distillation-summary.json) · '
    text+='[Export and audio checks](validation/distillation.json) · '
    text+='[Validation histories and held-out measurements](validation/distillation/)\n'
    report.write_text(text)
    (PAGE/'DISTILLATION.md').write_text(text)
    validation_files=list((REPO/'validation/distillation').glob('*.json'))+[
        REPO/'validation/distillation-summary.json',REPO/'validation/distillation.json']
    for filename in ['yue2-ordinary-loras-comfyui.zip','yue2-ordinary-loras-native.zip']:
        source=PAGE/filename
        temporary=source.with_suffix('.tmp.zip')
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as new:
            for entry in old.infolist():
                if entry.filename in ('README.md','DISTILLATION.md') or entry.filename.startswith('validation/'):
                    continue
                new.writestr(entry,old.read(entry.filename))
            new.writestr('README.md',text)
            new.writestr('DISTILLATION.md',text)
            for file in validation_files:new.write(file,file.relative_to(REPO))
        temporary.replace(source)
    markup=(PAGE/'index.html').read_text()
    def replace_download(match):
        path=match.group(1)
        name='yue2-ordinary-loras-comfyui.zip' if '_comfyui.safetensors' in path else 'yue2-ordinary-loras-native.zip'
        return 'href="'+URL+name+'"'
    markup=re.sub(r'href="(downloads/[^" ]+\.safetensors)"',replace_download,markup)
    for filename in ['yue2-ordinary-loras-comfyui.zip','yue2-ordinary-loras-native.zip']:
        markup=markup.replace('href="'+filename+'"','href="'+URL+filename+'"')
    archive=PAGE/'yue2-distillation-comparisons.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('index.html',markup)
        z.write(report,'README.md')
        for common in ['summary.json','verification.json','workflow.json','WORKFLOW_LICENSE','WEIGHTS_LICENSE.txt']:
            z.write(PAGE/common,common)
        for file in sorted((PAGE/'audio').rglob('*.flac')):z.write(file,file.relative_to(PAGE))
        for file in sorted((PAGE/'downloads').rglob('*.json')):z.write(file,file.relative_to(PAGE))
        for file in validation_files:z.write(file,file.relative_to(REPO))
        for file in sorted(EXPERIMENT.glob('*/audio/row*.json')):
            z.write(file,Path('audio')/file.parent.parent.name/file.name)
    files=[PAGE/name for name in ['yue2-ordinary-loras-comfyui.zip','yue2-ordinary-loras-native.zip','yue2-distillation-comparisons.zip']]
    manifest=dict(tag='distilled-rank8-20260917',experimental=True,
        assets=[dict(name=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in files],
        teacher_release='ntc-ai/yue2-concept-sliders:particle-1200-v1',
        converter_commit=verified['converter_commit'],
        source_hashes={name:digest(ROOT/'scripts'/name) for name in
            ['distill_yue2_particles.py','refine_yue2_distillation.py','report_yue2_distillation.py','verify_yue2_distillation.py']})
    for path in [PAGE/'artifact-manifest.json',REPO/'validation/distillation-artifacts.json']:
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
