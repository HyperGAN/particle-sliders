#!/usr/bin/env python3
"""Publish local distillation progress, downloads and matched listening clips."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'models/yue2-lora-distill-20260917'
PAGE = ROOT/'eval/listen/yue2-lora-distill-20260917'
RELEASE = ROOT.parent/'releases/yue2-concept-sliders/model'
CONVERTER = ROOT.parent/'.cache/conceptmod-yue2-lora'
CSS = '''body{font:17px/1.6 system-ui;margin:0;background:#10161c;color:#e6eef5}main{max-width:1440px;margin:auto;padding:24px}a{color:#89c9ff}h1{font-size:36px;line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;text-align:left;border-bottom:1px solid #33414c}.scroll{overflow:auto}.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,290px),1fr));gap:18px}.clip{padding:16px;background:#1a2630;border-radius:12px}audio{width:100%;height:58px;min-width:260px}summary{cursor:pointer}pre{white-space:pre-wrap;background:#1a2630;padding:18px;border-radius:10px}.muted{color:#afbdc9}.badge{color:#9bdec0}.row{border-top:1px solid #33414c;padding-top:8px}button{font:inherit;min-height:46px}'''


def link_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_mtime_ns == source.stat().st_mtime_ns:
        return
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def mean(rows, key):
    return sum(row[key] for row in rows)/len(rows)


def build():
    PAGE.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((RELEASE/'catalog.json').read_text())
    entries, sections = [], []
    table = []
    complete = 0
    for spec in catalog['sliders']:
        concept, label = spec['id'], html.escape(spec['label'])
        directory = EXPERIMENT/concept
        chosen = directory/'refined-evaluation.json'
        fitted = directory/'evaluation.json'
        if not chosen.exists() and not fitted.exists():
            table.append(f'<tr><td>{label}</td><td colspan="4">Fitting</td></tr>')
            continue
        record = json.loads((chosen if chosen.exists() else fitted).read_text())
        music = [m for m in record['heldout'] if m['region']=='music']
        boundary = [m for m in record['heldout'] if m['region']=='boundary']
        status = 'Selected; rendering' if chosen.exists() else 'First fit; refining'
        summary = directory/'audio/summary.json'
        recordings = []
        if summary.exists():
            status = 'Complete'
            complete += 1
            recordings = json.loads(summary.read_text())['recordings']
        downloads = ''
        if chosen.exists():
            checkpoint = directory/record['checkpoint']
            converted = checkpoint.with_name(checkpoint.stem+'_comfyui.safetensors')
            if not converted.exists():
                subprocess.run([sys.executable, '-m', 'conceptmod.convert', str(checkpoint)],
                    cwd=CONVERTER, env=dict(os.environ, PYTHONPATH=str(CONVERTER)), check=True,
                    stdout=subprocess.DEVNULL)
            for source in (checkpoint, checkpoint.with_suffix('.json'), converted,
                           converted.with_name(converted.name+'.json'), chosen):
                link_file(source, PAGE/'downloads'/concept/source.name)
            downloads = (f'<p><a download href="downloads/{concept}/{converted.name}">ComfyUI LoRA</a> · '
                         f'<a download href="downloads/{concept}/{checkpoint.name}">Native LoRA</a> · '
                         f'<a href="downloads/{concept}/refined-evaluation.json">Measurements</a></p>')
        entry = dict(concept=concept, status=status, selected=chosen.exists(),
            music_relative_mse=mean(music,'relative_steering_mse'),
            music_cosine=mean(music,'steering_cosine'), boundary_relative_mse=mean(boundary,'relative_steering_mse'),
            teacher_student_kl=mean(music,'teacher_student_kl'), teacher_base_kl=mean(music,'teacher_base_kl'),
            checkpoint=record['checkpoint'], checkpoint_sha256=record['checkpoint_sha256'])
        entries.append(entry)
        table.append(f'<tr><td><a href="#{concept}">{label}</a></td><td>{status}</td>'
                     f'<td>{entry["music_relative_mse"]:.3f}</td><td>{entry["music_cosine"]:.3f}</td>'
                     f'<td>{entry["boundary_relative_mse"]:.3f}</td></tr>')
        section = [f'<section id="{concept}"><h2>{label}</h2>{downloads}']
        for row in range(2):
            section.append(f'<div class="row"><h3>Held-out passage {row+1} · seed 1709</h3><div class="clips">')
            for kind, title in [('off','Off'),('teacher','Original particles'),
                                ('plain-ar','Plain LoRA · AR only'),('plain-all','Plain LoRA · AR + acoustic prefix')]:
                source = directory/'audio'/f'row{row}-{kind}.flac'
                if source.exists():
                    target = PAGE/'audio'/concept/source.name
                    link_file(source, target)
                    section.append(f'<div class="clip"><strong>{title}</strong><audio controls preload="none" '
                                   f'src="audio/{concept}/{source.name}"></audio></div>')
            section.append('</div></div>')
        section.append('</section>')
        sections.append(''.join(section))
    bundles = ''
    if (PAGE/'yue2-ordinary-loras-comfyui.zip').exists():
        bundles = '<p><strong><a download href="yue2-ordinary-loras-comfyui.zip">Download all 16 ComfyUI LoRAs</a></strong> · <a download href="yue2-ordinary-loras-native.zip">Native LoRAs</a> · <a href="verification.json">Validation</a></p>'
    description = f'''<h1>YuE2: particles → ordinary LoRAs</h1><p class="badge">{complete}/16 complete · both GPUs · rank 8 · experimental</p>{bundles}
<p>These are ordinary low-rank weight updates with no particle cloud, router or custom inference adapter.
The original particle release remains the reference. Download a ComfyUI file and put it in <code>models/loras/</code>.
Use <strong>Load LoRA</strong>, connect both MODEL and CLIP, set model strength to <strong>0</strong> and CLIP strength to <strong>1</strong>.</p>
<p><a download href="workflow.json">Standard-node ComfyUI workflow</a> · <a href="README.md">Method and limitations</a> · <a href="summary.json">All measurements</a></p>
<p>Each comparison uses the same caption, lyrics and seed. These are approximately 20-second diagnostic excerpts.
The fourth player tests applying the LoRA during acoustic-prefix conditioning as well, matching the scope of ComfyUI’s standard loader.
It is generated with the native runtime; it is not a ComfyUI audio render.</p>
<p class="muted">Lower relative steering MSE is better: 0 is an exact teacher match; 1 is the error from leaving the slider off.
Cosine measures direction (1 is aligned). These hidden-state diagnostics do not establish perceptual quality or full-song reliability.</p>
<div class="scroll"><table><thead><tr><th>Control</th><th>Status</th><th>Music relative MSE ↓</th><th>Music cosine ↑</th><th>First-token relative MSE ↓</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>'''
    content = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YuE2 ordinary LoRA distillation</title><style>'+CSS+'</style><main>'+description+''.join(sections)+'</main></html>'
    (PAGE/'index.html.tmp').write_text(content)
    (PAGE/'index.html.tmp').replace(PAGE/'index.html')
    (PAGE/'summary.json').write_text(json.dumps(dict(completed=complete,total=16,sliders=entries),indent=2)+'\n')
    print(f'{time.strftime("%H:%M:%S")} {len(entries)} fitted, {complete}/16 complete', flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        finished = build()
        if not args.watch or finished == 16:
            break
        time.sleep(20)
