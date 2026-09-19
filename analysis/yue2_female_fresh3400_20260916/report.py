"""Live training status, separate quality/style columns, and matched players."""
import html
from common import *


def main():
    PAGE.mkdir(parents=True,exist_ok=True)
    training=read(RUN/"status.json",{})
    progress=read(RUN/"progress.json",{})
    campaign=read(WORK/"campaign.json",{})
    write(PAGE/"status.json",dict(training=training,progress=progress,campaign=campaign))
    ranking=read(PAGE/"ranking.json",{})
    labels=[label for label in ["off","caption","reference600",*[f"step{s}" for s in MILESTONES]]
        if any((PAGE/f"row-{row}-seed-{seed}"/label/"song.wav").exists() for row,seed in CASES)]
    names={"off":"Off","caption":"Female caption","reference600":"Original 600"}
    names.update({f"step{s}":f"Fresh run · {s}" for s in MILESTONES})
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>YuE2 female · fresh training to 3400</title>',
        '<style>body{font:16px system-ui;background:#11161c;color:#eef2f5;max-width:1200px;margin:36px auto;padding:0 20px}p{line-height:1.6;color:#c1ccd7}a{color:#a5cdff}section{padding:22px;margin:26px 0;background:#1c2530;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}audio{width:100%}table{width:100%;border-collapse:collapse}th,td{padding:12px 8px;text-align:left;border-bottom:1px solid #334254}.scroll{overflow:auto}.chosen{color:#b3f5c8}pre{white-space:pre-wrap;font:14px/1.5 system-ui}h1{font-size:30px}h3{font-size:17px}</style>',
        '<h1>YuE2 female · fresh training to 3400</h1>',
        '<p id="status">Loading training progress…</p>',
        '<p>Fresh sampled histories from the first update, with one consistent training setup throughout. The original 600-step slider is preserved as a separate reference.</p>',
        '<p>Checkpoint selection prefers the latest clean candidate within 0.2 of the best enjoyment and production scores separately. Vocal-style similarity stays separate from that decision.</p>',
        '<p><a href="ranking.json">Full ranking</a> · <a href="protocol.json">Experiment settings</a> · <a href="waveform-audit.json">Waveform checks</a></p>']
    chosen=ranking.get("recommendation")
    if chosen:
        parts.append(f'<p class="chosen">Current provisional choice: <strong>{names[chosen["label"]]}</strong> · <a href="selected/female-yue2.safetensors">Download weights</a> · <a href="selected/selection.json">Selection details</a></p>')
    if ranking.get("ranking"):
        parts.append('<div class="scroll"><table><tr><th>Checkpoint</th><th>Enjoyment</th><th>Production</th><th>Female-style change</th><th>Quality shortlist</th><th>Technical flags</th></tr>')
        for row in ranking["ranking"]:
            if "mean" not in row: continue
            parts.append(f'<tr><td>{names[row["label"]]}</td><td>{row["mean"]["enjoyment"]:.3f}</td><td>{row["mean"]["production"]:.3f}</td><td>{row["mean_style_gain"]:+.4f}</td><td>{"Yes" if row["within_quality_tolerance"] else "No"}</td><td>{html.escape(str(row["flags"]))}</td></tr>')
        parts.append('</table></div>')
    parts.append('<p>Two held-out arrangements × two seeds, all at strength 1. Every comparison uses matched lyrics and seeds. Previews are capped near 20 seconds and may end mid-phrase. Scores are diagnostic proxies; the audio remains the listening evidence.</p>')
    for row,seed in CASES:
        parts.append(f'<section><h2>{"Electronic" if row==2 else "Plucked strings"} · seed {seed}</h2><div class="grid">')
        for label in labels:
            path=f"row-{row}-seed-{seed}/{label}/song.wav"
            if (PAGE/path).exists():
                parts.append(f'<div><h3>{names[label]}</h3><audio controls preload="none" src="{path}"></audio></div>')
        parts.append('</div></section>')
    parts.append('''<script>
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));
async function update(){try{const r=await fetch('status.json',{cache:'no-store'});const s=await r.json();const t=s.training||{};const c=s.campaign||{};document.getElementById('status').textContent=`${t.completed||0} / ${t.total||3400} updates · ${c.stage||t.status||'Preparing'}`;}catch(e){}}
update();setInterval(update,15000);
</script></html>''')
    temporary=PAGE/"index.html.tmp";temporary.write_text("\n".join(parts));temporary.replace(PAGE/"index.html")


if __name__=="__main__": main()
