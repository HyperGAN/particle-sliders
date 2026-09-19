"""Build the public card and use guides around the fixed selected-checkpoint package."""
from pathlib import Path
import csv
import importlib.util
import io
import json
import re
import sys
import yaml
from build_assets import WORK,ROOT,PACKAGE,VERSION,REPO,read,write,copy,names

spec=importlib.util.spec_from_file_location('previous_release',ROOT/'scripts/publish_music3_hub_20260909.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
URL=f'https://huggingface.co/{REPO}/resolve/main/'

def player(path):return f'<audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="{URL}{path}"></audio>'
def pair_block(pair,title):
    off,on=pair['off'],pair['on']
    return (f'### {title}\n\nSame prompt · same lyrics · seed **{pair["seed"]}** · selected step **{pair["checkpoint"][4:]}**\n\n'
      '<table style="display:table;width:100%;table-layout:fixed">\n<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>\n'
      f'<tbody><tr><td>{player(off["mp3"])}</td><td>{player(on["mp3"])}</td></tr></tbody>\n</table>\n\n'
      f'[Off WAV]({URL}{off["wav"]}) · [On WAV]({URL}{on["wav"]}) · [Target-caption reference]({URL}{pair["reference"]["mp3"]})\n\n')

def build():
    catalog=read(PACKAGE/'catalog.json');concepts=catalog['sliders']
    pairs=read(PACKAGE/f'samples/{VERSION}/pairs.json')['pairs']
    selected={c['id']:next(p for p in pairs if p['id']==c['id'] and p['featured']) for c in concepts}
    old.showcase_banner(PACKAGE,selected['female'],label='Female')
    front={'license':'mit','base_model':'MiniMaxAI/MiniMax-Music-3','tags':['music','text-to-music','lora','concept-sliders','minimax-music-3','comfyui'],
        'library_name':'diffusers','pipeline_tag':'text-to-audio'}
    card='---\n'+yaml.safe_dump(front,sort_keys=False)+'---\n\n'
    card+='''# MiniMax Music 3 concept sliders

## NEW — September 16, 2026

**16 newly retrained voice and genre LoRAs, with new matching samples and ComfyUI exports.** The downloads below are the selected checkpoints from the new four-prompt training runs.

![Same seed. Different sound. Off and On waveforms from the selected Female checkpoint.](assets/same-seed.svg)

**New voice and genre controls · Selected checkpoints · Native and ComfyUI downloads**

Steer the lead toward a female vocal, a piano-led arrangement toward heavy riffs, or a small band toward a club pulse. These LoRA adapters change the language model that plans music in [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).

**September 16, 2026:** all sixteen retrains are complete. This release uses the checkpoints selected by the final per-slider audit: steps 1,000, 2,000, 3,000 or 3,400 depending on the control. All voice and genre downloads and On samples below refer to those selected checkpoints.

[Choose a control](#choose-a-control) · [ComfyUI](#comfyui) · [Native loading](#native-loading) · [Selection criteria](#how-these-checkpoints-were-chosen) · [Training](#training) · [How it works + math](#how-the-sliders-learn)

## Press play

**Off** uses the base model. **On** adds the named adapter at strength **+1**. The prompt, lyrics and seed match within each pair; composition and phrasing can still change. Recordings retain their original levels and lengths.

'''
    for sid in ('female','metal','house','disco-funk','pop-punk','lofi'):
        c=next(c for c in concepts if c['id']==sid)
        card+=pair_block(selected[sid],c['label'])
    card+=f'[Hear all 64 Off/On comparisons](samples/{VERSION}/README.md). Every selected checkpoint has two arrangements × two seeds. Each featured example uses the first fixed arrangement and seed; examples were not chosen by their individual scores. Target-caption references are also included.\n\n'
    card+='## Choose a control\n\n**All sixteen downloads in this table are NEW as of September 16, 2026.** Each file adds one target sound. All are language-model attention adapters with rank 8 and alpha 8. Female and Male are separate controls; negative strength is not a trained opposite.\n\n'
    card+='| New control | Target sound | Selected step | Native | ComfyUI | New samples |\n|---|---|---:|---|---|---|\n'
    for c in concepts:
        desc=c['description'] if c['id'] not in ('female','male') else f'Encourages one adult {c["id"]} lead vocal'
        side=c['weights'].replace('.safetensors','.json')
        card+=f'| {c["label"]} | {desc} | {c["training_steps"]:,} | [Weights]({c["weights"]}) · [Settings]({side}) | [Weights]({c["comfyui_weights"]}) | [4 pairs](samples/{VERSION}/{c["id"]}/README.md) |\n'
    card+=f'''\n[Machine-readable catalog](catalog.json) · [Selection scores and checkpoint comparisons](evidence/{VERSION}/README.md) · [Release file hashes](release-manifest.json)

## ComfyUI

1. Download the **ComfyUI** file for your chosen control and place it in `ComfyUI/models/loras/`.
2. Use **Load LoRA** with your Music 3 text encoder connected to **CLIP**.
3. Set **strength_model = 0** and **strength_clip = 1** for the published On treatment. Compare with CLIP strength 0; start with one adapter.

Use a Music 3 text encoder with separate `q_proj`, `k_proj`, `v_proj` and `o_proj` layers. Text encoders with merged `qkv_proj` layers cannot apply the separate query/key/value factors in these files. See the [ComfyUI loading guide](comfyui/README.md).

All sixteen ComfyUI files were produced by the upstream conversion script from the selected native exports. Factors are BF16 and alpha scalars are FP32. The published recordings use the native adapters; ComfyUI precision and generation settings can change the resulting audio.

## Native loading

Download the native weight and its matching JSON settings file. Use the included [loading example](usage.md) with your existing Music 3 pipeline.

```python
from huggingface_hub import snapshot_download

folder = snapshot_download(
    "{REPO}",
    allow_patterns=[
        "catalog.json", "usage.md", "source/lora.py",
        "weights/{VERSION}/female/*",
        "prompts/{VERSION}/*",
    ],
)
```

Replace `female` with a control ID, or use `weights/{VERSION}/*/*` for all native files. For ComfyUI only, download `comfyui/{VERSION}/*`.

Start at direct multiplier **1** and compare with **0** using the same caption, lyrics and seed. The studio normalizes slider mixtures by host energy, so its fader position may differ from the direct multiplier used here.

## How these checkpoints were chosen

The final audit scored **80 checkpoint candidates**, with four matched clips per candidate. It excluded checkpoints with technical flags, then evaluated **Content Enjoyment (CE)** and **Production Quality (PQ)** separately using Audiobox Aesthetics.

For each control, a checkpoint qualifies when its mean CE is within **0.2 points** of the best clean CE and its mean PQ is within **0.2 points** of the best clean PQ. Among qualifying checkpoints, the latest checkpoint from the new training run is preferred. The two best values can come from different checkpoints. This is the `quality-later-v2` policy.

**Description similarity has no influence on selection.** CLAP measurements are kept as a separate diagnostic. Cached lyric checks are informational only. There is no combined style/quality score in this selection rule.

The 0.2-point tolerance is a preference rule, not a calibrated audible threshold. The [selection report](evidence/{VERSION}/README.md) shows how picks change at 0.1, 0.2 and 0.3 points. Four short clips support a shortlist; they do not establish universal quality, broad musical diversity, full-song reliability or behavior when stacking adapters.

All sixteen selected native exports contain finite tensors and match their saved full training states. Original-waveform checks cover silence, clipping, shortening, high-frequency excess, stereo issues, interior gaps and repeated or near-identical audio. These checks do not rule out subtle musical defects.

## Training

Each slider uses all **four existing prompt pairs**. A fresh rank-8 adapter first trains for 600 updates with all four rows per batch. Further updates use one row at a time in balanced shuffled passes, with a fresh frozen-base continuation on every update. Continuations supply ending supervision; the prompt-state style targets remain fixed per row.

Eight runs reached 3,400 updates, and the other eight stopped at the approved 2,000-update budget. The published checkpoints were selected afterward. Generator and critic learning rates stay at 0.0005 and 0.00075, with adversarial, feature-matching and ending terms each weighted 1. Milestone samples retain their first draws without seed retries.

[Method and equations](evidence/{VERSION}/formulation.md) · [Training record](evidence/{VERSION}/training.json) · [Checkpoint integrity](evidence/{VERSION}/integrity.json) · [Conversion record](evidence/{VERSION}/conversion.json)

'''
    card+=(WORK/'math-section.md').read_text()+'\n'
    parent=read(WORK/'remote-before.json')['revision']
    card+=f'## Earlier releases\n\nThe update-660 UNI16 release and older uni-lyric, v24 and bipolar weights remain at their existing paths. The root catalog and download table now point to the selected fresh-continuation release. [Previous model card](https://huggingface.co/{REPO}/blob/{parent}/README.md).\n'
    (PACKAGE/'README.md').write_text(card)
    (ROOT/'docs/hub-readme-fresh-selected.md').write_text(card)
    gallery=f'# Selected voice and genre comparisons\n\nAll 64 fixed Off/On pairs for `{VERSION}`. Two held-out arrangements × seeds 1709 and 2903 per control. The first arrangement and seed appear first consistently. Original waveforms and level-preserving MP3 previews are available for Off, selected On (+1), and the positive-caption reference.\n\n'
    for c in concepts:
        sid=c['id'];sub=f'# {c["label"]}: selected step {c["training_steps"]:,}\n\n'
        sub+=f'[Native weights]({URL}{c["weights"]}) · [ComfyUI weights]({URL}{c["comfyui_weights"]})\n\n'
        sub+='The Off and On arms share the neutral prompt, lyrics and seed. The reference uses the positive target caption with no adapter.\n\n'
        for p in [p for p in pairs if p['id']==sid]:
            sub+=pair_block(p,f'Arrangement {p["row"]-1} · seed {p["seed"]}')
        sub+=f'[Prompts and lyrics]({URL}{c["eval_prompts"]}) · [Machine-readable comparisons]({URL}samples/{VERSION}/pairs.json)\n'
        (PACKAGE/f'samples/{VERSION}/{sid}/README.md').write_text(sub)
        gallery+=f'## {c["label"]} · step {c["training_steps"]:,}\n\n'+sub.split('\n\n',1)[1]+'\n'
    (PACKAGE/f'samples/{VERSION}/README.md').write_text(gallery)
    native=(WORK/'previous/usage.md').read_text()
    native=native.replace('manual_seed(23)','manual_seed(1709)')
    native=native[:native.index('ComfyUI files from older releases')]+f'For the selected ComfyUI exports, use the [ComfyUI guide](comfyui/README.md). The native and ComfyUI catalog entries identify the same selected checkpoints in their respective formats.\n'
    (PACKAGE/'usage.md').write_text(native)
    (ROOT/'docs/hub-native-usage.md').write_text(native)
    comfy=f'''# ComfyUI: selected Music 3 language-model adapters

The [current catalog](../catalog.json) lists native and converted paths for the same sixteen selected checkpoints. Download the files under [`{VERSION}/`]({VERSION}/).

Place a converted `.safetensors` in `ComfyUI/models/loras/`, refresh the model list, and use **Load LoRA** with the Music 3 CLIP/text encoder connected. Set **strength_model = 0** and **strength_clip = 1**. Compare against CLIP strength 0 while keeping the prompt, lyrics and seed fixed. The matching `.safetensors.json` file records provenance and settings; ComfyUI uses the tensor file to apply the adapter.

These are text-encoder adapters. They target 36 attention layers × four projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), for 144 LoRA modules and 432 tensors including alpha values. Use an unmerged Music 3 text encoder with these separate projections. A merged `qkv_proj` text encoder leaves the separate query/key/value LoRA keys unused; applying only output projections is not the complete adapter.

Conversion used [mikkel/conceptmod's script](https://github.com/mikkel/conceptmod/blob/a8a9e898ea618d83f05505c5ece7c8e4ffa9c3df/scripts/convert_lora_comfyui.py):

```bash
python scripts/convert_lora_comfyui.py selected_native.safetensors
```

The script writes a `_comfyui.safetensors` file and a JSON sidecar. This release ran it on all sixteen selected exports. Every renamed factor was checked against its native FP32 source cast to BF16, and all FP32 alpha values were preserved. The [conversion evidence](../evidence/{VERSION}/conversion.json) records source and output hashes. The listening samples were generated through the native pipeline; no claim of bit-identical ComfyUI audio is made.

All sixteen converted files were also checked through ComfyUI's real CPU LoRA loader against the unmerged Music 3 text-encoder topology: **144 of 144 projections bound per checkpoint, with zero unused keys**. Query, key, value and output update calculations were checked on each export. [Loader validation and tested ComfyUI revision](../evidence/{VERSION}/comfyui-loader.json). This checks loading and weight updates; a complete ComfyUI audio generation was not run.

Start with one adapter at strength 1. The published screening covers strength 1 only; negative strengths and stacked controls are not validated by those comparisons.
'''
    (PACKAGE/'comfyui/README.md').write_text(comfy)
    report='# Checkpoint selection evidence\n\nThe release follows `quality-later-v2`: pass the technical screen, stay within 0.2 of the best clean mean on each quality dimension, then prefer the latest qualifying new-run checkpoint. Description similarity and lyrics do not choose checkpoints.\n\n'
    report+='| Control | Selected | CE | PQ | Below best CE / PQ | Pick at 0.1 / 0.2 / 0.3 |\n|---|---:|---:|---:|---|---|\n'
    decisions={d['id']:d for d in read(PACKAGE/f'evidence/{VERSION}/selection.json')['sliders']}
    output=io.StringIO();csvwriter=csv.writer(output);csvwriter.writerow(['id','step','native','native_sha256','comfyui','comfyui_sha256','ce','pq','ce_shortfall','pq_shortfall'])
    for c in concepts:
        d=decisions[c['id']];g=c['quality_shortfall'];picks=' / '.join(v['recommendation'].replace('step','') if v['recommendation'] else 'review' for v in d['sensitivity'])
        report+=f'| {c["label"]} | {c["training_steps"]} | {c["enjoyment"]:.3f} | {c["production"]:.3f} | {g["enjoyment"]:.3f} / {g["production"]:.3f} | {picks} |\n'
        csvwriter.writerow([c['id'],c['training_steps'],c['weights'],c['sha256'],c['comfyui_weights'],c['comfyui_sha256'],c['enjoyment'],c['production'],g['enjoyment'],g['production']])
    report+='\nThe tolerance comparison shows decision sensitivity; it is not a confidence interval or a listening-equivalence test. Means use four clips from two arrangements and two seeds. No four-independent-prompt uncertainty claim is made.\n\nIndie Rock 3400 was excluded for a high-frequency-energy flag; the selected Indie Rock 1000 passes the screen. A spectral flag can indicate intended brightness or an artifact and is not a diagnosis by itself.\n\n[Exact selection policy](selection-policy.json) · [All 80 candidate summaries](selection.json) · [Integrity checks](integrity.json) · [CSV of selected downloads](selected-checkpoints.csv)\n'
    (PACKAGE/f'evidence/{VERSION}/README.md').write_text(report)
    (PACKAGE/f'evidence/{VERSION}/selected-checkpoints.csv').write_text(output.getvalue())
    method=r'''# Fresh-continuation slider training

Each control is a rank-8, alpha-8 adapter on the frozen Music 3 language model's attention projections. At inference, strength s applies W(s) = W0 + s (alpha / rank) BA. The adapter has no trained negative pole.

A starting caption and target caption share lyrics, tempo and section tags. The frozen model provides prompt-state sequences H0 and H+, sampled at corresponding lyric tokens and the audio-start token. The adapted model supplies Hθ on the starting caption. The target and student changes are scaled by a fixed RMS calibration σ:

$$
x_+ = (H_+ - H_0)/\sigma, \qquad x_\theta = (H_\theta - H_0)/\sigma.
$$

A two-layer transformer critic D learns the target change with a relativistic pairing objective and an input-gradient cap:

$$
\mathcal L_D = \mathbb E[\operatorname{softplus}(D(x_\theta)-D(x_+))] + \mathcal R_{\mathrm{cap}}.
$$

The adapter minimizes adversarial comparison, feature matching and ending supervision, each with coefficient 1:

$$
\mathcal L_G = \mathbb E[\operatorname{softplus}(D(x_+)-D(x_\theta))]
+ \operatorname{MSE}(\mathbb E[\phi(x_\theta)],\mathbb E[\phi(x_+)]) + \mathcal L_{\mathrm{end}}.
$$

The feature term compares batch means of critic features. Warm-up uses all four rows per batch; continuation uses a single row per update, so this comparison is then between that update's student and teacher features. Prompt-state targets stay fixed per training row.

Ending supervision compares the audio-end versus semantic-continuation margin on the same frozen-base token history:

$$
m = \ell_{\mathrm{audio\_end}} - \log\sum_{j\in\mathcal S}e^{\ell_j},
\qquad \mathcal L_{\mathrm{end}}=\operatorname{MSE}(m_\theta,m_0).
$$

During the initial 600-update warm-up, each row has a fixed base history. From update 601 onward, every update samples a fresh base-model continuation with the adapter disabled. All four training rows participate in balanced shuffled passes. Seeds do not cycle and duplicate continuation tensors are rejected without substituting another seed. Fresh continuations diversify ending supervision; they do not replace the fixed prompt-state style teachers.

The critic's gradient cap penalizes only norms above 1, with coefficient 1. Generator and critic rates are 0.0005 and 0.00075, constant throughout. The continuation limits each adapter parameter update to L2 norm 2. The critic, both optimizers, sampler and RNG states carry across milestones. Explicit lyric hold is disabled.

Eight runs completed 3400 updates and eight completed 2000 under the approved cost budget. Selection happens after training using the separate quality rule in [selection-policy.json](selection-policy.json). These selected steps are not claims of convergence. Each selected export is verified against every corresponding LoRA tensor in its pinned full state.

[Full recipe and budgets](training.json) · [Checkpoint selection](README.md) · [Native loading](../../usage.md)
'''
    (PACKAGE/f'evidence/{VERSION}/formulation.md').write_text(method)
    (ROOT/'docs/hub-formulation-fresh-selected.md').write_text(method)
    for p in PACKAGE.rglob('*'):
        if p.is_file() and p.suffix in ('.md','.json','.yaml','.svg','.csv'):names(p.read_text(),str(p.relative_to(PACKAGE)))
    print('Built model card, 16 sample pages, native/ComfyUI guides and selection evidence.',flush=True)

if __name__=='__main__':build()
