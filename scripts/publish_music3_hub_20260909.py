"""Build, verify, then explicitly publish the September 9 Music 3 Hub release.

Uses existing recordings and weights only. Does not start GPU jobs or alter
frozen research records. Run with the existing minimax-music3 interpreter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
UNI = ROOT / 'analysis/uni16_20260906'
REWARD = ROOT / 'analysis/reward_game_v1_20260908'
FRESH = REWARD / 'block-v1/confirmation/robust-block-first-v1'
REPO = 'ntc-ai/minimax-music3-concept-sliders'
URL = f'https://huggingface.co/{REPO}/resolve/main/'
REWARD_SHA = '05a4c017ad476a7f65f120db362e546850e3891327d4d01df4bdd954c1e9b548'
SHOWCASE = ['metal', 'house', 'disco-funk', 'pop-punk', 'lofi', 'female']
sys.path.insert(0, str(WORK))
from app.rewriter import _artist_name_hit


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def names(value, location):
    if isinstance(value, dict):
        for k, v in value.items():
            names(v, f'{location}.{k}')
    elif isinstance(value, list):
        for i, v in enumerate(value):
            names(v, f'{location}[{i}]')
    elif isinstance(value, str) and _artist_name_hit('', value):
        # Do not reproduce prohibited names in new artifacts or logs.
        raise ValueError(f'Name policy violation at {location}; cannot publish these weights')


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', 'source/').replace(str(WORK) + '/', 'workspace/')
    return value


class Release:
    def __init__(self, folder):
        self.folder = folder
        self.assets = {}
        self.encodes = []

    def copy(self, source, dest, expected=None):
        source = Path(source)
        digest = sha(source)
        if expected and digest != expected:
            raise ValueError(f'Source hash mismatch: {source}')
        target = self.folder / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != digest:
            shutil.copyfile(source, target)
        self.assets[dest] = {'sha256': digest, 'source': portable(str(source))}
        return dest

    def checkpoint(self, source, dest, expected, prompt_dest):
        source = Path(source)
        meta = read(source.with_suffix('.json'))
        names(meta, source.name)
        with safe_open(source, framework='pt', device='cpu') as f:
            names(f.metadata() or {}, source.name + '.metadata')
            for k in f.keys():
                if not torch.isfinite(f.get_tensor(k)).all():
                    raise ValueError(f'Nonfinite weight: {source.name}')
            count = len(f.keys())
        self.copy(source, dest, expected)
        meta = portable(meta)
        meta['prompts_file'] = prompt_dest
        meta['weights_sha256'] = expected
        write((self.folder / dest).with_suffix('.json'), meta)
        return count

    def audio(self, source, dest, expected=None, raw=False):
        source = Path(source)
        digest = sha(source)
        if expected and digest != expected:
            raise ValueError(f'Audio hash mismatch: {source}')
        samples, sr = sf.read(source, dtype='float32', always_2d=True)
        if len(samples) < sr or not np.isfinite(samples).all():
            raise ValueError(f'Invalid showcase audio: {source}')
        if float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) < 1e-5:
            raise ValueError(f'Silent showcase audio: {source}')
        mp3 = dest + '.mp3'
        self.encodes.append((source, self.folder / mp3, None))
        result = {'mp3': mp3, 'source_sha256': digest, 'duration_s': len(samples) / sr}
        if raw:
            result['wav'] = self.copy(source, dest + '.wav', digest)
        return result

    def pair(self, off, on, dest):
        self.encodes.append((Path(off), self.folder / dest, Path(on)))
        return dest


def encode(job):
    off, dest, on = job
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = None
    if on is not None:
        a, sr = sf.read(off, dtype='float32', always_2d=True)
        b, br = sf.read(on, dtype='float32', always_2d=True)
        if sr != br or a.shape[1] != b.shape[1]:
            raise ValueError('Paired recordings must share sample rate and channels')
        frames = min(20 * sr, len(a), len(b))
        data = np.concatenate([a[:frames], np.zeros((sr, a.shape[1]), dtype='float32'), b[:frames]])
        fd, tmp = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        sf.write(tmp, data, sr, subtype='FLOAT')
        off = Path(tmp)
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(off),
                        '-map_metadata', '-1', '-vn', '-c:a', 'libmp3lame', '-b:a', '192k',
                        '-threads', '1', str(dest)], check=True)
    finally:
        if tmp:
            Path(tmp).unlink()


def player(path):
    return f'<audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="{URL}{path}"></audio>'


def show_pair(item, title, description=''):
    off, on = item['off'], item['on']
    links = ' · '.join(f'[{label}]({URL}{v.get("wav", v["mp3"])})' for label, v in [('Off download', off), ('On download', on)])
    duration = f'{off["duration_s"]:.1f}s / {on["duration_s"]:.1f}s'
    timing = f'Same prompt · same lyrics · **seed {item["seed"]}**'
    context = f'{description}\n\n' if description else ''
    score = ''
    if 'delta_ce' in item:
        score = (f'**Enjoyment model score:** {item["off_ce"]:.3f} → {item["on_ce"]:.3f} '
                 f'(**{item["delta_ce"]:+.3f}** points).\n\n')
    comparison = (
        '<table style="display:table;width:100%;table-layout:fixed">\n<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>\n'
        f'<tbody><tr><td>{player(off["mp3"])}</td><td>{player(on["mp3"])}</td></tr></tbody>\n'
        '</table>\n\n'
    )
    short_note = f'Original takes: {duration}. The Off take ended early.\n\n' if off['duration_s'] < 20 else ''
    return f'### {title}\n\n{context}{timing}\n\n{comparison}{short_note}{score}{links}\n\n'


# Listener-facing descriptions of the fixed prompts. Keep research group IDs
# in the evidence; they do not describe the sound of a listening example.
REWARD_EXAMPLES = {
    'independent-bank-b-00': ('Piano & brushed drums', 'Female lead vocal · 76 BPM. Soft grand piano, plucked double bass, brushes and a low clarinet response.'),
    'independent-bank-b-01': ('Open guitars & a rising refrain', 'Female lead vocal · 116 BPM. Open electric chords, picked bass, a dry kit and a quiet second vocal layer.'),
    'independent-bank-b-02': ('Fingerpicked guitar & warm harmonium', 'Male lead vocal · 88 BPM. Steel strings, low harmonium, upright bass and brushed floor tom.'),
    'independent-bank-b-03': ('Syncopated bass & clean guitar', 'Male lead vocal · 108 BPM. Short guitar strokes, bright electric piano and crisp snare backbeats.'),
    'independent-bank-b-04': ('Swung drums & mellow keys', 'Lead vocal · 84 BPM. Warm detuned keys, muted horn phrases, relaxed singing and light tape grain.'),
    'independent-bank-b-05': ('Acoustic guitar & dance pulse', 'Lead vocal · 120 BPM. Fingerpicked guitar, an even electronic kick, deep synth bass and an airy organ pad.'),
    'independent-bank-b-06': ('Synth melody & bell replies', 'Instrumental · 124 BPM. Rounded synth lead, steady kick, soft claps, moving sub bass and delicate shakers.'),
    'independent-bank-b-07': ('Picked guitar & cello replies', 'Instrumental · 100 BPM. Warm cello, plucked bass, a restrained kit and light piano countermelody.'),
}


def reward_highlights(pairs):
    return [next(p for p in pairs if p['voice'] == voice)
            for voice in ['unspecified', 'instrumental', 'male', 'female']]


def inline_formulation():
    """Use one math source for the main card and the standalone method page."""
    method = (ROOT / 'docs/hub-formulation-uni16-gan.md').read_text()
    method = re.sub(r'^(#{1,2}) ', r'\1# ', method, flags=re.MULTILINE)
    method = method.replace('](training.json)', '](evidence/uni16-gan-v1/training.json)')
    return method.replace('](../reward-refined-block/', '](evidence/reward-refined-block/')


def showcase_banner(folder, pair, label='Metal'):
    """Draw the actual matched excerpts at a shared amplitude scale."""
    envelopes = []
    for arm in ['off', 'on']:
        audio, sr = sf.read(folder / pair[arm]['wav'], always_2d=True)
        power = np.square(audio[:20 * sr]).mean(axis=1)
        envelopes.append(np.array([np.sqrt(chunk.mean()) for chunk in np.array_split(power, 80)]))
    amplitude = max(float(values.max()) for values in envelopes)
    waves = []
    for values, center, color in zip(envelopes, [118, 226], ['#81919c', '#c2ff56']):
        for i, value in enumerate(values):
            height = max(2, float(value) / max(amplitude, 1e-8) * 65)
            x = 613 + i * 5.5
            waves.append(f'<path d="M{x:.1f} {center-height/2:.1f}v{height:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="320" viewBox="0 0 1120 320" role="img" aria-labelledby="title desc">
<title id="title">Same seed. Different sound.</title>
<desc id="desc">Off and On amplitude envelopes from the {label} comparison, seed {pair['seed']}. Both excerpts use the same prompt and lyrics, displayed at the same amplitude scale.</desc>
<rect width="1120" height="320" rx="22" fill="#10171d"/>
<path d="M540 38v244" stroke="#2b3943"/>
<g font-family="Arial,Helvetica,sans-serif">
<text x="42" y="47" fill="#9dabb5" font-size="14" letter-spacing="2">MINIMAX MUSIC 3 / CONCEPT SLIDERS</text>
<text x="40" y="127" fill="#f4f7f9" font-size="57" font-weight="700" letter-spacing="-2">Same seed.</text>
<text x="40" y="191" fill="#c2ff56" font-size="57" font-weight="700" letter-spacing="-2">Different sound.</text>
<text x="43" y="274" fill="#c6d0d7" font-size="16">16 voice &amp; genre controls</text>
<rect x="564" y="65" width="72" height="27" rx="7" fill="#27343f"/>
<text x="578" y="84" fill="#c6d0d7" font-size="14" font-weight="700">OFF / 0</text>
<rect x="564" y="174" width="87" height="27" rx="7" fill="#c2ff56"/>
<text x="578" y="193" fill="#10171d" font-size="14" font-weight="700">ON / +1</text>
<text x="564" y="285" fill="#9dabb5" font-size="13" letter-spacing="1">{label.upper()} / SEED {pair['seed']} / FIRST 20 SECONDS</text>
</g>
''' + '\n'.join(waves) + '\n</svg>\n'
    path = folder / 'assets/same-seed.svg'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def build(folder):
    torch.set_num_threads(4)
    r = Release(folder)
    status = read(UNI / 'status.json')
    registry = read(WORK / 'app/sliders.json')
    catalog = read(UNI / 'catalog.json')
    catalog_by_id = {s['id']: s for s in catalog['sliders']}
    campaign = read(UNI / 'manifest.json')
    concepts, highlights, matrix = [], {}, []
    for slider in registry['sliders']:
        sid = slider['id']
        entry = catalog_by_id[sid]
        for split in ['train', 'eval']:
            p = Path(entry[f'{split}_prompts'])
            names(yaml.safe_load(p.read_text()), p.name)
            r.copy(p, f'prompts/uni16-gan-v1/{p.name}', campaign['source_files'][str(p)])
        source = Path(registry['root']) / slider['components'][0]['weights']
        expected = status['sliders'][sid]['bounded']['audit']['weights_sha256']
        dest = 'weights/' + slider['components'][0]['weights']
        prompt_dest = 'prompts/uni16-gan-v1/' + Path(entry['train_prompts']).name
        tensors = r.checkpoint(source, dest, expected, prompt_dest)
        assert tensors == 432
        item = {'id': sid, 'label': slider['label_plus'], 'description': slider['description'],
                'weights': dest, 'sha256': expected, 'kind': 'language_model', 'unit_scale': 1,
                'training_steps': 660, 'train_prompts': prompt_dest,
                'eval_prompts': 'prompts/uni16-gan-v1/' + Path(entry['eval_prompts']).name,
                'comparison_status': 'pending' if sid == 'reggaeton' else 'complete'}
        concepts.append(item)
        score_path = UNI / 'scores' / (sid + '.json')
        if not score_path.exists():
            continue
        report = read(score_path)
        records = [x for x in report['records'] if x['checkpoint']['sha256'] == expected]
        assert len(records) == 4 and all(x['eligible'] for x in records)
        best = max(records, key=lambda x: x['heuristic_score'])
        for rec in records:
            row = int(Path(rec['candidate']['audio']).parent.parent.name.split('-')[-1])
            prefix = f'samples/uni16-gan-v1/{sid}/row{row}-seed{rec["seed"]}'
            pair = {'id': sid, 'label': item['label'], 'row': row, 'seed': rec['seed'], 'weights_sha256': expected,
                    'heuristic_score': rec['heuristic_score'], 'components': rec['components'], 'featured': rec is best}
            for key, report_key in [('off', 'baseline'), ('on', 'candidate'), ('reference', 'positive_reference')]:
                raw = rec is best and key in ['off', 'on']
                arm = rec[report_key]
                pair[key] = r.audio(arm['audio'], prefix + '-' + key, arm['sha256'], raw=raw)
                pair[key]['measurements'] = {k: arm[k] for k in ['concept', 'enjoyment', 'production', 'lyrics']}
            pair['ab'] = r.pair(rec['baseline']['audio'], rec['candidate']['audio'], prefix + '-off-on.mp3')
            pair['caption'] = yaml.safe_load(Path(entry['eval_prompts']).read_text())['rows'][row]['neutral']
            pair['lyrics'] = yaml.safe_load(Path(entry['eval_prompts']).read_text())['rows'][row]['lyrics']
            matrix.append(pair)
            if rec is best:
                highlights[sid] = pair
        write(folder / f'evidence/uni16-gan-v1/scores/{sid}.json', portable(report))
    write(folder / 'catalog.json', {'version': 'uni16-gan-v1', 'sliders': concepts,
          'reward_slider_separate': 'reward/refined-block-step2/README.md'})
    write(folder / 'samples/uni16-gan-v1/pairs.json', {'selection': 'Highest recorded heuristic score within each slider at the deployed 660 checkpoint; four fixed first-draw cases each. No cross-genre quality ranking or new listening judgment.', 'pairs': matrix})
    write(folder / 'evidence/uni16-gan-v1/training.json', portable({k: campaign[k] for k in ['initialization', 'intervention', 'warmup_settings', 'bounded_recipe', 'evaluation', 'selection']}))
    write(folder / 'evidence/uni16-gan-v1/status.json', portable(status))
    r.copy(ROOT / 'docs/hub-formulation-uni16-gan.md', 'evidence/uni16-gan-v1/formulation.md')

    protocol = read(FRESH / 'protocol.json')
    assert protocol['candidate']['weights_sha256'] == REWARD_SHA
    source = Path(protocol['candidate']['path'])
    prompts = source.parent / 'training-prompts.json'
    names(read(prompts), prompts.name)
    reward_dest = 'reward/refined-block-step2/' + source.name
    prompt_dest = 'reward/refined-block-step2/training-prompts.json'
    r.copy(prompts, prompt_dest, protocol['candidate']['prompt_sha256'])
    assert r.checkpoint(source, reward_dest, REWARD_SHA, prompt_dest) == 648
    manifest = read(FRESH / 'manifest.json')
    names(manifest['families'], 'reward-families')
    families = {x['family']: x for x in manifest['families']}
    score = read(FRESH / 'scorecard.json')
    reward_pairs = []
    for row in sorted(score['rows'], key=lambda x: x['deltas']['off'], reverse=True):
        family = families[row['family']]
        pair = {'id': row['id'], 'family': row['family'], 'seed': row['seed'],
                'voice': family['voice'], 'caption': family['caption'], 'lyrics': family['lyrics'],
                'off_ce': row['off'], 'on_ce': row['candidate'], 'original_ce': row['original'],
                'delta_ce': row['deltas']['off'], 'weights_sha256': REWARD_SHA}
        for key, arm in [('off', 'off'), ('on', 'candidate'), ('original', 'original')]:
            observation = read(FRESH / 'observations' / f'{row["id"]}-{arm}.json')
            assert observation['status'] == 'complete'
            assert observation['seed'] == row['seed']
            if arm == 'candidate':
                assert sha(observation['arm']['checkpoint']) == REWARD_SHA
                assert observation['arm']['multiplier'] == 1
            pair[key] = r.audio(row['audio'][arm], f'samples/reward-refined-block/{row["id"]}-{key}', observation['audio_sha256'], raw=True)
            write(folder / f'evidence/reward-refined-block/observations/{row["id"]}-{arm}.json', portable(observation))
        pair['ab'] = r.pair(row['audio']['off'], row['audio']['candidate'], f'samples/reward-refined-block/{row["id"]}-off-on.mp3')
        reward_pairs.append(pair)
    for filename, relative in [('scorecard.json', 'scorecard.json'), ('preservation.json', 'intent/preservation.json'), ('protocol.json', 'protocol.json')]:
        write(folder / 'evidence/reward-refined-block' / filename, portable(read(FRESH / relative)))
    write(folder / 'evidence/reward-refined-block/training.json', portable(read(REWARD / 'recipes/robust-block-intent-v1.json')))
    write(folder / 'evidence/reward-refined-block/fixtures.json', manifest['families'])
    write(folder / 'samples/reward-refined-block/pairs.json', {'selection': 'All 16 fresh cases, ordered by recorded CE gain against Off. Not an independent human preference ranking.', 'pairs': reward_pairs})

    reward_intro = ('## Listen: reward slider\n\n'
        'This experimental adapter is trained to raise **Content Enjoyment (CE)**, an automated audio score on a 0–10 scale. '
        'It changes how the acoustic model renders the music. The release name is `refined block step 2`.\n\n'
        '| Check | Result |\n|---|---|\n'
        '| Higher enjoyment score | **15 of 16** fresh comparisons beat Off; mean gain **+0.110 points**. |\n'
        '| Uncertainty | 95% interval **+0.081 to +0.140**, resampling whole prompt families. |\n'
        '| Song preservation | **Failed.** One dynamics check failed versus Off; 12 comparisons flagged lyrics, style, voice or dynamics versus the original reward adapter. |\n'
        '| Still to check | Independent replication, use alongside studio sliders at fixed energy, and full-song endings. |\n\n'
        'A higher model score does not establish human preference. The four highlights below have the largest recorded score gain in each vocal/instrumental prompt group. '
        'Titles and descriptions summarize the prompts; both takes use the same prompt, lyrics and seed.\n\n')
    reward_gallery = '# Reward slider: all 16 comparisons\n\nUse the separate **Off** and **On** players to compare each pair. Both use the same prompt, lyrics and seed. Each player starts at the beginning of its take; original levels are preserved. This gallery includes the main-page highlights and every other fresh comparison.\n\nCases are ordered by model-score gain; the final case has a slightly lower score with the adapter On. Titles describe the musical prompts. “Lead vocal” means the prompt leaves the singer’s gender open.\n\n' + reward_intro.replace('The four highlights below have the largest recorded score gain in each vocal/instrumental prompt group.', 'All 16 fresh comparisons appear below, including the lower-scoring case.')
    reward_gallery += '| Musical prompt | Seed | Off score | On score | Change | Audio |\n|---|---|---|---|---|---|\n'
    for p in reward_pairs:
        title, _ = REWARD_EXAMPLES[p['family']]
        reward_gallery += f'| {title} | {p["seed"]} | {p["off_ce"]:.3f} | {p["on_ce"]:.3f} | {p["delta_ce"]:+.3f} | [Off]({URL}{p["off"]["mp3"]}) · [On]({URL}{p["on"]["mp3"]}) |\n'
    reward_gallery += '\n'
    for p in reward_pairs:
        reward_gallery += show_pair(p, *REWARD_EXAMPLES[p['family']])
        reward_gallery += f'[Original reward adapter control WAV]({URL}{p["original"]["wav"]})\n\n'
    (folder / 'samples/reward-refined-block/README.md').write_text(reward_gallery)
    concept_gallery = '# Voice and genre: all 60 comparisons\n\n**Two players. One seed.** Play Off and On separately from the beginning. Both takes use the same starting prompt, lyrics and seed. Most takes run 20 seconds; shorter takes are kept and labeled with their original lengths.\n\nEach control has two musical prompts rendered with two seeds. The selected highlight leads each group, followed by the other three comparisons. Selection uses the recorded automated score within that control; it does not rank genres or measure human preference. The target-caption reference lets you hear the base model with the requested sound written into the prompt and the adapter Off.\n\nReggaeton weights are available; its listening grid is pending.\n\n'
    for c in concepts:
        if c['id'] not in highlights:
            continue
        description = f'Encourages one adult {c["id"]} lead vocal' if c['id'] in ['female', 'male'] else c['description']
        concept_gallery += f'## {c["label"]}\n\n{description}. Phrasing and composition may also change.\n\n'
        for p in sorted([p for p in matrix if p['id'] == c['id']], key=lambda p: not p['featured']):
            concept_gallery += show_pair(p, f'Prompt {p["row"] + 1}' + (' · selected highlight' if p['featured'] else ''))
            concept_gallery += f'[Hear the target-caption reference]({URL}{p["reference"]["mp3"]})\n\n'
    (folder / 'samples/uni16-gan-v1/README.md').write_text(concept_gallery)

    header = {'license': 'mit', 'base_model': 'MiniMaxAI/MiniMax-Music-3', 'tags': ['music', 'text-to-music', 'lora', 'concept-sliders', 'minimax-music-3'],
              'library_name': 'diffusers', 'pipeline_tag': 'text-to-audio'}
    showcase_banner(folder, highlights['metal'])
    card = '---\n' + yaml.safe_dump(header, sort_keys=False, allow_unicode=True) + '---\n\n'
    card += """# MiniMax Music 3 concept sliders

![Same seed. Different sound. Off and On waveforms from the Metal comparison.](assets/same-seed.svg)

**16 voice & genre controls · Off / On · Same seed**

Push a piano-led song toward heavy riffs, a small band toward a club pulse, or a clean arrangement toward soft tape warmth. These LoRA adapters steer the music inside [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).

[Voice & genre](#voice--genre) · [All 16 controls](#choose-a-control) · [Download](#download-and-use) · [How it works + math](#how-the-sliders-learn) · [Experimental reward slider](#experimental-reward-slider)

## Press play

Two players per example. **OFF** is the base model; **ON** adds the named slider at **+1**. The prompt, lyrics and seed match within each pair. Play either take from the start; pause it before switching. These are separate generations, so the phrasing and arrangement can change.

"""
    reward_showcase = '## Experimental reward slider\n\nThis acoustic adapter is trained toward a higher Content Enjoyment score. **Off** uses the base model; **On** adds the reward adapter at **+1**. These four examples were selected by score within each vocal/instrumental prompt group. The adapter remains experimental: song-preservation checks failed, and score gains do not establish better sound.\n\n'
    for p in reward_highlights(reward_pairs):
        reward_showcase += show_pair(p, *REWARD_EXAMPLES[p['family']])
    reward_showcase += '[All 16 reward comparisons](samples/reward-refined-block/README.md) · [Download reward weights and view results](reward/refined-block-step2/README.md)\n\n'
    card += '## Voice & genre\n\n'
    for sid in SHOWCASE:
        description = next(c['description'] for c in concepts if c['id'] == sid)
        if sid == 'female':
            description = 'Encourages one adult female lead vocal'
        card += show_pair(highlights[sid], highlights[sid]['label'], '**Target sound:** ' + description + '.')
    card += '[Explore all 60 Off/On pairs](samples/uni16-gan-v1/README.md). Each featured pair was selected by the recorded score within that control. Selection is not a human listening judgment. Players preserve the recordings’ relative levels; WAV downloads contain the original takes.\n\n'
    card += '## Choose a control\n\nEach download adds one target sound. Female and Male use separate files; negative strength is not a trained opposite. All sixteen are language-model adapters, rank 8 / alpha 8, saved at training update 660.\n\n| Control | Target sound | Download | Comparisons |\n|---|---|---|---|\n'
    for c in concepts:
        description = f'Encourages one adult {c["id"]} lead vocal' if c['id'] in ['female', 'male'] else c['description']
        sidecar = c['weights'].replace('.safetensors', '.json')
        card += f'| {c["label"]} | {description} | [Weights]({c["weights"]}) · [Settings]({sidecar}) | ' + ('Pending' if c['id'] == 'reggaeton' else '4 pairs') + ' |\n'
    card += '\n**Reggaeton:** the trained file is available, but a failed render interrupted its comparison grid. Audio validation is pending.\n\n'
    card += '''## Download and use

Start with **one control at +1** and compare it with Off. Keep your starting caption, lyrics and seed fixed. Each weight file needs its matching JSON settings file alongside it.

```python
from huggingface_hub import snapshot_download

folder = snapshot_download(
    "ntc-ai/minimax-music3-concept-sliders",
    allow_patterns=[
        "catalog.json", "usage.md", "source/lora.py",
        "weights/uni16-gan-v1/female-*/*",  # One voice control to try.
        "prompts/uni16-gan-v1/*",
    ],
)
```

Then follow the [loading example](usage.md) to attach the adapter to your existing Music 3 pipeline. Replace `female-*` with a folder from the table above, or use `weights/uni16-gan-v1/*/*` to download all sixteen controls.

| Adapter | Attach to | Published On setting |
|---|---|---|
| Voice & genre | Language model; the JSON file selects its attention projections. | Direct multiplier `1` |

The studio rescales slider mixtures by host energy, so its fader position can differ from the direct multiplier used here. These downloads use the included native LoRANetwork loader; ComfyUI exports at older paths belong to earlier releases.

[Training and inference project](https://github.com/ntc-ai/sliders-conceptmod) · [Machine-readable catalog](catalog.json) · [Release file hashes](release-manifest.json)

'''
    card += reward_showcase
    card += inline_formulation() + '\n'
    card += '## Earlier releases\n\nOlder uni-lyric, v24 and bipolar weights and recordings remain at their existing paths. They are historical comparisons and do not define the current palette.\n'
    (folder / 'README.md').write_text(card)
    (ROOT / 'docs/hub-readme-uni16-gan.md').write_text(card)
    reward_summary = reward_intro.split('A higher model score')[0].replace('## Listen: reward slider', '## Results at a glance')
    reward_card = '# Reward slider: refined block step 2\n\n' + reward_summary + f'[Download native acoustic LoRA]({URL}{reward_dest}) · [Matching settings file]({URL}{reward_dest.replace(".safetensors", ".json")})\n\n'
    reward_card += 'Rank 8 / alpha 8; `MiniMaxMusic3TransformerBlock`; 216 attention/feed-forward projections. Direct multiplier `1` matches the recorded On treatment. This is an experimental listening release and is not part of the voice/genre palette.\n\n'
    reward_card += f'Checkpoint SHA-256: `{REWARD_SHA}`.\n\n'
    reward_card += '[All fresh Off/On comparisons](../../samples/reward-refined-block/README.md) · [Formulation](../../evidence/uni16-gan-v1/formulation.md) · [Fresh scorecard](../../evidence/reward-refined-block/scorecard.json) · [Preservation report](../../evidence/reward-refined-block/preservation.json)\n\n'
    reward_card += 'The failed dynamics check measures peak level relative to RMS level (crest factor). The first twenty seconds of a take do not establish full-song completion. A higher automated score does not establish human preference. See the linked method for the objective, equations and exact training settings.\n'
    (folder / 'reward/refined-block-step2/README.md').write_text(reward_card)
    r.copy(ROOT / 'conceptmod/textsliders/lora.py', 'source/lora.py')
    r.copy(ROOT / 'docs/hub-native-usage.md', 'usage.md')
    print(f'Encoding {len(r.encodes)} MP3s from verified existing audio', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(encode, r.encodes))
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ['.md', '.json', '.yaml', '.svg']:
            names(p.read_text(), str(p.relative_to(folder)))
    files = {str(p.relative_to(folder)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'release-manifest.json'}
    write(folder / 'release-manifest.json', {'release': '2026-09-09-uni16-gan-refined-reward', 'repo': REPO,
        'weights': {'concept': 16, 'reward': 1}, 'concept_pairs': len(matrix), 'reward_pairs': len(reward_pairs),
        'preview': {'maximum_seconds_per_arm': 20, 'gap_seconds': 1, 'short_takes': 'Use the shorter duration for both excerpts; retain original first draw and label duration.', 'on_multiplier': 1, 'normalization': 'none', 'codec': 'MP3 192 kbps'},
        'showcase': {'controls': SHOWCASE, 'players': 'Separate Off and On recordings, same prompt, lyrics and seed within each pair; original take lengths.', 'reward': 'Four matched Off/On comparisons on the main page', 'banner': 'Actual Metal Off/On first-20-second RMS envelopes at a shared amplitude scale'},
        'name_validation': 'Existing project denylist and shorthand validator passed for prompts, metadata and published text; sound descriptions manually reviewed.',
        'reward_confirmed': False, 'files': files})
    print(json.dumps({'files': len(files)+1, 'bytes': sum(x['bytes'] for x in files.values()), 'concept_pairs': len(matrix), 'reward_pairs': len(reward_pairs)}, indent=2), flush=True)


def validate(folder):
    manifest = read(folder / 'release-manifest.json')
    for name, v in manifest['files'].items():
        assert sha(folder / name) == v['sha256'], name
    card = (folder / 'README.md').read_text()
    front = yaml.safe_load(card.split('---', 2)[1])
    assert not front.get('widget'), 'Showcase comparisons belong in the paired players'
    pairs = read(folder / 'samples/uni16-gan-v1/pairs.json')['pairs']
    featured = {p['id']: p for p in pairs if p['featured']}
    reward_pairs = read(folder / 'samples/reward-refined-block/pairs.json')['pairs']
    expected = [featured[sid][arm] for sid in SHOWCASE for arm in ['off', 'on']]
    expected += [p[arm] for p in reward_highlights(reward_pairs) for arm in ['off', 'on']]
    assert card.index('## Voice & genre') < card.index('## Choose a control') < card.index('## Download and use') < card.index('## Experimental reward slider')
    assert re.findall(r'<audio[^>]+src="([^"]+)"', card) == [URL + arm['mp3'] for arm in expected]
    for arm in expected:
        path = folder / arm['mp3']
        duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(path)], text=True))
        assert abs(duration - arm['duration_s']) < 0.2, (path, duration)
    for md in folder.rglob('*.md'):
        for href in re.findall(r'\]\(([^)]+)\)', md.read_text()):
            if href.startswith(URL):
                target = folder / href[len(URL):]
            elif href.startswith(('http:', 'https:', '#')):
                continue
            else:
                target = md.parent / href.split('#')[0]
            assert target.exists(), (md, href)
        for src in re.findall(r'src="([^"]+)"', md.read_text()):
            assert src.startswith(URL) and (folder / src[len(URL):]).exists(), (md, src)
    print('Verified file hashes, card links, paired Off/On sources and original player durations.', flush=True)


def publish(folder, parent):
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import CommitOperationAdd, HfApi, ModelCard
    validate(folder)
    ModelCard((folder / 'README.md').read_text()).validate()
    api = HfApi()
    assert api.model_info(REPO).sha == parent, 'Remote changed; review the new head before publishing'
    operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(folder)), path_or_fileobj=str(p))
                  for p in sorted(folder.rglob('*')) if p.is_file()]
    result = api.create_commit(repo_id=REPO, repo_type='model', operations=operations, parent_commit=parent,
        commit_message='Publish UNI16 GAN palette and separate refined reward slider with Off/On showcases', num_threads=4)
    write(folder.parent / 'published.json', {'repo': REPO, 'commit': result.oid, 'url': result.commit_url})
    print(result.commit_url, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'validate', 'publish'])
    parser.add_argument('--folder', type=Path, default=Path('/tmp/hf-music3-release-20260909/package'))
    parser.add_argument('--parent')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.action == 'build':
        build(args.folder)
        validate(args.folder)
    elif args.action == 'validate':
        validate(args.folder)
    else:
        if not args.parent:
            parser.error('publish requires the reviewed --parent commit')
        publish(args.folder, args.parent)
