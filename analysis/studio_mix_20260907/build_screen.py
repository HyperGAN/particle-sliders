"""Freeze a 25-render studio mixing screen; this does not launch GPU work."""
import json
from pathlib import Path

import yaml

from geometry import APP_ROOT, WORK, sha
from app import sliders
from app.rewriter import _artist_name_hit


def main():
    fixture = WORK.parent/'uni16_20260906/prompts/prompts-female-uni-16-v1-eval.yaml'
    row = yaml.safe_load(fixture.read_text())['rows'][2]
    pairs = [('female', 'pop'), ('country', 'indie-rock'), ('house', 'acoustic-folk')]
    targets = [
        ('Pop song', 'One adult female lead vocalist.',
         'Crisp kick and snare support a rounded bass line, bright keyboard chords and a short clear hook.',
         'The hook opens into wider chords and stronger drum accents, leaving the lead clear.'),
        ('Country and indie rock song', 'One adult lead vocalist.',
         'Steel-string acoustic strum and chiming electric guitar share the harmony over moving electric bass and a human drum kit; short twangy fills answer the lead.',
         'The guitars widen, a brief pedal-steel swell answers the hook, and the drum backbeat grows fuller.'),
        ('Acoustic folk and house song', 'One adult lead vocalist.',
         'Fingerpicked acoustic guitar carries the harmony over a steady four-on-the-floor kick, offbeat hats and a rolling bass line; the close strings retain natural detail.',
         'A second acoustic strum and broad keyboard chords lift the hook while the club pulse stays steady.'),
    ]
    jobs = []

    def add(label, settings, energy, caption=None):
        settings = sliders.validate_settings(settings)
        job = dict(id=f'{len(jobs):02d}', label=label, sliders=settings,
                   energy=dict(language_model=energy), seed=101, requested_duration=20,
                   caption=caption or row['neutral'], lyrics=row['lyrics'],
                   lora_components=sliders.resolve(settings, host_energy={'language_model': energy}))
        if _artist_name_hit('', json.dumps(job, ensure_ascii=False)):
            raise ValueError('Screen failed name validation')
        jobs.append(job)

    add('Off', [], 0)
    for slider_id in sorted({i for pair in pairs for i in pair}):
        for energy in (1., 2.):
            add(f'{slider_id} solo energy {energy:g}', [dict(id=slider_id, scale=1)], energy)
    for (a, b), (genre, voice, verse, chorus) in zip(pairs, targets):
        for energy in (2., 2.5, 2.8):
            add(f'{a} + {b} equal mix energy {energy:g}',
                [dict(id=a, scale=.5), dict(id=b, scale=.5)], energy)
        caption = (f'Global Metadata:\nGenre: {genre}. BPM 106. Meter: 4/4. '
                   'The verse is focused and the chorus opens into a fuller sound. '
                   'A balanced studio mix with a clear lead, defined bass and modest room depth.\n'
                   f'Vocal Details:\n{voice} Melodic singing with clear words and natural phrasing. '
                   'A clear solo lead stays in front throughout, with natural vocal detail and light room ambience.\n'
                   f'Arrangement:\nVerse: {verse}\nChorus: {chorus}')
        add(f'{a} + {b} caption reference', [], 0, caption)
    assert len(jobs) == 25
    paths = {c['weights'] for j in jobs for c in j['lora_components']}
    result = dict(
        status='planned_not_rendered', physical_gpu=1,
        render_entrypoint='app.generator.generate; default studio merge mode',
        fixture=dict(path=str(fixture), sha256=sha(fixture), row=2, seed=101),
        confirmation=dict(row=3, seed=303, status='reserved'),
        registry_sha256=sha(sliders.REGISTRY_PATH),
        builder_sha256=sha(__file__),
        source_sha256={str(APP_ROOT/'app'/p): sha(APP_ROOT/'app'/p)
                       for p in ('generator.py', 'lora_runtime.py', 'sliders.py')},
        checkpoints=[dict(weights=p, sha256=sha(p),
                          sidecar_sha256=sha(sliders._sidecar_path(Path(p)))) for p in sorted(paths)],
        interpretation=('Initial diagnostic only. Solo energy 1 matches each contribution in an equal mix at energy 2; '
                        'solo energy 2 matches the whole studio budget. Mix energies 2.5/2.8 test gain alone. '
                        'Caption references specify achievable combined behavior, not a numeric merge target. '
                        'Preserve full audio, observed duration and first failures; compare aligned 20-second excerpts. '
                        'One fixture and seed cannot establish a production default.'),
        jobs=jobs)
    (WORK/'screen.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Prepared {len(jobs)} jobs; no GPU work launched.')


if __name__ == '__main__':
    main()
