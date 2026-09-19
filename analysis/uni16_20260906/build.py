"""Author and validate the uni16 listening palette; never mutate old prompts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from app.rewriter import _artist_name_hit
from scripts.audit_prompt_leaks import audit_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Original lyric sheets shared across styles, with separate evaluation sheets.
# Their subjects carry no genre or singer-identity cue.
LYRICS = [
    '[verse]\nThe corner shop has changed its hours\nI leave my bicycle outside\nYour folded note is in my pocket\nThe traffic clears on either side\n[chorus]\nThere is a seat beside the window\nI keep it open for a while',
    '[verse]\nWe carry boxes up the stairway\nA wooden chair has lost a screw\nYou draw a map upon the cardboard\nAnd mark the rooms we never knew\n[chorus]\nThe empty walls return our laughter\nA little space for something new',
    '[verse]\nThe bus pulls in beside the market\nI watch the doors release the crowd\nA folded coat rests on my shoulder\nThe crossing signal clicks aloud\n[chorus]\nI take the longer road this evening\nAnd let the quiet settle down',
    '[verse]\nYou put the cups along the counter\nI turn the handle on the screen\nThe table leans against the doorway\nWe find a place to fit between\n[chorus]\nA small adjustment in the morning\nCan change the way the room is seen',
    '[verse]\nThe station clock is running early\nMy paper ticket hides the date\nI hear a cart across the platform\nAnd lean my shoulder on the gate\n[chorus]\nAnother train will cross the river\nThere is a little time to wait',
    '[verse]\nA loose thread catches on my jacket\nThe kitchen radio turns low\nYou set a bowl beside the basket\nAnd leave the garden boots below\n[chorus]\nWe measure flour without a recipe\nAnd watch the evening shadows grow',
    '[verse]\nThe painted number leaves the mailbox\nA crooked nail holds on the frame\nWe take a brush out to the pavement\nAnd give the little door a name\n[chorus]\nThe street is changing by the hour\nBut there is room for us to stay',
    '[verse]\nI find your glove beneath the cushion\nThe missing button turns up too\nA small receipt falls from the lining\nFor something neither of us knew\n[chorus]\nWe put the scattered things together\nAnd leave a little room to move',
]

# Each sound has a distinct rhythmic/instrumental identity. No language, accent,
# singer gender, exact key, lyrical topic or target BPM is encoded in a genre.
SOUNDS = [
    dict(id='pop', label='Pop', genre='contemporary pop', bpm=[92,104,116,124],
         description='Clear hooks, crisp drums and a polished chorus',
         texture='A clear, polished studio sound with a firm low end and a wide chorus.',
         verse='A compact melodic hook answers the vocal over a tight kick, dry snare, rounded bass and short keyboard chords.',
         chorus='Bright synth layers and rhythmic guitar widen the hook; the drums stay crisp and leave the lead clearly in front.'),
    dict(id='hiphop', label='Hip-Hop', genre='hip-hop with trap drums', bpm=[78,90,132,144],
         description='Rapped verses, deep sub bass and nimble hats',
         texture='A spacious drum-led mix with deep controlled sub bass and close vocal detail.',
         verse='A sparse minor piano figure sits above deep sub bass, a dry snare and syncopated closed hats with brief rolls. The beat allows a half-time feel at faster tempos.',
         chorus='The bass phrase becomes more insistent and a short keyboard response marks the hook while the drum pattern stays spacious.',
         delivery='Rhythmic rapped verses with clear consonants and short breaths; a compact melodic hook in the chorus.'),
    dict(id='rnb', label='R&B', genre='contemporary R&B', bpm=[72,86,98,110],
         description='Warm keys, deep pocket and fluid vocal phrasing',
         texture='A warm intimate studio mix, smooth low end and short restrained vocal ambience.',
         verse='Electric piano chords with extended harmony move around a rounded bass line; a soft kick and rim snare settle behind the beat.',
         chorus='A light pad and clean guitar answers add width while the syncopated bass and relaxed drum pocket remain clear.',
         delivery='Fluid melodic singing with relaxed phrasing behind the beat and brief restrained runs at phrase endings.'),
    dict(id='indie-rock', label='Indie Rock', genre='indie rock', bpm=[92,108,122,136],
         description='Chiming guitars, moving bass and a human drum kit',
         texture='An immediate band recording with audible drum-room depth and clear guitar separation.',
         verse='Chiming electric guitar figures interlock with a moving electric bass line and a close live drum kit; small timing variations keep the groove human.',
         chorus='A second lightly overdriven rhythm guitar broadens the chords, the drummer opens the hats and adds a short tom pickup.'),
    dict(id='pop-punk', label='Pop Punk', genre='pop punk', bpm=[116,136,156,176],
         description='Palm-muted power chords and driving chorus drums',
         texture='A punchy guitar-band mix with tight low end, bright snare attack and an open chorus.',
         verse='Palm-muted electric power chords lock to picked bass and a driving live kick and snare pattern.',
         chorus='The guitar mutes open into wide power chords; crash accents, an active bass line and concise fills push the melodic hook forward.'),
    dict(id='metal', label='Metal', genre='modern melodic metal', bpm=[88,108,132,156],
         description='Heavy guitar riffs, tight kicks and big melodic choruses',
         texture='A heavy but clearly separated mix with controlled low guitar weight and defined drum transients.',
         verse='Low-tuned distorted guitars play a precise palm-muted riff with bass following its accents; tight kick patterns and a firm snare make space for the vocal.',
         chorus='The riff opens into sustained heavy chords and a melodic guitar counterline; brief double-kick bursts lead into the next phrase.'),
    dict(id='country', label='Country', genre='contemporary country', bpm=[76,92,108,124],
         description='Acoustic strum, twangy fills and an easy backbeat',
         texture='A warm clear band mix with natural strings, a close lead and modest room ambience.',
         verse='Steel-string acoustic strums mark the pulse over electric bass and a relaxed kick and snare; a clean electric guitar adds short twangy answers.',
         chorus='Pedal-steel swells and a wider acoustic strum lift the chords; the backbeat remains steady and the instrumental fills fit between vocal phrases.'),
    dict(id='acoustic-folk', label='Acoustic Folk', genre='acoustic folk', bpm=[66,78,92,106],
         description='Fingerpicked strings and a warm small-room performance',
         texture='A close natural room recording with audible string detail and gentle dynamics.',
         verse='Fingerpicked acoustic guitar carries the harmony with a soft upright bass foundation and light brushed percussion.',
         chorus='A second acoustic strum and a small piano response add movement while the arrangement stays open around the lead.'),
    dict(id='house', label='House', genre='vocal house', bpm=[112,120,126,132],
         description='Steady club kick, offbeat hats and a rolling bass line',
         texture='A clean spacious club mix with a steady low-end pulse and controlled sidechain movement.',
         verse='A four-on-the-floor kick supports offbeat open hats, handclaps and a rolling syncopated bass line; short piano stabs answer the vocal.',
         chorus='A broad chord layer and brighter percussion open the hook; a brief filtered drum pickup returns smoothly to the full groove.'),
    dict(id='disco-funk', label='Disco Funk', genre='disco funk', bpm=[96,108,116,124],
         description='Elastic bass, clipped guitar and bright dance-floor strings',
         texture='A bright warm dance-band mix with lively bass articulation and crisp percussion.',
         verse='An elastic electric bass line moves under clipped sixteenth-note rhythm guitar, a steady kick, snare and open hats; hand percussion adds syncopation.',
         chorus='Short string swells and brass-like keyboard accents lift the chords while the bass and guitar keep their interlocking dance groove.'),
    dict(id='kpop', label='K-pop', genre='K-pop production', bpm=[90,104,118,132],
         description='Sharp synth hooks, tight edits and a big chorus lift',
         texture='A detailed glossy mix with precise drum edits, deep bass and bright separated synth layers.',
         verse='A distinctive short synth hook, clipped electronic drums and sub bass alternate with small spaces around the solo lead.',
         chorus='The hook expands into wide synth chords, extra percussion and rhythmic bass; a brief drum stop announces the chorus without interrupting the sung line.'),
    dict(id='reggaeton', label='Reggaeton', genre='reggaeton', bpm=[84,94,104,114],
         description='Dembow drums, rounded sub bass and clipped melodic hooks',
         texture='A clear percussive mix with a rounded sub foundation and close dry lead.',
         verse='A repeating dembow kick and snare pattern drives rounded sub bass, clipped keyboard plucks and light hand percussion.',
         chorus='An answering synth hook and extra shaker widen the groove while the dembow accents stay unmistakable.'),
    dict(id='afrobeats', label='Afrobeats', genre='contemporary Afrobeats', bpm=[88,100,110,120],
         description='Interlocking percussion, melodic bass and buoyant guitar',
         texture='A warm airy mix with clear percussion detail, rounded low end and a relaxed lead.',
         verse='Syncopated kick and rim clicks interlock with shakers and hand percussion; a melodic bass line and short clean-guitar plucks create a buoyant rolling groove.',
         chorus='A bright keyboard response and another light percussion pattern lift the hook while the guitar and bass keep their dancing conversation.'),
    dict(id='lofi', label='Lo-fi', genre='lo-fi soul pop', bpm=[64,76,88,100],
         description='Soft swung drums, mellow keys and gentle tape warmth',
         texture='An intimate mellow mix with gently softened transients and light tape saturation; the lead remains intelligible.',
         verse='Soft swung kick and snare sit under warm electric piano voicings and a rounded bass line; a quiet guitar figure fills the spaces.',
         chorus='A muted pad and a small chord variation add warmth while the laid-back beat keeps its soft pocket.'),
]

CONTEXTS = [
    ('contemporary song', 'piano chords', 'electric bass', 'a steady light drum kit'),
    ('guitar-led song', 'clean electric guitar chords', 'electric bass', 'a close drum kit'),
    ('keyboard-led song', 'soft keyboard chords', 'rounded synth bass', 'a simple programmed beat'),
    ('acoustic song', 'acoustic guitar chords', 'upright bass', 'light hand percussion'),
    ('small-band song', 'warm organ chords', 'electric bass', 'a dry rim-click drum pattern'),
    ('piano-led song', 'upright piano chords', 'muted electric bass', 'brushed drums'),
    ('electronic song', 'short synth chords', 'soft synth bass', 'a sparse electronic beat'),
    ('string-led song', 'plucked acoustic strings', 'acoustic bass', 'a shaker and soft hand drum'),
]


def caption(genre, bpm, voice, delivery, texture, verse, chorus):
    return (f'Global Metadata:\nGenre: {genre}. BPM {bpm}. Meter: 4/4. '
            f'The verse is focused and the chorus opens into a fuller sound. {texture}\n'
            f'Vocal Details:\n{voice} {delivery} A clear solo lead stays in front throughout, '
            'with natural vocal detail and light room ambience.\n'
            f'Arrangement:\nVerse: {verse}\nChorus: {chorus}')


def rows_for(item, evaluation=False):
    rows = []
    for i in range(4):
        j = i + (4 if evaluation else 0)
        genre, chords, bass, drums = CONTEXTS[j]
        bpm = item['bpm'][i] + (2 if evaluation else 0)
        gender = None if item['id'] in ('male','female') else [None,'female','male',None][i]
        voice = f'One adult {gender + " " if gender else ""}lead vocalist.'
        delivery = 'Melodic singing with clear words and natural phrasing.'
        texture = 'A balanced studio mix with a clear lead, defined bass and modest room depth.'
        verse = f'{chords.capitalize()} carry the harmony over {bass} and {drums}.'
        chorus = f'The {chords} broaden and the rhythm adds a light accent at the section change, leaving space for the lead.'
        neutral = caption(genre,bpm,voice,delivery,texture,verse,chorus)
        if item['id'] in ('male','female'):
            positive = caption(genre,bpm,f'One adult {item["id"]} lead vocalist.',delivery,texture,verse,chorus)
        else:
            positive = caption(item['genre'],bpm,voice,item.get('delivery',delivery),item['texture'],item['verse'],item['chorus'])
        rows.append(dict(target=neutral,neutral=neutral,negative=neutral,positive=positive,lyrics=LYRICS[j]))
    return rows


def validate_document(path, item, evaluation=False):
    d = yaml.safe_load(path.read_text())
    assert all('/' not in d[k] and '\\' not in d[k] for k in ('plus_label','minus_label')), 'Labels must be single filename components'
    assert d['minus_label'] == 'Off' and d['recommended_range'] == [0.,1.]
    assert len(d['rows']) == 4
    assert not _artist_name_hit('',json.dumps(d,ensure_ascii=False)), path
    assert not audit_file(path), audit_file(path)
    for row in d['rows']:
        assert row['target'] == row['neutral'] == row['negative']
        assert row['positive'] != row['neutral']
        for cell in ('positive','neutral'):
            assert all(row[cell].count(h)==1 for h in ('Global Metadata:','Vocal Details:','Arrangement:'))
            assert 'Verse:' in row[cell] and 'Chorus:' in row[cell]
        if item['id'] in ('male','female'):
            assert row['positive'].replace(f'One adult {item["id"]} lead vocalist.','One adult lead vocalist.') == row['neutral']
        else:
            for gender in ('male','female'):
                phrase=f'One adult {gender} lead vocalist.'
                assert (phrase in row['positive']) == (phrase in row['neutral'])
    return dict(path=str(path),sha256=sha(path),rows=4,validation='passed')


def main():
    if (WORK/'manifest.json').exists():
        raise SystemExit('Campaign already frozen; create a new version to change prompts.')
    items = [dict(id=g,label=g.title(),description=f'One adult {g} lead, with the song\'s phrasing and timbre intact',bpm=[90,118,104,76]) for g in ('female','male')] + SOUNDS
    assert len(items) == len({x['id'] for x in items}) == 16
    audit=[]
    for item in items:
        for split in ('train','eval'):
            path=WORK/'prompts'/f'prompts-{item["id"]}-uni-16-v1-{split}.yaml'
            data=dict(plus_label=item['label'],minus_label='Off',recommended_range=[0.,1.],
                      slider_positive=item['description'],slider_negative='The same song with the style control off.',
                      rows=rows_for(item,split=='eval'))
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True,width=105))
            item[split+'_prompts']=str(path)
            audit.append(validate_document(path,item,split=='eval'))
        assert not set(r['lyrics'] for r in rows_for(item)) & set(r['lyrics'] for r in rows_for(item,True))
    old=json.loads((ROOT.parent/'app/sliders.json').read_text())
    current=[]
    for s in old['sliders']:
        component=s['components'][0]
        sidecar=ROOT/'models'/Path(component['weights']).with_suffix('.json')
        d=json.loads(sidecar.read_text())
        prompts=Path(d['prompts_file'])
        if not prompts.is_absolute():prompts=ROOT/prompts
        p=yaml.safe_load(prompts.read_text())
        current.append(dict(id=s['id'],label=s['label_plus'],weights=component['weights'],
            steps=d['steps'],method=d.get('lm_target'),prompts=str(prompts),rows=len(p['rows']),
            adversarial=d.get('adv',{}).get('enabled',False),
            name_validation='rejected_do_not_reuse' if _artist_name_hit('',json.dumps(p,ensure_ascii=False)) else 'passed'))
    write(WORK/'current-catalog-audit.json',current)
    write(WORK/'prompt-audit.json',audit)
    write(WORK/'catalog.json',dict(version='uni16-gan-v1',status='authored_requires_training_and_audio_review',
        audience='Broad mix of mainstream and genre listeners; provisional product selection, not a measured preference ranking',
        sliders=items,
        combinations=[dict(label='Warm dance pop',sliders={'pop':.55,'disco-funk':.45}),
                      dict(label='Intimate R&B',sliders={'rnb':.65,'lofi':.35}),
                      dict(label='Country rock',sliders={'country':.65,'indie-rock':.35}),
                      dict(label='Afro pop',sliders={'afrobeats':.65,'pop':.35}),
                      dict(label='Club pop',sliders={'house':.65,'pop':.35}),
                      dict(label='Heavy melodic guitars',sliders={'metal':.65,'pop-punk':.35})],
        combination_status='Authored listening hypotheses; combination audio and safe summed strengths are not yet established',
        vocal_controls='Choose at most one of male/female. Both off preserves the prompt. Neither control sets lyrics, language or genre.'))
    print('Authored 16 sliders, 64 training rows and 64 disjoint evaluation rows. All prompt checks passed.')


if __name__ == '__main__':
    main()
