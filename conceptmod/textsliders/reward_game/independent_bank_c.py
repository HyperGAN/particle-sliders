"""Fresh replication fixtures prepared before complete-flow candidate training."""
from ..reward_sliders.specs import validate_families

SEEDS = (327673, 393241)


def families():
    settings = [
        (92, 'female', 'Plucked upright bass, a small brushed kit, mellow vibraphone and short muted trumpet answers. The close lead sings with an easy swing and clear consonants in a small dry room.', {},
         ['The paper crane has lost a corner', 'I fold another from the blue', 'You clear a place beside the saucer', 'And show me where the edges meet'],
         ['Leave a fold beside the window', 'Let the morning find it there']),
        (98, 'female', 'Soft electric piano chords, a rounded electronic bass, a sparse rim-click groove and clipped nylon guitar notes. An intimate melodic lead leaves space between phrases; a quiet harmony joins the refrain.', {'female': .8, 'rnb': 1.2},
         ['The lift is waiting on the ground floor', 'A yellow glove is on the rail', 'I press the button with my elbow', 'And watch the numbers turn to pale'],
         ['Every floor has someone waiting', 'Every door can open wide']),
        (132, 'male', 'A compact acoustic kit, firm picked bass, crunchy open electric guitar chords and a thin tremolo guitar response. A low sung lead stays intelligible as the refrain opens into longer notes.', {'male': .8, 'indie-rock': 1.2},
         ['The gate is leaning toward the garden', 'Its rusty latch has come undone', 'We bring a hammer from the cellar', 'And square the hinges in the sun'],
         ['Let the path stay open longer', 'Let the old gate swing again']),
        (106, 'male', 'A softly shuffled drum kit, fingerpicked acoustic guitar, plucked bass and low resonator guitar fills. Warm plain lead singing with gentle rasp, short phrases and a small amount of room sound.', {'male': .8, 'country': 1.2},
         ['The flour settles on the counter', 'A wooden spoon is in the bowl', 'You draw a circle through the middle', 'And leave a space to turn the dough'],
         ['Keep the table near the fire', 'Keep a little flour for me']),
        (118, 'unspecified', 'A steady dance kick, crisp offbeat hats, a syncopated electric bass, short rhythm guitar strokes and narrow bright synth chords. A clearly sung refrain answers the bass motif with playful timing.', {'house': 1., 'disco-funk': 1.},
         ['The paint is drying on the signboard', 'We prop it up against a chair', 'You add a line beneath the arrow', 'And brush a fleck out of your hair'],
         ['There is room beside the lettering', 'There is time to get it right']),
        (82, 'unspecified', 'Quiet strummed acoustic guitar, a steady low cello pulse, brushed snare and a breathy wooden flute response. A natural melodic voice carries distinct words, with a restrained low harmony in the refrain.', {'acoustic-folk': 1., 'country': 1.},
         ['The rain has filled the empty flowerpot', 'A little branch lies on the stone', 'We tip the water by the stairway', 'And bring the folding chairs back home'],
         ['When the clouds have crossed the courtyard', 'We can set the chairs outside']),
        (128, 'instrumental', 'A rubbery monophonic synth sequence, deep sustained bass, a dry electronic kick and metallic percussion. A rounded organ plays a contrasting answer; vary the rhythm while keeping the melodic pattern recognizable.', {}, [], []),
        (112, 'instrumental', 'A woody marimba theme above plucked double bass, soft toms and short chamber-string responses. Alternate the mallet melody with a warmer low-register string answer, building a clear acoustic ensemble groove.', {}, [], []),
    ]
    rows = []
    for index, (bpm, voice, instruments, styles, verse, refrain) in enumerate(settings):
        vocal = ('Instrumental from beginning to end, without singing, spoken words, humming or vocal samples.'
                 if voice == 'instrumental' else
                 f"One adult {'lead' if voice == 'unspecified' else voice+' lead'} vocalist sings clearly in a comfortable natural register.")
        caption = f'Global Metadata:\nA developing musical arrangement at {bpm} BPM in 4/4. Introduce a distinct motif and shape the instrumental answers around it.\nVocal Details:\n{vocal}\nArrangement:\n{instruments}'
        if index == 6:
            lyrics = '[metallic percussion opening]\n[rubbery sequence motif]\n[organ answer over bass]\n[sequence rhythm variation]\n[low bass and percussion passage]\n[organ and synth exchange]\n[full rhythm return]\n[electronic instrumental cadence]'
        elif index == 7:
            lyrics = '[woody mallet opening]\n[marimba theme with plucked bass]\n[low string answer]\n[soft tom variation]\n[mallet and strings exchange]\n[bass pulse passage]\n[chamber ensemble return]\n[acoustic instrumental cadence]'
        else:
            lyrics = '[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[outro]'
        rows.append(dict(family=f'independent-bank-c-{index:02d}', split='test',
            title=f'Fresh arrangement C{index:02d}', voice=voice, bpm=bpm,
            caption=caption, lyrics=lyrics, style_multipliers=styles))
    validate_families(rows)
    return rows
