"""Original, sound-only pilot fixtures. Splits are fixed before any scoring."""
from .specs import validate_families


# Instrument, density, time-feel, and voice describe audible properties only.
DESIGNS = [
    ('acoustic', 'sparse', 78, 'female', 'Fingerpicked steel strings, upright bass and brushed snare', {}),
    ('electronic', 'dense', 126, 'unspecified', 'Short bright synth chords, firm kick, open hats and layered bass', {'female': 1.2, 'pop': .8}),
    ('acoustic', 'dense', 118, 'male', 'Strummed strings, twangy fills, chiming electric guitar and a close live kit', {'country': 1.2, 'indie-rock': .8}),
    ('electronic', 'sparse', 92, 'male', 'Warm electric piano, soft sub notes and dry syncopated drum taps', {'lofi': 1.}),
    ('acoustic', 'sparse', 104, 'instrumental', 'Nylon strings trade short phrases with bowed bass and gentle shakers', {}),
    ('electronic', 'dense', 124, 'female', 'A four beat kick under fingerpicked strings, rolling bass and offbeat hats', {'house': 1., 'acoustic-folk': 1.}),
    ('acoustic', 'dense', 144, 'unspecified', 'Muted power chords, moving bass and a tight fast drum kit', {'pop-punk': 1.}),
    ('electronic', 'sparse', 72, 'unspecified', 'Low round bass, a slow broken beat and a single wavering synth motif', {}),
    ('electronic', 'dense', 120, 'female', 'Crisp programmed drums, bright keyboard hooks and sustained bass', {'female': 1.2, 'pop': .8}),
    ('acoustic', 'sparse', 82, 'male', 'Close fingerpicked guitar, woody bass and a light brushed backbeat', {}),
    ('acoustic', 'dense', 132, 'instrumental', 'Clipped rhythm guitar, elastic electric bass and a bright human kit', {'disco-funk': 1.}),
    ('electronic', 'sparse', 96, 'unspecified', 'A dry hand drum loop, soft synth plucks and low rounded bass', {'house': 1., 'acoustic-folk': 1.}),
    ('acoustic', 'dense', 116, 'male', 'Steady acoustic strumming, twangy electric fills and a roomy backbeat', {'country': 1.2, 'indie-rock': .8}),
    ('electronic', 'sparse', 80, 'female', 'Muted electric piano, short synth bass and a restrained broken beat', {}),
    ('electronic', 'dense', 138, 'instrumental', 'Interlocking synth arpeggios, a deep pulse and crisp programmed percussion', {'house': 1.}),
    ('acoustic', 'sparse', 100, 'unspecified', 'A dry plucked guitar, upright bass and light wooden percussion', {'acoustic-folk': 1.}),
]

VERSES = [
    ('I mend the pocket by the window', 'A little thread can hold the day', 'The kettle trembles on the burner', 'I let the morning find its way'),
    ('The lift is stuck above the lobby', 'We take the stairs with bags of bread', 'You count the steps in crooked numbers', 'I hum the ones you leave unsaid'),
    ('A broken latch beside the garden', 'Still clicks against the leaning gate', 'We set our coats upon the railing', 'And give the rust another day'),
    ('The last receipt is in my jacket', 'The ink has faded into blue', 'I bought a pear and missed the tram stop', 'Then walked the long way home with you'),
    ('', '', '', ''),
    ('The spare key hides beneath a flower', 'Its edges cool against my palm', 'We move the chairs into the hallway', 'And make a little room for calm'),
    ('The cracked wheel rattles down the pavement', 'We race the shadows to the bend', 'You catch the box before it tumbles', 'And laugh until the streetlights end'),
    ('The small clock skips a quiet second', 'The ceiling gathers passing rain', 'I draw a square upon the table', 'And start the count from one again'),
    ('A paper boat beside the gutter', 'Is waiting for the rain to rise', 'We nudge it free with fallen branches', 'And follow where the water slides'),
    ('The window sticks in colder weather', 'I lift the frame and hold it wide', 'A distant bell comes through the curtains', 'And settles by the chair inside'),
    ('', '', '', ''),
    ('Your mitten falls beside the station', 'I tuck it in the outer fold', 'The timetable has lost its corners', 'We share the bench against the cold'),
    ('The porch step rocks beneath the bucket', 'A little water finds the stone', 'We watch the line creep through the gravel', 'And give the thirsty roots a home'),
    ('A borrowed lamp beside the mattress', 'Throws little circles on the wall', 'We move the shade a finger closer', 'And watch the tallest shadow fall'),
    ('', '', '', ''),
    ('The toolbox waits beside the doorway', 'A silver screw rolls to my shoe', 'We fix the chair with uneven footsteps', 'And leave a place beside it too'),
]

REFRAINS = [
    ('Pull the loose end gently', 'Let the little seam stay strong'),
    ('One more floor together', 'There is rhythm in the climb'),
    ('Leave the gate a little wider', 'There is room for us to pass'),
    ('Let the road unfold between us', 'We have time to take it slow'),
    ('', ''),
    ('Set the small things down beside me', 'Let the open room reply'),
    ('Keep the loose wheel turning', 'We can reach the bend in time'),
    ('Count the quiet after thunder', 'Let the empty second stay'),
    ('Carry what the rain can carry', 'Let the folded paper go'),
    ('Hold the wooden frame a moment', 'Let the outside wander in'),
    ('', ''),
    ('Keep a little warmth between us', 'Till the waiting turns to miles'),
    ('Follow where the water wanders', 'Every root can find a drink'),
    ('Move the circle slowly closer', 'Let the corners come to rest'),
    ('', ''),
    ('Every leg can learn to settle', 'Every seat can make some room'),
]


def pilot():
    rows = []
    for index, ((source, density, bpm, voice, instruments, styles), verse, refrain) in enumerate(zip(DESIGNS, VERSES, REFRAINS)):
        lead = ('Instrumental throughout; no singing or spoken words.' if voice == 'instrumental' else
                f"One adult {'lead' if voice == 'unspecified' else voice+' lead'} vocalist. Clear melodic words, natural breath and unhurried phrase endings.")
        caption = (f'Global Metadata:\nA {source} arrangement at {bpm} BPM in 4/4, with a {density} texture. '
                   'A steady pulse and a coherent verse-to-chorus development.\n'
                   f'Vocal Details:\n{lead}\nArrangement:\n{instruments}. '
                   'Begin with the main motif, develop it in the verse and open the harmony in the chorus. '
                   'Keep a clear foreground, defined bass and natural room depth.')
        lyrics = (f'[instrumental]\n[intro]\n[verse]\n[chorus]\n[bridge]\n[outro]\n[instrumental {index}]'
                  if voice == 'instrumental' else
                  '[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain+refrain)+
                  '\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain+refrain)+'\n[outro]')
        rows.append(dict(family=f'pilot-{index:02d}', split='train' if index < 8 else 'dev' if index < 12 else 'test',
                         title=f'Reward fixture {index:02d}', caption=caption, lyrics=lyrics,
                         source=source, density=density, bpm=bpm, voice=voice, style_multipliers=styles))
    validate_families(rows)
    return rows
