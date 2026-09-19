"""New musical families with splits declared before scoring."""
from ..reward_sliders.specs import validate_families


TEXTS = [
 ('The bakery is closing early|I brush the flour from my sleeve|You wrap the loaf in folded paper|And hold the door before I leave', 'Save the end for morning|Let the kitchen smell of bread'),
 ('A loose tile rings beneath my heel|The tunnel carries every sound|I tap a second little answer|And hear it travel underground', 'Send a rhythm down the passage|Let it find another wall'),
 ('The laundrette is almost empty|A red scarf circles in the glass|You read the weather from a leaflet|I watch the folded shadows pass', 'Round and round the colours travel|Then we carry warmth back home'),
 ('A wooden bird above the counter|Has lost the paint along its wing|We give it blue upon a Sunday|And wait for nothing but the spring', 'Set it where the light can reach it|Give the quiet bird a sky'),
 ('The ferry rope is wet and heavy|I pull the knot across the rail|You hold the ticket in your jacket|And watch the gulls beyond the sail', 'Let the far bank come to meet us|Let the water keep its pace'),
 ('A copper pan beneath the staircase|Has caught a patient drop of rain|We move it closer to the doorway|And hear the single note again', 'One small sound can fill a hallway|One clear note can mark the hour'),
 ('The market clock is running slowly|The fruit seller has closed the shade|We split a peach beside the fountain|And count the coins that we have saved', 'Keep the sweetness on your fingers|Keep a moment for the shade'),
 ('A bicycle bell inside the cupboard|Reminds me of a summer hill|I press the lever with my thumb now|The bright ring holds the kitchen still', 'There is distance in a small sound|There is sunlight in the ring'),
 ('The florist ties the stems with ribbon|I choose the ones with crooked leaves|You carry water to the landing|And clear a place beside the keys', 'Let a little green grow near us|Let the narrow hallway bloom'),
 ('A paper map beside the compass|Has folded roads across the sea|We trace the line along the shoreline|And leave the evening journey free', 'Take the bend beside the water|We can choose the turning there'),
 ('The tennis net is loose and heavy|We lift its middle from the clay|A yellow ball rolls to the entrance|I stop it with my foot halfway', 'Raise the middle one more little|Let the afternoon begin'),
 ('An empty jar upon the windowsill|Is full of coloured bits of thread|I wind the longest round a pencil|And mend the cuff before the bed', 'Every colour holds a moment|Every small repair can stay'),
 ('The upstairs neighbour grows tomatoes|A silver basin takes the rain|She brings a handful down on Tuesday|And asks about the windowpane', 'Pass the little bowl between us|There is plenty in a few'),
 ('The station cat has found a suitcase|And settled on the leather strap|We wait until the last announcement|Then lift it gently from its nap', 'There is time before the leaving|There is room to rest a while'),
 ('A painted sign beside the orchard|Has turned its letters to the sun|We carry it beneath the awning|And count the nails before we run', 'Hold the corner while I set it|Let the message meet the road'),
 ('The record shop has moved its shelves|A narrow aisle becomes a square|You find a stool behind the curtain|I set my folded jacket there', 'Make a little space for listening|Let the room receive the sound'),
 ('A wooden spoon beside the mixing bowl|Has left a circle in the flour|We roll the dough across the table|And mark the oven for an hour', 'Give the small loaf time to rise now|Let the waiting warm the room'),
 ('The playground swing is moving slowly|No one sits upon the seat|We hold the chain against the weather|And brush the gravel from our feet', 'Leave the gate upon the catch now|Let the next small footsteps in'),
 ('A row of lamps beside the river|Reflects in patches on the tide|I count a missing light among them|You find its shimmer on the side', 'Look a little past the surface|Let the moving water tell'),
 ('The postbox leans beside the corner|A sparrow balances above|We put the letter through the narrow slot|And hear it settle with a shove', 'Let the little folded message|Find the hands that know its shape'),
 ('A woollen blanket on the balcony|Is moving gently in the heat|You bring the pegs in from the railing|I fold the corners till they meet', 'Hold the warm end just a moment|Let the summer settle in'),
 ('The second-hand coat has a secret|A little marble in the seam|I roll it over on the counter|It holds a tiny line of green', 'Keep the small and shining question|Let the pocket tell its tale'),
 ('A ladder rests beside the apple tree|The lowest branch has bent with fruit|We put a crate beneath the shadow|And shake the dirt from every boot', 'Take the ripest from the lower branch|Leave the higher ones to turn'),
 ('The tram has stopped beside the crossing|A driver wipes the misted glass|We wait beneath a tilted shelter|And watch the yellow windows pass', 'Keep a place beneath the shelter|Let the next long carriage come'),
 ('A tin of chalk beside the doorway|Has left a little dust of blue|We draw a square across the paving|And leave a space to step into', 'Find the middle of the blue lines|Let the pavement be a game'),
 ('The small repair shop shuts at seven|A desk lamp lights a broken phone|You test the cord beside the counter|I wait until the dial has tone', 'Let the little wire connect us|Let the quiet find a voice'),
 ('A folded tent beside the doorway|Still smells of grass and morning rain|We spread it wide across the landing|And count the small brass pegs again', 'Bring the outside through the doorway|Let the travelling days stay near'),
]

# Four voices per training block, then balanced dev, teacher check and final blocks.
DESIGNS = [
 (82,'female','Soft piano, brushed snare, bowed bass and a close small-room lead',{}),
 (130,'male','Chopped electric guitar, octave bass and dry dance drums',{'male':.8,'disco-funk':1.2}),
 (98,'unspecified','Warm electric keys, short sub notes and staggered hand percussion',{'lofi':1.}),
 (138,'instrumental','Interwoven synth arpeggios, a firm kick and sharply separated hats',{'house':1.}),
 (110,'female','Chiming guitars, reed keyboard, loose human drums and restrained room reflections',{'female':.8,'indie-rock':1.2}),
 (72,'male','Fingerpicked nylon strings, upright bass and a quiet brush pattern',{'acoustic-folk':1.}),
 (124,'unspecified','Rounded synth chords, plucked guitar and a syncopated low drum',{'house':1.,'acoustic-folk':1.}),
 (90,'instrumental','Muted trumpet phrases, tremolo keys, double bass and scattered rim clicks',{}),
 (148,'female','Tight palm-muted guitars, quick live snares and melodic bass',{'female':.8,'pop-punk':1.2}),
 (104,'male','Steel-string strumming, twangy guitar answers and woody live drums',{'country':1.2,'indie-rock':.8}),
 (118,'unspecified','Bright electric keys, interlocking shakers and a supple low bass motif',{'afrobeats':1.}),
 (126,'instrumental','Clipped rhythm guitar, elastic bass, handclaps and a close funk kit',{'disco-funk':1.}),
 (86,'female','Close breathy pitched lead, softly pulsed electric piano and a light broken beat',{'female':1.,'rnb':1.}),
 (122,'male','Low rounded bass, dry offbeat keys and lightly swung electronic percussion',{'male':.8,'reggaeton':1.2}),
 (108,'unspecified','Acoustic guitar ostinato, rim clicks, deep bass and airy sustained keys',{'country':1.,'rnb':1.}),
 (134,'instrumental','Bowed strings, pulsing synth bass, crisp kick and soft wooden ticks',{'house':1.,'lofi':1.}),
 (80,'female','Low piano chords, warm bass and a dry brushed kit under a clear intimate lead',{}),
 (116,'male','Clipped guitar, syncopated bass and taut small-room drums',{'male':.8,'disco-funk':1.2}),
 (100,'unspecified','Nylon guitar, airy flute and interlocking hand percussion',{}),
 (140,'instrumental','Bright rolling synth figures, a low steady kick and clean offbeat hats',{'house':1.}),
 (112,'female','Ringing single-coil guitar, soft reed keys and loose acoustic drums',{'female':.8,'indie-rock':1.2}),
 (76,'male','Close fingerpicked steel strings, woody low notes and a spare brushed snare',{'male':.8,'acoustic-folk':1.2}),
 (120,'unspecified','Soft plucked strings over rounded synth bass and a broken dance pulse',{'house':1.,'acoustic-folk':1.}),
 (96,'instrumental','Muted brass, short keyboard replies, deep bass and a dry close kit',{}),
 (92,'female','Short electric piano chords, tuned bass and sparse fingertip percussion',{'female':.8,'rnb':1.2}),
 (126,'male','Scratchy rhythm guitar, a bouncing low octave and crisp live snare',{'male':.8,'disco-funk':1.2}),
 (106,'unspecified','Steel strings, a high guitar response and an open but quiet live kit',{'country':1.,'indie-rock':1.}),
 (136,'instrumental','Layered short synth plucks, a deep four-beat pulse and precise metallic ticks',{'house':1.,'lofi':1.}),
 (88,'female','Soft close vocal, felted piano, bowed bass and a restrained shaker',{}),
 (118,'male','Dry rhythm guitar, rounded bass, handclaps and a tight human kick',{'male':.8,'disco-funk':1.2}),
 (102,'unspecified','A nylon-string motif, plucked bass and dry hand drums in a narrow room',{'acoustic-folk':1.}),
 (142,'instrumental','Interlocking bell-like synth notes, clipped bass and a clear dance kick',{'house':1.}),
 (114,'female','Chiming electric guitar, mellow keyboard replies and a relaxed live backbeat',{'female':.8,'indie-rock':1.2}),
 (78,'male','Steel-string fingerpicking, low bowed strings and softly brushed drums',{}),
 (128,'unspecified','Plucked acoustic figures, rounded synth chords and lightly swung dance percussion',{'house':1.,'acoustic-folk':1.}),
 (94,'instrumental','Breathy flute, resonant wooden keys, upright bass and short rim-click phrases',{}),
]


def fresh():
    rows=[];text_index=0
    for index,(bpm,voice,sound,styles) in enumerate(DESIGNS):
        group='train' if index<16 else 'dev' if index<24 else 'causal' if index<28 else 'final'
        split={'train':'train','dev':'dev','causal':'test','final':'transfer'}[group]
        lead=('Instrumental throughout; no singing, speech or wordless vocal sounds.' if voice=='instrumental' else
              f"One adult {'lead' if voice=='unspecified' else voice+' lead'} vocalist with clear pitched words and natural breath.")
        caption=(f'Global Metadata:\nA coherent arrangement at {bpm} BPM in 4/4. '
                 'Start promptly with the main motif; develop the verse and let the chorus open the harmony.\n'
                 f'Vocal Details:\n{lead}\nArrangement:\n{sound}. '
                 'Give the foreground room, keep the bass defined, and let short answering phrases leave spaces.')
        if voice=='instrumental':
            # A distinct original instrumental instruction sheet for every family.
            lyrics=f'[instrumental]\n[intro: {sound.lower()}]\n[verse: establish a short motif]\n[chorus: broaden the harmony]\n[bridge: thin the texture]\n[outro]'
        else:
            verse,chorus=TEXTS[text_index];text_index+=1
            verse=verse.replace('|','\n');chorus=chorus.replace('|','\n')
            lyrics=f'[verse]\n{verse}\n[chorus]\n{chorus}\n{chorus}\n[verse]\n{verse}\n[chorus]\n{chorus}\n{chorus}\n[outro]'
        rows.append(dict(family=f'search-{group}-{index:02d}',split=split,group=group,
                         title=f'Reward search fixture {index:02d}',caption=caption,lyrics=lyrics,
                         voice=voice,bpm=bpm,style_multipliers=styles))
    assert text_index==len(TEXTS)
    validate_families(rows)
    return rows
