"""Unused fresh fixtures authored before any feed-forward development result."""
from ..reward_sliders.specs import validate_families


SEEDS=(196613,262147)


def families():
    settings=[
        (76,'female','A softly struck grand piano, plucked double bass, light brushes and a low clarinet response. Close, plainly sung lead phrases with audible breaths and gentle room reflections.',{},
         ['The wool is caught around the spindle','A loose end curls across my knee','You hold the thread against the window','And find the knot I could not see'],
         ['Turn it slowly through your fingers','Leave a little room to breathe']),
        (116,'female','An open electric guitar chord pattern, rounded picked bass and a dry acoustic kit with a loose hi-hat. A direct melodic lead rises into a refrain with a quiet second vocal layer.',{'female':.8,'indie-rock':1.2},
         ['The bus has crossed the upper hillside','Its windows flash above the trees','I lift my bag beside the shelter','And feel the gravel through my knees'],
         ['There is room along the back row','There is sky beyond the bend']),
        (88,'male','Fingerpicked steel-string guitar, low harmonium chords, soft upright bass and brushed floor tom. Warm lead singing, lightly rough at phrase endings, with a restrained backing response.',{'male':.8,'acoustic-folk':1.2},
         ['A copper pot is on the table','The handle holds a trace of heat','We pour the water in the basin','And pull the chairs in from the street'],
         ['Set the day beside the doorway','Let the kettle find its rest']),
        (108,'male','A tight syncopated electric bass, short clean guitar strokes, bright electric piano chords and crisp snare backbeats. A sung lead sits close to the beat, with compact unison answers in the refrain.',{'male':.8,'disco-funk':1.2},
         ['The clock is missing half its numbers','The second hand is painted green','We hang it higher in the hallway','Above the space we swept between'],
         ['Keep a little time for moving','Keep a little space for me']),
        (84,'unspecified','A softly swung drum groove, warm detuned electric keys, a mellow bass line and short muted horn phrases. Distinct melodic singing, a relaxed pocket and light tape grain.',{},
         ['The greenhouse glass is wet by morning','Small silver beads along the pane','We clear a shelf above the seedlings','And write their labels out again'],
         ['Give the roots another season','Give the leaves another day']),
        (120,'unspecified','An even electronic kick, dry rim clicks, fingerpicked acoustic guitar, a deep smooth synth bass and an airy organ pad. Clear sung phrases cross the steady pulse without rushing.',{'house':1.,'acoustic-folk':1.},
         ['The woven sail is on the shoreline','A row of pebbles holds it flat','We mend the corner by the water','And leave a little seam for that'],
         ['Let the wind arrive in its time','Let the quiet hold us here']),
        (124,'instrumental','A short rounded synth lead over a steady electronic kick, soft claps, a moving sub bass and delicate shaker accents. Answer the lead with a contrasting hollow bell phrase and return to the pulse.',{'house':1.},[],[]),
        (100,'instrumental','An acoustic guitar plays a picked melody above a warm cello, plucked bass and a dry restrained drum kit. Alternate the picked lead with a low bowed answer, adding a light piano countermelody.',{},[],[]),
    ]
    rows=[]
    for index,(bpm,voice,instruments,styles,verse,refrain) in enumerate(settings):
        vocal=('Instrumental throughout with no singing, speech, humming or vocal samples.' if voice=='instrumental' else
               f"One adult {'lead' if voice=='unspecified' else voice+' lead'} vocalist sings distinct words in a natural register.")
        caption=f'Global Metadata:\nA clear musical arrangement at {bpm} BPM in 4/4, with a recurring motif and evolving instrumental responses.\nVocal Details:\n{vocal}\nArrangement:\n{instruments}'
        if index==6:lyrics='[electronic rhythm introduction]\n[rounded synth melody]\n[hollow bell answer]\n[lead and bass variation]\n[shaker rhythm passage]\n[melody return]\n[bell and bass response]\n[instrumental ending]'
        elif index==7:lyrics='[acoustic guitar introduction]\n[picked melody and bass]\n[low cello response]\n[piano countermelody]\n[drum and bass passage]\n[guitar theme return]\n[bowed ensemble variation]\n[acoustic instrumental ending]'
        else:lyrics='[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[outro]'
        rows.append(dict(family=f'independent-bank-b-{index:02d}',split='test',title=f'Fresh arrangement B{index:02d}',
            voice=voice,bpm=bpm,caption=caption,lyrics=lyrics,style_multipliers=styles))
    validate_families(rows)
    return rows
