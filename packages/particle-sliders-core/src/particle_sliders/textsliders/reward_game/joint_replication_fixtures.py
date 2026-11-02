"""Unused independently authored replication fixtures; no candidate audio yet."""
from ..reward_sliders.specs import validate_families


def families():
    settings=[
        (80,'female','Felt piano, a soft viola counterline, plucked upright bass and a brushed snare. Close lead vocal with gentle breaths and a small room sound.',{},
         ['The folded map lies on the blanket','We trace the river to the bend','You mark a spot beside the footbridge','And draw a line around the end'],
         ['There is somewhere still to wander','There is daylight left to spend']),
        (120,'female','Ringing electric guitar chords, a warm overdriven bass, dry snare and loose ride cymbal. A lively sung lead with a short doubled refrain.',{'female':.8,'indie-rock':1.2},
         ['The bicycle is on the landing','A little oil will free the chain','We spin the wheel beside the doorway','And hear it turning clear again'],
         ['Let the spokes catch all the sunlight','Let the road begin again']),
        (92,'male','Nylon guitar arpeggios, quiet accordion chords, an upright bass and a soft hand drum. Intimate lead singing with an unhurried melodic line.',{'male':.8,'acoustic-folk':1.2},
         ['The old stone steps are dry by evening','We bring the cushions from the shed','You set a lantern on the railing','A pool of amber light is spread'],
         ['Stay until the shadows soften','Stay and let the hours unfold']),
        (112,'male','A clipped wah rhythm guitar, syncopated electric bass, bright organ stabs and a close dry kit. Confident sung lead with short backing harmonies.',{'male':.8,'disco-funk':1.2},
         ['We push the gate across the gravel','The latch is lighter than before','You tap the key against the railing','I answer with a knock upon the door'],
         ['Find the rhythm in the waiting','Find a place to set it free']),
        (86,'unspecified','A relaxed swung drum loop, a warm electric piano, low muted trumpet answers and a soft bass pulse. Clear melodic singing with loose phrase endings.',{},
         ['A paper boat beside the fountain','Is turning slowly in the blue','We watch it drift around the circle','And wonder where it travels to'],
         ['Give the little boat a current','Give the afternoon a tune']),
        (122,'unspecified','Steady electronic kick and crisp shakers beneath acoustic guitar picking, a deep synth bass and soft string pads. A clearly pitched lead vocal above the pulse.',{'house':1.,'acoustic-folk':1.},
         ['We carry boxes to the attic','The stairs are narrow near the top','You slide the smallest through the opening','I wait until the footsteps stop'],
         ['Every little thing has somewhere','Every journey has a pause']),
        (126,'instrumental','A clean square-wave synth motif, a round sub bass, dry claps and a tight electronic dance kick. Let a softer bell tone answer the main hook.',{'house':1.},[],[]),
        (102,'instrumental','A bright mandolin melody, low bowed cello answers, upright bass and a restrained acoustic kit. Alternate a short picked theme with a longer bowed phrase.',{},[],[]),
    ]
    rows=[]
    for index,(bpm,voice,instruments,styles,verse,refrain) in enumerate(settings):
        vocal=('Instrumental throughout, without vocals, speech, humming or vocal samples.' if voice=='instrumental' else
               f"One adult {'lead' if voice=='unspecified' else voice+' lead'} vocalist singing distinct words with natural timing.")
        caption=f'Global Metadata:\nA developing arrangement at {bpm} BPM in 4/4. Establish the main motif early and vary the instrumental responses between phrases.\nVocal Details:\n{vocal}\nArrangement:\n{instruments}'
        if index==6:lyrics='[synth pulse opening]\n[square-wave theme]\n[bell response]\n[bass interlude]\n[theme variation]\n[electronic close]'
        elif index==7:lyrics='[picked string opening]\n[mandolin theme]\n[bowed response]\n[rhythm interlude]\n[ensemble variation]\n[final instrumental phrase]'
        else:lyrics='[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[outro]'
        rows.append(dict(family=f'joint-replication-draft-{index:02d}',split='test',title=f'Independent fixture {index:02d}',
                         voice=voice,bpm=bpm,caption=caption,lyrics=lyrics,style_multipliers=styles))
    validate_families(rows)
    return rows
