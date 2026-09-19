"""Unused independent replication draft; fixed candidate required before rendering."""
from ..reward_sliders.specs import validate_families


def families():
    settings=[
        (76,'female','Soft electric piano, a low clarinet line, bowed bass and brushed toms. A close singing voice with gentle consonants; instruments answer in the pauses.',{},
         ['The kettle waits beside the basin','A folded towel beside the door','You set a cup upon the counter','I hear your footsteps cross the floor'],
         ['Let the water settle slowly','Let the room become a place']),
        (118,'female','Open electric guitar chords, a moving bass line, floor toms and loose cymbals. An energetic sung lead over a compact band playing in one room.',{'female':.8,'indie-rock':1.2},
         ['We stack the chairs beneath the awning','The folding tables lean in rows','You find the missing silver fastener','And hold the corner while it goes'],
         ['One more turn and it is steady','One more thing that we can mend']),
        (94,'male','Steel-string acoustic picking, low harmonium notes and soft heel taps. A warm dry sung lead with a single harmony line at the ends of phrases.',{'male':.8,'acoustic-folk':1.2},
         ['A pair of boots beside the stairway','A coat that needs a second hook','We clear a space along the railing','And find the page inside the book'],
         ['There is shelter in the small things','There is time to put them right']),
        (114,'male','Muted electric piano chords, a buoyant bass line, palm-muted guitar and a crisp kit with open hi-hat accents. A firm melodic lead vocal and short backing replies.',{'male':.8,'disco-funk':1.2},
         ['The station clock is running early','We have a minute left to spare','You count the tiles along the platform','I trace a rhythm in the air'],
         ['Keep the little moment moving','Keep a little room to smile']),
        (84,'unspecified','A softly broken drum pattern, mellow keyboard chords, a breathy reed melody and an even bass pulse. A clearly pitched lead voice with relaxed timing.',{},
         ['We lift the frame against the plaster','And mark a point above the chair','You hold the ruler at the center','I take a step and leave it there'],
         ['Every room can find its balance','Every wall can hold a view']),
        (124,'unspecified','A steady electronic dance kick, syncopated acoustic strumming, low synth bass and airy sustained chords. A clear sung melody above the rhythm.',{'house':1.,'acoustic-folk':1.},
         ['The narrow path behind the houses','Has room enough for us to pass','We move the branch beside the walkway','And watch the light across the grass'],
         ['Take the path as it is given','Let the next step come in time']),
        (120,'instrumental','A plucked synth arpeggio, rounded sub bass, a tight four-on-the-floor kick and dry handclaps. Alternate two short melodic phrases above the pulse.',{'house':1.},[],[]),
        (104,'instrumental','Warm marimba, low cello pizzicato and a restrained acoustic kit. Trade the marimba theme with a short answering cello phrase while the rhythm develops.',{},[],[]),
    ]
    rows=[]
    for index,(bpm,voice,instruments,styles,verse,refrain) in enumerate(settings):
        vocal=('Instrumental only, with no singing, speech, humming or vocal samples.' if voice=='instrumental' else
               f"One adult {'lead' if voice=='unspecified' else voice+' lead'} vocalist singing clear words with natural breath and phrasing.")
        caption=f'Global Metadata:\nA complete musical arrangement at {bpm} BPM in 4/4. Start with a distinct motif and develop it with small changes in the rhythm and instrumentation.\nVocal Details:\n{vocal}\nArrangement:\n{instruments}'
        if index==6:lyrics='[arpeggio opening]\n[electronic theme]\n[second motif]\n[percussion interlude]\n[electronic theme]\n[instrumental close]'
        elif index==7:lyrics='[marimba opening]\n[cello answer]\n[ensemble theme]\n[rhythm variation]\n[ensemble return]\n[instrumental closing phrase]'
        else:lyrics='[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[outro]'
        rows.append(dict(family=f'replication-draft-{index:02d}',split='test',title=f'Replication fixture {index:02d}',voice=voice,bpm=bpm,caption=caption,lyrics=lyrics,style_multipliers=styles))
    validate_families(rows);return rows
