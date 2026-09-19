"""Unused confirmation fixture draft; rendering requires a frozen passing candidate."""
from ..reward_sliders.specs import validate_families


def families():
    settings=[
        (78,'female','Upright piano, bowed low strings and soft brushed snare. A close lead voice with audible breath; leave space after each line.',{},
         ['I thread the needle by the window','The small bent seam is coming loose','You hold the sleeve against the daylight','I pull the thread and let it move'],
         ['Every stitch can find its place','Leave a little room for grace']),
        (122,'female','Bright electric guitar, tambourine, melodic bass and a lively acoustic kit. Short guitar answers between sung lines.',{'female':.8,'indie-rock':1.2},
         ['The painted sign is nearly ready','We turn it gently on its side','You brush the letters round the corner','I keep the dripping edge held high'],
         ['Let the color have its moment','Let the open doorway shine']),
        (96,'male','Fingerpicked nylon guitar, bowed bass and light wooden percussion. Dry intimate lead vocal with a restrained harmony on the refrain.',{'male':.8,'acoustic-folk':1.2},
         ['The garden hose has found a tangle','We spread the loops across the stone','A little water finds the planter','Before the warming day is gone'],
         ['Give the roots another minute','Let the quiet water through']),
        (116,'male','Clipped rhythm guitar, springy electric bass, bright keys and a dry dance kit. Firm sung lead and compact backing responses.',{'male':.8,'disco-funk':1.2},
         ['We roll the carpet from the hallway','The wooden floor begins to show','You tap a rhythm with the handle','I catch the beat and move it slow'],
         ['There is room enough for turning','There is room enough to stay']),
        (88,'unspecified','Mellow reed keys, a rounded bass line, shaker and hand drums. A clearly pitched lead voice; unhurried phrases over a steady pocket.',{},
         ['A little bridge above the gutter','Is made of planks we found nearby','We test the smallest one together','And watch a paper lantern rise'],
         ['Carry only what is needed','Keep a steady step with me']),
        (126,'unspecified','A four-on-the-floor kick, plucked acoustic strings, soft synth chords and a clean low bass. A sung lead with clear words above the pulse.',{'house':1.,'acoustic-folk':1.},
         ['The rolling shutter lifts in stages','A strip of morning fills the shop','We bring the baskets to the pavement','And set the little clock on top'],
         ['Open up the day together','Let the ordinary morning in']),
        (124,'instrumental','A short bell-like synth melody, rounded bass, tight electronic kick and clipped handclaps. Introduce a second answering motif as the groove develops.',{'house':1.},[],[]),
        (108,'instrumental','A vibraphone melody, warm organ bass and an acoustic drum kit with gentle rim clicks. Let the mallet phrase and drums trade short variations.',{},[],[]),
    ]
    rows=[]
    for index,(bpm,voice,instruments,styles,verse,refrain) in enumerate(settings):
        vocal=('Instrumental throughout. No singing, spoken words, humming, or vocal samples.' if voice=='instrumental' else
               f"One adult {'lead' if voice=='unspecified' else voice+' lead'} vocalist with clearly pitched words and natural phrasing.")
        caption=(f'Global Metadata:\nA coherent arrangement at {bpm} BPM in 4/4. Begin promptly with a clear musical motif and let the arrangement develop.\n'
                 f'Vocal Details:\n{vocal}\nArrangement:\n{instruments}')
        if index==6:lyrics='[instrumental intro]\n[synth hook]\n[bass variation]\n[drum break]\n[synth hook]\n[instrumental outro]'
        elif index==7:lyrics='[mallet introduction]\n[instrumental theme]\n[organ response]\n[drum variation]\n[instrumental theme]\n[outro]'
        else:lyrics='[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(refrain*2)+'\n[outro]'
        rows.append(dict(family=f'confirmation-draft-{index:02d}',split='test',title=f'Confirmation fixture {index:02d}',
                         voice=voice,bpm=bpm,caption=caption,lyrics=lyrics,style_multipliers=styles))
    validate_families(rows)
    return rows
