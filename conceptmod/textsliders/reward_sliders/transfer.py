"""Fresh transfer families, frozen after pilot decisions and before full songs."""
from .specs import validate_families


def families():
    settings = [
        (76,'female','A low piano, bowed bass and quiet brushed drums',{},
         ['The evening train has crossed the valley','A cup is cooling by the sill','I turn the handle on the radio','And let the room grow quiet still'],
         ['Leave a little light beside me','I can find the handle in the dark']),
        (128,'male','Clipped electric guitar, syncopated bass and bright dance drums',{'male':.8,'disco-funk':1.2},
         ['The orange crate is full of buttons','We sort the small ones from the large','You find a coat that needs a fastener','I trace the pocket with a card'],
         ['Every little piece can matter','Every empty buttonhole can wait']),
        (94,'unspecified','An airy flute motif, plucked strings and dry hand percussion',{},
         ['The orchard path has lost its marker','We tie a ribbon to a branch','The fruit is still too green for picking','We leave the basket by the fence'],
         ['Let the turning weather take it','Let the ripening have another day']),
        (112,'female','Warm electric keys, melodic bass and interlocking shakers',{'female':.8,'afrobeats':1.2},
         ['I bring the folding table outside','You lay the cloth against the breeze','We put a stone on every corner','And rest our hands upon our knees'],
         ['There is room beside the garden','There is shade enough for both of us']),
        (68,'male','Close nylon guitar, a soft low drum and a small string response',{'male':.8,'acoustic-folk':1.2},
         ['A single sock behind the washer','Has gathered dust along the seam','I shake it out beside the back door','And hang it where the sun comes in'],
         ['Find the pair among the washing','Let the lonely little things come home']),
        (140,'unspecified','Bright synth arpeggios, quick dry snares and firm low bass',{'pop':1.,'house':1.},
         ['The crossing light is counting backwards','We slow our feet before the curb','A cyclist rings a crooked rhythm','We answer with a laughing word'],
         ['Wait until the road is ready','There is time enough to cross together']),
        (88,'female','Loose live drums, a reed keyboard and soft chiming guitar',{'female':.8,'indie-rock':1.2},
         ['The library has changed its doorway','The old return slot wears a plate','We carry books around the corner','And find a person at the gate'],
         ['There is more than one way inside','There is more than one good place to start']),
        (122,'male','Deep rounded bass, clipped keys and syncopated electronic percussion',{'male':.8,'reggaeton':1.2},
         ['A flattened box becomes a picture','You cut a frame around the crease','I draw the view beyond the rooftop','And fill the empty sky with leaves'],
         ['Put the crooked frame around it','Give the ordinary wall a view']),
        (84,'instrumental','A muted trumpet melody over double bass, sparse drums and tremolo keys',{},
         [],[]),
        (132,'instrumental','Bowed strings over pulsing synth bass, dry kick and short wooden clicks',{'house':1.,'lofi':1.},
         [],[]),
        (102,'unspecified','An acoustic guitar motif over deep bass, rim clicks and airy synth pads',{'country':1.,'rnb':1.},
         ['The bicycle is leaning sideways','A fallen twig has caught the chain','We set it straight beside the doorway','And test the wheel against the rain'],
         ['Turn it once and let it settle','There is still another mile to ride']),
        (152,'female','Sharp palm-muted guitars, tight bass and a brisk human drum kit',{'female':.8,'pop-punk':1.2},
         ['We pack the last cup in a sweater','The empty shelf has changed its tone','I check the drawer behind the counter','And find the note we left alone'],
         ['Take the small things that will travel','Leave a little kindness in the room']),
    ]
    genres=['chamber pop','dance funk','acoustic chamber folk','percussion pop','acoustic ballad','arpeggiated dance',
            'guitar pop','syncopated electronic pop','small ensemble jazz','orchestral dance','acoustic soul','fast guitar pop']
    rows=[]
    for i,(bpm,voice,instruments,styles,verse,chorus) in enumerate(settings):
        seconds=75. if bpm<100 else 65.
        vocal = 'Instrumental throughout, with no vocal sounds.' if voice=='instrumental' else (
                f"One adult {'lead' if voice=='unspecified' else voice+' lead'} vocalist, clearly pitched words and natural breath.")
        caption=(f'Global Metadata:\nA complete song of about {seconds:g} seconds at {bpm} BPM in 4/4.\n'
                 f'Vocal Details:\n{vocal}\nArrangement:\n{instruments}. '
                 'A brief introduction, two verses and recurring chorus, a contrasting bridge and a final resolved cadence. '
                 'Let the motif develop across the sections and finish with a short natural instrumental ending.')
        lyrics=('[intro]\n[instrumental]\n[verse]\n[chorus]\n[bridge]\n[chorus]\n[outro]' if i==8 else
                '[intro]\n[verse]\n[instrumental]\n[chorus]\n[solo]\n[chorus]\n[outro]' if i==9 else
                '[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(chorus+chorus)+
                '\n[verse]\n'+'\n'.join(verse)+'\n[chorus]\n'+'\n'.join(chorus+chorus)+
                '\n[bridge]\n'+'\n'.join(verse[:2])+'\n[chorus]\n'+'\n'.join(chorus+chorus)+'\n[outro]')
        rows.append(dict(family=f'transfer-{i:02d}',split='transfer',title=f'Reward transfer {i:02d}',
                         voice=voice,bpm=bpm,genre=genres[i],caption=caption,lyrics=lyrics,style_multipliers=styles,
                         intended_seconds=seconds,render_cap_seconds=seconds+30.))
    validate_families(rows)
    return rows
