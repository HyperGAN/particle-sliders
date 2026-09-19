"""Broader training support with distinct evaluation lyrics, declared upfront."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from .data import ROOT,sha,digest,validate_prompts,check_disjoint
from ..train_lm_slider_music3 import _load_rows

TRAIN_LYRICS=[
    ('The porch is full of moving shadows','A small blue truck goes rolling by','I set the seedlings on the railing','And watch a feather cross the sky'),
    ('A painted cup beside the radio','Has kept the warmth of yesterday','I thread the needle through the fabric','And mend the seam along the way'),
    ('We turn the sign toward the sidewalk','And draw the shutters open wide','The first warm loaf is on the counter','The flour settles at my side'),
    ('The orchard fence is leaning over','A weather vane begins to turn','We fill the cart with fallen branches','And stack the pieces we will burn'),
    ('I walk the dog around the corner','The pavement shines beneath the trees','A folded note is in my jacket','The ink is drying in the breeze'),
    ('The empty stage is made of timber','The curtain catches on a nail','We test the lights above the doorway','And paint a stripe along the rail'),
    ('The little store has closed for winter','A string of lights is hanging low','I put a star above the window','And brush the step clear of the snow'),
    ('We take the bowls out to the courtyard','And set the benches in a row','A quiet guest arrives with flowers','The candles make the table glow'),
]
EVAL_LYRICS=[
    ('The bakery awning shakes at sunrise','A sparrow lands beside the sign','We write the prices on the chalkboard','And hang the aprons on the line'),
    ('I find a spool beneath the staircase','A length of thread runs through the room','The orange curtains move together','And gather light against the gloom'),
    ('The ferry horn rolls through the harbor','A rope slips slowly from the post','I keep a shell inside my mitten','And count the chimneys on the coast'),
    ('We bring the cushions from the attic','And move the sofa through the hall','A crooked frame hangs by the doorway','The afternoon runs down the wall'),
    ('A cricket sings behind the water tank','The tools are leaning by the shed','I leave my boots beside the doorstep','And shake the sawdust from the bed'),
    ('The bookshop bell rings twice at closing','A careful hand folds back the page','We set the ladder by the shelves now','And lock the little iron cage'),
    ('The tram conductor checks the mirror','A pair of pigeons leaves the wire','I find the corner of my ticket','And fold it while the brakes expire'),
    ('We draw a square around the sapling','The heavy spade rests in the clay','A cloud rolls slowly past the rooftop','We wash the muddy gloves away'),
]
ARRANGEMENTS=[
    (84,'Muted guitar chords, brushed snare, warm bass and a close dry room.'),
    (116,'Short electric piano chords, firm kick, syncopated bass and crisp shaker.'),
    (138,'Bright plucked synth, rounded electronic bass, clipped drums and a clear wide mix.'),
    (66,'Resonant piano, sparse bowed strings, soft mallets and a quiet room.'),
]


def vocal(caption):
    caption=caption.replace('\\n','\n')
    return caption.split('Vocal Details:',1)[1].split('Arrangement:',1)[0].strip()


def build(source,out):
    rows,meta=_load_rows(source)
    validate_prompts(rows)
    original=rows[0]
    nv=vocal(original.get('neutral') or original['target']);pv=vocal(original['positive'])
    def make(lines,arrangement,offset=0):
        bpm,sound=arrangement
        def caption(voice):
            return f'Global Metadata:\nBPM {bpm+offset}. A complete song with a clear lead vocal.\nVocal Details:\n{voice}\nArrangement:\n{sound}'
        neutral=caption(nv)
        return dict(target=neutral,neutral=neutral,positive=caption(pv),negative=neutral,
            lyrics='[verse]\n'+'\n'.join(lines[:2])+'\n[chorus]\n'+'\n'.join(lines[2:]))
    training=[make(lines,arrangement) for lines in TRAIN_LYRICS for arrangement in ARRANGEMENTS]
    evaluation=[make(lines,ARRANGEMENTS[i%4],offset=3) for i,lines in enumerate(EVAL_LYRICS)]
    check_disjoint(training,evaluation)
    old_manifest=ROOT/'analysis/gan_bcap/convergence_20260905/audio/manifest.json'
    if old_manifest.exists():
        for fixture in json.loads(old_manifest.read_text())['fixtures']:
            held=yaml.safe_load(Path(fixture['path']).read_text())
            check_disjoint(training,held)
    validate_prompts(training);validate_prompts(evaluation)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fixtures={}
    for name,examples in [('train',training),('evaluation',evaluation)]:
        payload={k:meta[k] for k in ('plus_label','minus_label') if k in meta}
        payload.update(recommended_range=[0.,1.],rows=examples)
        path=out/f'{name}.yaml';content=yaml.safe_dump(payload,sort_keys=False,allow_unicode=True)
        if path.exists() and path.read_text()!=content:raise ValueError('Declared fixture file changed')
        path.write_text(content)
        fixtures[name]=dict(path=str(path.resolve()),sha256=sha(path),rows=len(examples),
            unique_lyrics=len({r['lyrics'] for r in examples}),lyric_hashes=sorted({digest(r['lyrics']) for r in examples}))
    manifest=dict(version=1,source_prompt_sha256=sha(source),generator_sha256=sha(__file__),fixtures=fixtures,
        split='Disjoint lyrics; training contains 8 lyric sheets across 4 arrangements. Evaluation has 8 separate sheets.',
        protected_convergence_holdout=str(old_manifest) if old_manifest.exists() else None)
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Locked fixture manifest changed')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.out),indent=2))
