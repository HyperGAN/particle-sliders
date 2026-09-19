"""Freeze scored training preferences and a four-comparison early screen."""
from copy import deepcopy
from pathlib import Path
import time

from ..reward_sliders.specs import ROOT,read_json,write_json,sha,digest,validate_families
from ..reward_sliders.experiment import verify
from ..reward_search.setup import arm

SOURCE=ROOT/'analysis/reward_search_20260908'
DEFAULT_RUN=ROOT/'analysis/reward_preference_20260908'


def fresh_confirmation():
    designs=[
        ('female',84,'Warm piano, a close brushed kit, bowed low strings and a clearly pitched intimate lead',{}),
        ('male',120,'Clipped rhythm guitar, dry handclaps, syncopated bass and a relaxed chest-register lead',{'male':.8,'disco-funk':1.2}),
        ('unspecified',106,'Fingerpicked steel strings, resonant wood percussion and rounded electric bass',{'acoustic-folk':1.}),
        ('instrumental',132,'A short repeating bell figure, pulsing low synth, crisp kick and separated hats',{'house':1.})]
    verses=[
        'A blue cup waits beside the kettle\nThe handle catches morning light\nI turn the spoon against the saucer\nAnd lift the curtain from the night',
        'The shoe repair has changed its window\nA leather belt has taken shape\nWe test the buckle on the counter\nAnd fold the receipt into the tape',
        'A garden hose across the gravel\nHas drawn a river through the dust\nWe move the chair into the shadow\nAnd leave the gate upon its rust']
    result=[]
    for index,(voice,bpm,sound,styles) in enumerate(designs):
        instrumental=voice=='instrumental'
        lead='No singing, speech or wordless vocals.' if instrumental else f'One adult {voice if voice!="unspecified" else "solo"} lead; clear pitched words and natural breath.'
        lyrics=f'[instrumental]\n[intro: {sound.lower()}]\n[verse: a clear repeating motif]\n[chorus: open the harmony]\n[outro]' if instrumental else f'[verse]\n{verses[index]}\n[chorus]\nLeave a little room beside us\nLet the ordinary shine\nLeave a little room beside us\nLet the ordinary shine\n[outro]'
        result.append(dict(family=f'preference-confirm-{index:02d}',split='transfer',group='preference-confirm',
            voice=voice,bpm=bpm,title=f'Preference confirmation {index:02d}',
            caption=f'Global Metadata:\nA coherent arrangement at {bpm} BPM in 4/4, with a prompt start and a clear main motif.\nVocal Details:\n{lead}\nArrangement:\n{sound}. Let short answering phrases leave space and keep the low register defined.',
            lyrics=lyrics,style_multipliers=styles))
    return result


def freeze(run=DEFAULT_RUN):
    from ..reward_sliders.render import resolve_styles
    from ..reward_sliders.evaluate import extra_style_hashes
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    path=run/'manifest.json'
    if path.exists():m=read_json(path);verify(m);return m
    old=read_json(SOURCE/'manifest.json');verify(old)
    observations=read_json(SOURCE/'training-observations.json')
    assert len(observations)==96 and all(r['split']=='train' for r in observations)
    preferences=[];replays=[]
    for family in sorted({r['family'] for r in observations}):
        candidates=sorted([r for r in observations if r['family']==family],key=lambda r:(r['reward']['scalar'],r['seed']))
        assert len(candidates)==4
        rejected,chosen=candidates[0],candidates[-1]
        for row in (chosen,rejected):
            assert row['status']=='complete' and row['reward']['valid'] and row['arm']['name']=='off'
            assert sha(row['trajectory'])==row['trajectory_sha256'] and sha(row['audio'])==row['audio_sha256']
            replays.append(dict(observation=row))
        gap=chosen['reward']['scalar']-rejected['reward']['scalar']
        preferences.append(dict(family=family,chosen=chosen['id'],rejected=rejected['id'],ce_gap=gap,
            chosen_ce=chosen['reward']['scalar'],rejected_ce=rejected['reward']['scalar'],weight=min(2.,max(.25,gap))))
    controls=[]
    for family in (f for f in old['families'] if f.get('group')=='causal'):
        source=SOURCE/'stages/causal/observations'/f"{family['family']}-s5521-off.json"
        row=read_json(source);assert row['status']=='complete' and sha(row['audio'])==row['audio_sha256']
        controls.append(dict(source=str(source),sha256=sha(source),observation=row))
    initial=arm(old['search']['reference']['checkpoint'],.5,'incumbent-half')
    m=deepcopy(old);m.pop('compatible_manifest_sha256',None)
    m.update(schema='reward-semantic-preference-v1',name='reward-ce-preference-continuation',created_unix=time.time())
    m['families']+=fresh_confirmation();validate_families(m['families'])
    for family in fresh_confirmation():
        comps=resolve_styles(family['style_multipliers']);m['style_components'][family['family']]=comps;extra_style_hashes(m,comps)
    for name in m['style_hashes']:
        m['file_stats'][name]=dict(size=Path(name).stat().st_size,mtime_ns=Path(name).stat().st_mtime_ns)
    m['source_hashes'].update({str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    m['preference']=dict(initial=initial,frames=500,seed=7,training_families=24,training_pairs=24,
        data='highest and lowest CE of four existing Off takes within each exact training family',
        semantic_targets='501 bit-exact feedback reconstructions; discarded warmup excluded; first 500 emitted tokens',
        objective='mean semantic log-probability preference, reference-relative; full CFG, untruncated vocabulary; reference KL and prompt anchor',
        full_audio_likelihood=False,reward_backpropagated=False,reference_paper='https://arxiv.org/abs/2305.18290',
        beta=5.,kl_weight=.1,prompt_weight=.1,gross_kl_stop=.5,gradient_norm=1.,
        learning_rates={'gentle':1e-5,'faster':3e-5},checkpoints=[30,60,120],
        screen='four independent-from-training families; one saved Off seed each; rank wins, then worst delta, then mean',
        continuation='Both arms get steps 30 and 60; step 120 only for an arm passing step 60: 3/4 wins, mean >= .02, worst >= -.30',
        confirmation='Only a passing best development checkpoint receives four fresh paired Off/candidate renders (8 clips)',
        max_new_audio=36,baseline_audio_reused=True,no_automatic_promotion=True)
    write_json(run/'preferences.json',preferences);write_json(run/'controls.json',controls)
    for gpu in (0,1):write_json(run/f'replay-gpu{gpu}.json',replays[gpu::2])
    write_json(path,m)
    for source,expected in m['source_hashes'].items():
        dest=run/'provenance'/Path(source).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(source).read_bytes());assert sha(dest)==expected
    write_json(run/'audit/setup.json',dict(passed=True,training_pairs=24,training_histories=48,
        reused_training_observations_sha256=sha(SOURCE/'training-observations.json'),
        new_audio_for_training=0,preferences_sha256=sha(run/'preferences.json'),
        training_and_screen_families_disjoint=not bool({p['family'] for p in preferences}&{x['observation']['family'] for x in controls})))
    return m
