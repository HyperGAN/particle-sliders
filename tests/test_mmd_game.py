"""Check that a persistent leaderboard cannot silently change its comparison."""
from argparse import Namespace
from pathlib import Path
import copy

import pytest

from analysis.gan_bcap.autonomous_audio import RULE,CONCEPTS,score_components
from analysis.gan_bcap.mmd_game_20260905 import game as G


def fixture(tmp_path):
    prompts=tmp_path/'prompts.yaml';prompts.write_text('fixture')
    weights=tmp_path/'candidate.safetensors';weights.write_bytes(b'test adapter')
    protocol=dict(prompts=str(prompts),prompts_sha256=G.sha(prompts),rows=[0,1],seeds=[7,23],duration=20.,
        gpu=1,scales=[0.,1.],rule=RULE,concepts=CONCEPTS,concept='gender',scorer_sha256='frozen-scorer',
        sources={str(G.ROOT/'analysis/gan_bcap/render_v2.py'):G.sha(G.ROOT/'analysis/gan_bcap/render_v2.py')})
    G.write(tmp_path/'protocol.json',protocol)
    protocol=G.read(tmp_path/'protocol.json')
    protocol['_sha256']=G.sha(tmp_path/'protocol.json')
    records=[]
    for row in protocol['rows']:
        folder=tmp_path/f'row-{row}'
        G.write(folder/'render_spec.json',dict(prompts_sha256=protocol['prompts_sha256'],row=row,
            seeds=[7,23],duration=20.,scales=[0.,1.],seed_retries=0,
            renderer_sha256=protocol['sources'][str(G.ROOT/'analysis/gan_bcap/render_v2.py')],
            checkpoints=[dict(path=str(weights),sha256=G.sha(weights),steps=720)]))
        G.write(folder/'game-provenance.json',dict(gpu=1,protocol_sha256=protocol['_sha256']))
        for seed in protocol['seeds']:
            leaf=folder/f'candidate-s{seed}';leaf.mkdir()
            values={}
            for kind in ['candidate','baseline','positive_reference']:
                path=leaf/f'{kind}.wav';path.write_bytes(f'{row}/{seed}/{kind}'.encode())
                values[kind]=dict(audio=str(path),sha256=G.sha(path),concept=.025 if kind=='candidate' else 0.,
                    enjoyment=7.,production=8.,lyrics=.9,hf14k_fraction=0.,rms=.1)
            score,components=score_components(values['candidate'],values['baseline'],values['positive_reference'])
            records.append(dict(checkpoint=dict(path=str(weights),sha256=G.sha(weights),steps=720),
                seed=seed,fixture=G.digest([protocol['prompts_sha256'],row,seed,20.]),**values,
                heuristic_score=score,components=components,eligible=True))
    report=tmp_path/'scores.json'
    G.write(report,dict(status='complete',rule=RULE,concepts=CONCEPTS,concept='gender',source_sha256='frozen-scorer',records=records))
    return protocol,weights,report


def test_complete_matched_report_and_changed_weights(tmp_path):
    protocol,weights,report=fixture(tmp_path)
    rows,controls=G.validate_records(protocol,weights,[report],{})
    assert len(rows)==4 and len(controls)==4
    weights.write_bytes(b'different adapter')
    with pytest.raises(ValueError,match='Checkpoint changed'):G.validate_records(protocol,weights,[report],{})


@pytest.mark.parametrize('change,message',[
    ('missing','exactly one clip'),('duplicate','exactly one clip'),('score','Score does not recompute'),
    ('rule','frozen scorer'),('gpu','this GPU'),('renderer','different renderer'),('checkpoint','render specification')])
def test_invalid_leaderboard_comparisons_are_rejected(tmp_path,change,message):
    protocol,weights,report=fixture(tmp_path)
    blob=G.read(report)
    if change=='missing':blob['records'].pop()
    elif change=='duplicate':blob['records'][-1]=copy.deepcopy(blob['records'][0])
    elif change=='score':blob['records'][0]['heuristic_score']+=.5
    elif change=='rule':blob['rule']=dict(blob['rule'],version='retuned-to-candidate')
    elif change=='gpu':
        path=tmp_path/'row-0/game-provenance.json';meta=G.read(path);meta['gpu']=0;G.write(path,meta)
    else:
        path=tmp_path/'row-0/render_spec.json';meta=G.read(path)
        if change=='renderer':meta['renderer_sha256']='different'
        else:meta['checkpoints']=[]
        G.write(path,meta)
    G.write(report,blob)
    with pytest.raises(ValueError,match=message):G.validate_records(protocol,weights,[report],{})


def test_registration_recovers_after_interrupted_state_publish(tmp_path,monkeypatch):
    _,weights,report=fixture(tmp_path)
    G.write(tmp_path/'opponents.json',dict(opponents=[]))
    G.publish(tmp_path,dict(version=1,protocol_sha256=G.sha(tmp_path/'protocol.json'),entries={},
        active_round=None,completed_rounds=[],controls={}))
    args=Namespace(id='seed',weights=weights,family='mmd',round=None,label=None,scores=[report])
    publish=G.publish
    def interrupt(*args):raise OSError('Simulated interrupted write')
    monkeypatch.setattr(G,'publish',interrupt)
    with pytest.raises(OSError):G.register(tmp_path,args)
    assert not G.read(tmp_path/'state.json')['entries']
    assert (tmp_path/'entries/seed/entry.json').exists()
    monkeypatch.setattr(G,'publish',publish)
    G.register(tmp_path,args)
    state=G.read(tmp_path/'state.json')
    assert state['champions']==dict(overall='seed',mmd='seed')
    assert len(state['controls'])==4
    with pytest.raises(ValueError,match='already exists'):G.register(tmp_path,args)


def test_same_fixture_requires_identical_controls(tmp_path):
    protocol,weights,report=fixture(tmp_path)
    _,controls=G.validate_records(protocol,weights,[report],{})
    blob=G.read(report);record=blob['records'][0]
    changed=tmp_path/'alternative-control.wav';changed.write_bytes(b'new control')
    record['baseline'].update(audio=str(changed),sha256=G.sha(changed))
    G.write(report,blob)
    with pytest.raises(ValueError,match='same control audio'):
        G.validate_records(protocol,weights,[report],controls)


def test_rounds_preserve_wins_failures_and_resume_state(tmp_path):
    protocol,_,_=fixture(tmp_path)
    G.write(tmp_path/'opponents.json',dict(opponents=[]))
    def entry(name,family,score,round_name=None):
        return dict(id=name,family=family,score=score,worst_score=score,eligible=True,round=round_name,weights_sha256=name)
    state=dict(version=1,protocol_sha256=G.sha(tmp_path/'protocol.json'),entries={
        'reference':entry('reference','reference',.6),'seed':entry('seed','mmd',.5)},
        active_round=None,completed_rounds=[],controls={})
    G.publish(tmp_path,state)
    args=Namespace(hypothesis='Smaller search steps find descent',change='More step fractions',parent=None,budget=150)
    G.begin(tmp_path,args)
    with pytest.raises(ValueError,match='Resume active round'):G.begin(tmp_path,args)
    state=G.read(tmp_path/'state.json');state['entries']['better']=entry('better','mmd',.7,'round-0001');G.publish(tmp_path,state)
    note=tmp_path/'notes.md';note.write_text('Measured gain with the frozen judge.')
    G.finish(tmp_path,Namespace(round='round-0001',notes_file=note,next_experiment='Check more histories',failed=False))
    _,state=G.load_game(tmp_path)
    assert state['active_round'] is None and state['champions']['overall']=='better'
    assert state['completed_rounds'][0]['outcome']=='new overall lead'
    G.begin(tmp_path,args)
    G.finish(tmp_path,Namespace(round='round-0002',notes_file=note,next_experiment='Repair failed training',failed=True))
    _,state=G.load_game(tmp_path)
    assert state['champions']['mmd']=='better'
    assert state['completed_rounds'][1]['outcome']=='failed experiment'


def test_calibration_round_can_win_overall_without_changing_mmd_champion(tmp_path):
    fixture(tmp_path)
    G.write(tmp_path/'opponents.json',dict(opponents=[]))
    def entry(name,family,score,round_name=None):
        return dict(id=name,family=family,score=score,worst_score=score,eligible=True,round=round_name,weights_sha256=name)
    state=dict(version=1,protocol_sha256=G.sha(tmp_path/'protocol.json'),entries={
        'reference':entry('reference','reference',.8),'seed':entry('seed','mmd',.75)},
        active_round=None,completed_rounds=[],controls={})
    G.publish(tmp_path,state)
    G.begin(tmp_path,Namespace(hypothesis='Modest strength increase',change='Alpha times1.25',
        family='calibration',parent='reference',budget=0))
    state=G.read(tmp_path/'state.json')
    state['entries']['calibrated']=entry('calibrated','calibration',1.1,'round-0001')
    G.publish(tmp_path,state)
    note=tmp_path/'notes.md';note.write_text('A measured strength calibration, zero optimizer updates.')
    G.finish(tmp_path,Namespace(round='round-0001',notes_file=note,next_experiment='Confirm extra fixtures',failed=False))
    _,state=G.load_game(tmp_path)
    assert state['champions']==dict(overall='calibrated',mmd='seed')
    record=state['completed_rounds'][0]
    assert record['family']=='calibration' and record['additional_update_attempts']==0
    assert record['outcome']=='new overall lead'
    assert record['gain_vs_previous_mmd'] is None
