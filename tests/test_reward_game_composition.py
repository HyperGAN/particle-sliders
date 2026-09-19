import pytest
from conceptmod.textsliders.reward_game import composition
from conceptmod.textsliders.reward_game.core import write,digest


def test_composition_counts_ties_and_missing_outputs_honestly(tmp_path,monkeypatch):
    name='example';folder=tmp_path/'composition'/name;manifest={}
    cases=[dict(id=f'f{f}-s{s}',family=f'f{f}',seed=s) for f in range(4) for s in (1,2)]
    arms=[dict(name='style-reduced'),dict(name='fixed-energy')]
    protocol=dict(cases=cases,arms=arms)
    monkeypatch.setattr(composition,'verify',lambda *args:(protocol,manifest))
    for case in cases:
        for arm in arms:
            delta=.2 if case!=cases[-1] else 0.
            write(folder/'observations'/f"{case['id']}-{arm['name']}.json",dict(arm=arm,
                provenance=dict(manifest_sha256=digest(manifest)),status='complete',reward=dict(valid=True,scalar=7.+(delta if arm['name']=='fixed-energy' else 0.))))
    card=composition.score(tmp_path,name)
    assert card['passed'] and card['wins']==7 and card['equal_family_gain']==pytest.approx(.175)
    assert not card['research_complete']
    (folder/'observations'/f"{cases[-1]['id']}-fixed-energy.json").unlink()
    card=composition.score(tmp_path,name)
    assert not card['passed'] and card['equal_family_gain'] is None and card['invalid']==[cases[-1]['id']]
