import pytest
import torch

from conceptmod.textsliders.reward_game.policy_objective import balanced_advantages, clipped_loss


def test_clipping_removes_only_updates_beyond_the_improvement_bound():
    positive=torch.tensor([0.,.4,-.4],requires_grad=True)
    clipped_loss(positive,torch.zeros(3),1.).backward()
    assert positive.grad[0]<0 and positive.grad[1]==0 and positive.grad[2]<0
    negative=torch.tensor([0.,.4,-.4],requires_grad=True)
    clipped_loss(negative,torch.zeros(3),-1.).backward()
    assert negative.grad[0]>0 and negative.grad[1]>0 and negative.grad[2]==0


def test_advantages_are_family_relative_and_skip_ambiguous_pairs():
    rows=[dict(id='a',family='f',seed=1,ce=7.),dict(id='b',family='f',seed=2,ce=8.),
          dict(id='c',family='g',seed=1,ce=8.),dict(id='d',family='g',seed=2,ce=8.01)]
    assert balanced_advantages(rows)==dict(a=-1.,b=1.,c=0.,d=0.)
    shifted=[dict(r,ce=r['ce']+2.) for r in rows]
    assert balanced_advantages(shifted)==balanced_advantages(rows)
    with pytest.raises(ValueError,match='distinct'):
        balanced_advantages(rows+[rows[0]])
