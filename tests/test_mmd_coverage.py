import torch

from analysis.gan_bcap.mmd_game_20260905.coverage import replace_history


def test_history_replacement_preserves_condition_and_recomputes_required_geometry(tmp_path):
    old = dict(prompt_embeds=torch.ones(2, 5, 4), frame_embeds=torch.ones(2, 7, 4),
               real=torch.ones(1, 3, 4), energy_neutral=torch.ones(2, 10, 4),
               energy_target=torch.ones(2, 10, 4), span_mask=torch.ones(2, 5),
               prompt_hash='condition', continuation_teacher=torch.ones(1), policy_teacher=torch.ones(1))
    frames = torch.arange(8.).reshape(1, 2, 4)
    path = tmp_path/'history.pt'
    torch.save(frames, path)
    new = replace_history(old, frames, seed=17, path=path, parent_sha256='parent', ended=False)
    assert torch.equal(new['prompt_embeds'], old['prompt_embeds'])
    assert torch.equal(new['span_mask'], old['span_mask'])
    assert new['frame_embeds'].shape == (2, 2, 4)
    assert torch.equal(new['frame_embeds'][0], frames[0])
    assert torch.equal(new['frame_embeds'][1], frames[0])
    assert new['energy_target'].shape == (2, 5, 4)
    assert not new['energy_target'].any()
    assert 'continuation_teacher' not in new and 'policy_teacher' not in new
    assert old['frame_embeds'].shape == (2, 7, 4)
    assert old['energy_target'].all()
    frames.zero_()
    assert new['frame_embeds'].sum() > 0
