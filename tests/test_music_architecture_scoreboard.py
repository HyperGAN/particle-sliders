import json

from scripts.music_architecture_scoreboard import (
    QUALITY_TOLERANCE, audio_score, rank, training_score,
)
from scripts.search_yue2_critic import CANDIDATES, critic_args


def test_training_score_rewards_late_lock_and_rejects_short_runs():
    assert training_score([])['training_ready'] is False
    rising = [dict(cos_pos=.2 + i*.03, grad_norm=2., g_adv=1.) for i in range(20)]
    flat = [dict(cos_pos=.2, grad_norm=2., g_adv=1.) for _ in range(20)]
    assert training_score(rising)['training_fitness'] > training_score(flat)['training_fitness']


class Measurer:
    def measure(self, path):
        on = path.parent.name == 'metal'
        ce = 7.1 if on else 7.
        pq = 8. if on else 8.05
        concept = .4 if on else .1
        return dict(windows=[dict(aesthetics=dict(CE=ce,PQ=pq),
            concept=dict(distorted=concept))])


def test_audio_score_is_matched_and_quality_gated(tmp_path):
    for seed in (1,2):
        for take in ('off','metal'):
            path=tmp_path/f'row-0-seed-{seed}'/take/'audio.flac'
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
    score=audio_score(tmp_path,Measurer())
    assert score['pairs']==2 and score['quality_pass']
    assert score['enjoyment_delta'] > 0
    assert score['production_delta'] >= -QUALITY_TOLERANCE
    assert score['concept_delta'] > 0


def test_rank_fails_closed_and_prefers_enjoyment_before_train_proxy():
    rows=[
        dict(candidate='proxy',training_ready=True,audio_ready=False,
             training_fitness=99.),
        dict(candidate='quality',training_ready=True,audio_ready=True,quality_pass=True,
             enjoyment_delta=.1,production_delta=0.,concept_delta=.2,training_fitness=1.),
        dict(candidate='lower',training_ready=True,audio_ready=True,quality_pass=True,
             enjoyment_delta=0.,production_delta=.5,concept_delta=.5,training_fitness=50.),
    ]
    assert [row['candidate'] for row in rank(rows)] == ['quality','lower','proxy']


def test_search_contains_control_and_transferred_music3_transformer():
    assert CANDIDATES[0]['critic']=='mlp'
    transfer=next(row for row in CANDIDATES if row['name'].startswith('music3_mix'))
    args=critic_args(transfer)
    assert args[args.index('--critic')+1]=='mix'
    assert args[args.index('--critic_tokens')+1]=='16'
    assert args[args.index('--critic_width')+1]=='128'
