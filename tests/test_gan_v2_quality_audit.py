import numpy as np
import pytest

from analysis.gan_bcap.quality_audit_v2 import compare_diversity,quality_decision,duplicate_groups
from conceptmod.textsliders.gan_v2.metrics import clip_diagnostics


def test_partial_two_mode_collapse_is_detected_despite_large_total_spread():
    reference=np.eye(8)
    candidate=reference[[0,0,0,0,1,1,1,1]]
    result=compare_diversity(candidate,reference,reference)
    assert result['status']=='collapse_suspected'
    assert result['near_duplicate_groups']['candidate']['count']==2
    assert result['effective_group_ratio']==pytest.approx(.25)
    assert result['candidate']['central_radius']>.1


def test_group_count_is_invariant_to_sample_order():
    x=np.eye(4)[[0,0,1,2,2,3]]
    a=duplicate_groups(x,.001)
    b=duplicate_groups(x[[5,1,4,2,0,3]],.001)
    assert a==b


def fixture(margin):
    base=dict(rms=.1,duration=20.,lyrics=.9,concept=-.2,hf14k_fraction=.01,clipped_fraction=0.)
    positive=dict(base,concept=.1)
    diagnostic=clip_diagnostics(dict(base,concept=margin),base,positive)
    clips=[dict(prompt=str(i),diagnostics=diagnostic) for i in range(8)]
    diversity={str(i):dict(status='no_collapse_detected') for i in range(8)}
    return clips,diversity


def test_validation_boolean_cannot_pass_a_wrong_endpoint():
    clips,diversity=fixture(-.05)
    assert quality_decision(clips,diversity,judge_calibration=dict(validated_on_reserved_labels=True))['decision']=='quality_unvalidated'
    calibration=dict(validated_on_reserved_labels=True,artifact_sha256='a'*64,
                     description_threshold=0.,minimum_endpoint_rate=.9)
    result=quality_decision(clips,diversity,judge_calibration=calibration)
    assert result['decision']=='reject_candidate'
    assert result['calibrated_endpoint_rate']==0.
    positive_clips,_=fixture(.1)
    assert quality_decision(positive_clips,diversity,judge_calibration=calibration)['decision']=='pass_declared_quality_gates'
