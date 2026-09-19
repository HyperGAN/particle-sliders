import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank import summarize
from dsp import measure, warnings


def test_waveform_screen_recognizes_gain_and_polarity_but_not_different_audio():
    from waveforms import correlation
    rng=np.random.default_rng(7);a=rng.normal(size=40000);b=rng.normal(size=40000)
    assert correlation(a,-2*a)>.999999
    assert correlation(a,b)<.05
    assert correlation(a,b[:10000]) is None
    assert correlation(np.zeros(40000),a) is None


def test_missing_lyrics_remain_missing_and_components_match_original_rule():
    from dsp import diagnostics, components
    from common import ROOT
    from analysis.gan_bcap.autonomous_audio import RULE, score_components
    from conceptmod.textsliders.gan_v2.metrics import clip_diagnostics
    b=dict(concept=.02,enjoyment=7.,production=8.,lyrics=.9,rms=.1,duration=20.,clipped_fraction=0.,hf14k_fraction=.003)
    c=dict(b,concept=.06,enjoyment=7.3,lyrics=.5,hf14k_fraction=.008)
    _,old=score_components(c,b,b)
    assert components(c,b,b,RULE)=={k:old[k] for k in ("concept","enjoyment","production","artifacts")}
    assert diagnostics(c,b,b)==clip_diagnostics(c,b,b)
    c["lyrics"]=None
    assert diagnostics(c,b,b)["lyric_status"]=="not_rescored"
    assert "lyric_proxy_regression" not in diagnostics(c,b,b)["failures"]
    a=candidate("step2000",[.1]*4)
    a["clips"][0]["candidate"]["lyrics"]=None
    assert summarize([a],dict(close_call_gap=.1))["ranking"][0]["mean"]["lyrics"] is None


def candidate(label, scores, flags=(), lyric_flags=()):
    clips=[]
    for i,score in enumerate(scores):
        row, seed = 2+i//2, i%2
        clips.append(dict(fixture=f"{row}-{seed}", row=row, seed=seed, score=score,
            technical_flags=list(flags), lyric_flags=list(lyric_flags),
            candidate=dict(sha256=f"{label}-{i}", concept=.1, enjoyment=7., production=8., lyrics=.9),
            baseline=dict(concept=.02), diagnostics=dict(reaches_positive_description_reference=True)))
    return dict(label=label, step=int(label[4:]), weights=label, integrity=dict(weights_sha256=label), clips=clips)


def test_bad_audio_cannot_win_with_a_high_quality_score():
    result=summarize([candidate("step2000",[.1]*4),candidate("step3400",[2.]*4,["near_silence"])],dict(close_call_gap=.1))
    assert result["recommendation"]["label"]=="step2000"
    assert result["ranking"][-1]["status"]=="avoid_pending_review"


def test_asr_warning_is_visible_but_does_not_choose_the_winner():
    result=summarize([candidate("step2000",[.4]*4,lyric_flags=["lyric_proxy_regression"]),candidate("step3400",[.1]*4)],dict(close_call_gap=.1))
    assert result["recommendation"]["label"]=="step2000"
    assert result["ranking"][0]["lyric_flags"]=={"lyric_proxy_regression":4}


def test_prompt_disagreement_is_a_close_call_even_with_larger_mean_gap():
    result=summarize([candidate("step2000",[1.,1.,-.1,-.1]),candidate("step3400",[0.]*4)],dict(close_call_gap=.1))
    assert result["comparison"]["prompt_wins"]==1 and result["comparison"]["close_call"]
    assert result["coverage"]["prompts"]==2


def test_unmatched_or_duplicate_fixtures_fail_closed():
    a=candidate("step2000",[.1]*4);b=candidate("step3400",[.2]*4)
    b["clips"][0]["fixture"]="unmatched"
    with pytest.raises(ValueError,match="Unmatched"):summarize([a,b],dict(close_call_gap=.1))
    a["clips"][0]["fixture"]=a["clips"][1]["fixture"]
    with pytest.raises(ValueError,match="unique"):summarize([a],dict(close_call_gap=.1))


def test_identical_output_across_seeds_is_not_a_clean_recommendation():
    a=candidate("step2000",[.9]*4);a["clips"][1]["candidate"]["sha256"]=a["clips"][0]["candidate"]["sha256"]
    result=summarize([a],dict(close_call_gap=.1))
    assert result["recommendation"] is None
    assert result["ranking"][0]["flags"]["identical_output_across_seeds"]==1


def test_mono_cancellation_and_gap_screen_preserve_original_audio(tmp_path):
    sr=8000;t=np.arange(sr*5)/sr;x=.1*np.sin(2*np.pi*221.3*t)
    normal=tmp_path/"normal.wav";cancelled=tmp_path/"cancelled.wav";gap=tmp_path/"gap.wav"
    sf.write(normal,np.stack([x,x],1),sr,subtype="FLOAT")
    sf.write(cancelled,np.stack([x,-x],1),sr,subtype="FLOAT")
    g=x.copy();g[sr:sr*4]=0;sf.write(gap,np.stack([g,g],1),sr,subtype="FLOAT")
    baseline=measure(normal)
    assert "mono_cancellation" in warnings(measure(cancelled),baseline,baseline)
    assert "long_interior_gap" in warnings(measure(gap),baseline,baseline)
    assert sf.info(normal).duration==5
