"""Check fixed budgets, all-row sampling, recovery and health-stop semantics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import seed_for, warmup_args, WORK, RUNTIME, read
from render import catastrophic


def test_no_seed_reuse_and_exact_budget():
    seeds=[seed_for(i,s) for i in range(16) for s in range(601,3401)]
    assert len(seeds)==44800 and len(set(seeds))==44800
    for step in (600,3401):
        with pytest.raises(ValueError):seed_for(0,step)


def test_all_four_rows_are_balanced_and_resume_sampler_exactly():
    sys.path.insert(0,str(RUNTIME))
    from conceptmod.textsliders.gan_v2.state import RowSampler
    sampler=RowSampler(4,1,seed=7)
    first=[sampler.next()[0] for _ in range(13)]
    saved=sampler.state_dict()
    expected=[sampler.next()[0] for _ in range(31)]
    restored=RowSampler(4,1,seed=123)
    restored.load_state_dict(saved)
    assert [restored.next()[0] for _ in range(31)]==expected
    sampler=RowSampler(4,1,seed=7)
    rows=[sampler.next()[0] for _ in range(2800)]
    assert [rows.count(i) for i in range(4)]==[700]*4
    assert all(sorted(rows[i:i+4])==[0,1,2,3] for i in range(0,2800,4))


def test_short_natural_ending_does_not_stop_but_catastrophic_audio_does():
    assert not catastrophic([dict(flags=["short"],duration=15.)]*4)
    assert catastrophic([dict(flags=["near_silent"],duration=20.)])
    assert catastrophic([dict(flags=["clipping"],duration=20.)])
    assert catastrophic([dict(flags=["short"],duration=2.)]*4)
    assert not catastrophic([dict(flags=[],duration=20.)]*4)


def test_protocol_checks_every_thousand_and_assigns_both_gpus():
    m=read(WORK / "manifest.json")
    c=read(WORK / "catalog.json")["sliders"]
    assert m["evaluation"]["steps"]==[1000,2000,3000,3400]
    assert len(c)==16 and all(i["fresh_rows"]==[0,1,2,3] for i in c)
    assert sum(i["gpu"]==0 for i in c)==sum(i["gpu"]==1 for i in c)==8
    assert m["continuation"]["end"]==3400 and m["continuation"]["batch"]==1
    assert all(i["warmup_signature"]["settings"]["adv_batch"]==4 for i in c)
