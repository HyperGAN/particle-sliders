"""Pinned inputs and paths for the homogeneous YuE2 female training run."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
RUN = ROOT / "models/female-yue2-fresh3400-20260916"
PAGE = ROOT / "eval/listen/yue2-female-fresh3400-20260916"
AUDIT = ROOT / "analysis/uni16_release_audit_20260914"
PREVIOUS = ROOT / "models/female-yue2-uni16-600-20260916"
PY = "/ml2/music/.cache/yue2-test-env/bin/python"
MILESTONES = (600,1000,2000,3000,3400)
TRAIN = ROOT / "conceptmod/textsliders/data/prompts-yue2-female.yaml"
EVAL = ROOT / "conceptmod/textsliders/data/prompts-yue2-female-eval.yaml"
CASES = [(row,seed) for row in (2,3) for seed in (1709,2903)]
sys.path.insert(0,str(ROOT))


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def weights(label):
    if label == "reference600": return PREVIOUS / "female-yue2-uni16-600_600.safetensors"
    return RUN / f"female-yue2-fresh_{label}.safetensors"


def state_path(label):
    return PREVIOUS / "female-yue2-uni16-600_state.pt" if label == "reference600" else RUN / f"state-{label}.pt"


def environment(gpu="1"):
    result=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HOME="/ml2/music/.cache/huggingface",
        HF_HUB_OFFLINE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    result.pop("TRANSFORMERS_CACHE",None)
    return result


def prepare_protocol():
    import yaml
    sys.path.insert(0,str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    train=yaml.safe_load(TRAIN.read_text())["rows"]
    evaluation=yaml.safe_load(EVAL.read_text())["rows"]
    assert not ({r["lyrics"] for r in train}&{r["lyrics"] for r in evaluation})
    assert not _artist_name_hit("",json.dumps(train+evaluation))
    policy=read(AUDIT/"selection-policy.json")
    policy=dict(policy, legacy_policy="reference600 is the user-approved prior YuE2 run, not part of this fresh rerun.")
    files=[TRAIN,EVAL,AUDIT/"selection.py",AUDIT/"rank.py",AUDIT/"dsp.py",AUDIT/"waveforms.py",
        ROOT/"slider_selection/features.py",ROOT/"analysis/gan_bcap/autonomous_audio.py",
        *[ROOT/"conceptmod/textsliders"/name for name in ("infer_yue2.py","train_lora_yue2_fresh.py",
            "yue2_uni.py","yue2_backend.py","lm_adv.py","lm_gan.py","lora.py")],*WORK.glob("*.py")]
    spec=dict(version="yue2-female-always-fresh-v1",created=time.time(),horizon=3400,
        batch=1,phase_changes=[],parameter_step_limit=None,rank=8,alpha=8,
        g_lr=.0005,d_lr=.00075,seed=7,history_seed_start=3000001,history_tokens=250,
        history_backend="native_cuda_graph",history_policy="Fresh frozen-base history on every update from step 1",
        prompt_policy="Balanced shuffled passes over four training rows",milestones=list(MILESTONES),
        ranking_policy=policy,rows=[2,3],seeds=[1709,2903],max_tokens=500,nominal_seconds=20,
        reference_feedback="User listened to the prior 600-step comparisons and reported that they work well.",
        prior_checkpoint_sha256=sha(weights("reference600")),gpu=1,
        rendering="torch-eager, AR adapter only, strength 1, matched prompts and seeds",
        sources={str(p.relative_to(ROOT)):sha(p) for p in files})
    old=read(WORK/"protocol.json")
    if old:
        spec["created"]=old["created"]
        if old!=spec: raise ValueError("Pinned campaign inputs or protocol changed")
    else: write(WORK/"protocol.json",spec)
    PAGE.mkdir(parents=True,exist_ok=True)
    write(PAGE/"protocol.json",spec)
    return spec


def verify_protocol():
    spec=read(WORK/"protocol.json")
    if not spec: raise ValueError("Missing campaign protocol")
    for path,expected in spec["sources"].items():
        if sha(ROOT/path)!=expected: raise ValueError(f"Pinned input changed: {path}")
    return spec
