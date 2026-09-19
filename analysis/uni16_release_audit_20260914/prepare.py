import time
from common import *
from analysis.uni16_20260906.score import DESCRIPTIONS
from analysis.gan_bcap.autonomous_audio import RULE
from app.rewriter import _artist_name_hit


def main():
    if (WORK / "protocol.json").exists():
        verify(); return
    assert not _artist_name_hit("", str(DESCRIPTIONS))
    paths = [ROOT / name for name in ("analysis/uni16_20260906/score.py",
        "analysis/gan_bcap/autonomous_audio.py", "slider_selection/features.py", "scripts/lm_score.py",
        "conceptmod/textsliders/gan_v2/metrics.py")]
    paths += [WORK / name for name in ("common.py", "dsp.py", "worker.py", "rank.py")]
    write(WORK / "protocol.json", dict(version=1, created=time.time(),
        training_manifest_sha256=sha(CAMPAIGN / "manifest.json"), sources={str(p):sha(p) for p in paths},
        candidates=["step600", "published660", "step1000", "step2000", "step3000", "step3400"],
        controls=["off", "caption"], rows=[2,3], seeds=[1709,2903], duration=20., strength=1.,
        measurement_rule=RULE, descriptions=DESCRIPTIONS,
        ranking_weights=dict(concept=.5, enjoyment=.25, production=.25, artifacts=.25),
        lyric_policy="User reports lyrics consistently fine. Read already-cached ASR as an informational check; no new transcription pass. Missing lyric measurements remain null. Lyrics do not affect ranking or eligibility.",
        rank_policy="Within each slider only: integrity/technical issues first, then the fixed style/quality score. Use the existing component scales and clipping, omit lyrics and renormalize positive weights. No automatic release.",
        close_call_gap=.1, close_call_interpretation="Heuristic reporting tolerance, not statistical equivalence or a calibrated audible threshold.",
        coverage="Two held-out arrangements with two seeds each. Four clips are not four independent prompts. These monitoring clips support a provisional shortlist, not release certification.",
        diversity="Exact cross-seed duplicates checked. Two seeds per prompt are insufficient to clear the existing four-seed diversity gate. Broad diversity, pitch artifacts, full-song structure and slider combinations remain unvalidated.",
        stopping="Score every currently available complete four-case candidate, then follow newly rendered milestones through each approved training budget. Use CPU only.",
        references={"audiobox":"https://github.com/facebookresearch/audiobox-aesthetics",
                    "clap":"https://huggingface.co/laion/clap-htsat-unfused"}))
    print("Frozen audit rule and metric sources; two prompt groups, four matched cases per candidate.")


if __name__ == "__main__": main()
