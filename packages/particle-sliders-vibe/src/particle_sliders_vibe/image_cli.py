"""Product CLI for creating one character-vibe slider from a phrase."""
import argparse
from pathlib import Path
import os
import re

from . import collection_lock


def entry(argv, *, product, env_model, default_model, host_factory):
    p = argparse.ArgumentParser(prog=product)
    p.add_argument("--vibe", required=True, metavar="PHRASE", help="train a character-vibe slider")
    p.add_argument("--model",type=Path,default=os.environ.get(env_model,default_model))
    p.add_argument("--output",type=Path,help="run directory to create or resume")
    p.add_argument("--measure",action="store_true",help="score the eight ranking and sixteen confirmation cases")
    a = p.parse_args(argv)
    if not a.vibe.strip():
        p.error("provide a nonempty phrase")
    slug = re.sub(r"[^\w-]+","-",a.vibe.strip()).strip("-") or "slider"
    root = (a.output or Path.cwd()/"vibes"/product/slug).resolve()
    # Captures must exactly replay the selected images, also after process restart.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from .image_run import ImageRun
    from .image_workflow import finish
    with collection_lock(root):
        host = host_factory(a.model)
        try:
            result = finish(ImageRun(root,host,a.vibe),measure=a.measure)
            print(result["slider"]["path"],"alpha",result["alpha"],flush=True)
            print("Teacher comparisons", root / "teachers.html", flush=True)
            if a.measure:
                for panel in ("ranking", "confirmation"):
                    measured = result[panel]
                    print(f"{panel} score {measured['score']:.4f}, "
                          f"{measured['working']}/{len(measured['rows'])} working", flush=True)
                structure = result["structure_summary"]
                print(f"Structure: {structure['pass']} intact, {structure['fail']} broken, "
                      f"{structure['unmeasured']} uncertain", flush=True)
        finally:
            host.runtime.mixer.close()
