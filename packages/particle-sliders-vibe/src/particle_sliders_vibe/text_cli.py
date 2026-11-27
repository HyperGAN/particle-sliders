"""Native Glimmer and Tiny Aya prompt and vibe entry point."""

import argparse
from pathlib import Path


def main(argv=None, *, model=None):
    parser = argparse.ArgumentParser()
    if model is None:
        parser.add_argument("--model", required=True, choices=("glimmer", "tiny-aya"))
    else:
        parser.set_defaults(model=model)
    parser.add_argument("--snapshot", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--vibe", help="Train the style described in this phrase")
    mode.add_argument("--prompt", help="Generate a reply")
    parser.add_argument("--out", type=Path, help="New training directory")
    parser.add_argument("--adapter", type=Path, help="A saved text vibe adapter")
    parser.add_argument(
        "--strength", type=float, help="Override the adapter's fitted strength"
    )
    args = parser.parse_args(argv)
    if args.vibe is not None and (
        not args.vibe.strip()
        or args.out is None
        or args.adapter
        or args.strength is not None
    ):
        parser.error("--vibe requires a phrase and --out")
    if args.prompt is not None and (
        args.out
        or not args.prompt.strip()
        or (args.strength is not None and not args.adapter)
    ):
        parser.error("--prompt takes an optional --adapter and its --strength")
    from .text_model import TextModel
    from .text_device import gpu_slot
    from .text_workflow import train_vibe

    with gpu_slot(args.model):
        host = TextModel.load(args.model, args.snapshot)
        try:
            if args.vibe:

                def progress(stage, step, total, message):
                    print(f"{stage} {step}/{total} {message}", flush=True)

                path, _ = train_vibe(host, args.vibe, args.out, progress=progress)
                print(path)
            else:
                strengths = {}
                if args.adapter:
                    meta = host.load_adapter("slider", args.adapter)
                    strengths["slider"] = (
                        args.strength if args.strength is not None else meta["alpha"]
                    )
                print(host.reply(args.prompt, strengths=strengths)["text"])
        finally:
            host.mixer.close()


if __name__ == "__main__":
    main()
