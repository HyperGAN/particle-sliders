#!/usr/bin/env python3
"""Read-only CPU/offline audit; no model loads, encoding, dispatch or admission."""
import argparse
import json
import os
from pathlib import Path
import time

from conceptmod.textsliders.pg09_remaining_admission import CORE,sha
from conceptmod.textsliders.pg09_remaining_offline import (
    stat_model_assets,validate_active_pins,validate_moss_cache,validate_supra_cache,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('offline audit requires CUDA_VISIBLE_DEVICES empty')
    if args.output.exists():
        raise FileExistsError(args.output)
    started=time.monotonic()
    result=dict(schema='remaining-pg09-offline-review-v1',status='validating',
                model_weights_opened=False,native_forward=False,T5_encode=False,dispatch=False,
                canonical_sources_edited=False,active_moss_priority=True)
    result['active_pins_before']=validate_active_pins()
    result['model_assets']=stat_model_assets()
    print(json.dumps(dict(stage='model-assets-and-active-pins',pass_=True)),flush=True)
    result['supra']=validate_supra_cache()
    print(json.dumps(dict(stage='supra-native-cache-schema',pass_=True)),flush=True)
    result['final_form']=validate_moss_cache(progress=lambda value:print(json.dumps(dict(stage='final-form-hashes',**value)),flush=True))
    result['active_pins_after']=validate_active_pins()
    result['source_snapshot']={str(p):sha(p) for p in [
        CORE/'conceptmod/textsliders/pg09_remaining_campaigns.py',
        CORE/'conceptmod/textsliders/pg09_backends.py',
        CORE/'packages/particle-sliders-core/src/particle_sliders/pg09.py']}
    result.update(status='offline_validated_not_admitted',elapsed_seconds=time.monotonic()-started,
                  context_prerequisite='MISSING_DERIVED_ASSET: exact19 native contexts must be created under owner admission')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'],output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
