"""CPU-only actual runtime/module/source check under proposed command-card env."""
import json
import importlib.metadata
import argparse
from inspect_audio_pg09_assets import runtime_identity
from particle_sliders.pg09 import validate_pg09_domain

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--family', choices=('yue2', 'music3'), required=True)
args = parser.parse_args()
width = dict(yue2=2048, music3=4096)[args.family]
receipt = dict(family=args.family, native_response_shape=[1, width], runtime=runtime_identity())
receipt['distribution_metadata_directory'] = str(importlib.metadata.distribution('particlegan')._path)
try:
    receipt['domain'] = validate_pg09_domain((1, width), response_domain='continuous_vector',
        pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
except Exception as error:
    receipt['domain_error'] = type(error).__name__ + ': ' + str(error)
print(json.dumps(receipt, indent=2))
