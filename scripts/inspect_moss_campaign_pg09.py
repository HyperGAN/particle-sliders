"""Read-only CPU cache admission report; never loads pretrained model weights."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from conceptmod.textsliders.pg09_moss_campaign import MossCampaignRows, coverage_budget  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-scale', action='store_true', help='verify/read all training bundles for frozen edit RMS')
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.monotonic()
    rows = MossCampaignRows(args.campaign)
    indices = [0, 1, len(rows) - 1]
    sample = [rows[i] for i in indices]
    conditions = rows.conditions(indices)
    report = dict(provider=rows.summary(), coverage=coverage_budget(unsigned_states=len(rows) // 2),
                  conditions_shape=list(conditions.shape), condition_row_ids_used=False,
                  sample_indices=indices, sample_native_times=[float(p.row.timestep) for p in sample],
                  sample_signs=[p.row.scale for p in sample],
                  native_response_shape=list(sample[0].target.shape),
                  samples_finite=bool(all(p.target.isfinite().all() for p in sample)),
                  matched_evaluation=dict(seed=rows.recipe['evaluation_seed'], steps=50,
                                         width=rows.recipe['settings']['resolution'],
                                         height=rows.recipe['settings']['resolution'],
                                         guidance_scales=[3., 3., 1., 3., 3.],
                                         adapter_strengths=[0., 0., 0., -1., 1.],
                                         labels=['off', 'direct-positive', 'direct-positive-cfg1', 'minus', 'plus']),
                  protocol_corrections=['adapter affects conditional branch only', 'negative-layer skip disabled'],
                  old_objective=rows.recipe['settings']['objective'],
                  new_objective='public PG0.9 adversarial + required PG constraints; no MSE',
                  model_weights_loaded=False, cuda_initialized=torch.cuda.is_initialized())
    if args.fit_scale:
        report['training_edit_rms_response_scale'] = rows.response_scale()
        report['scale_policy'] = 'single frozen training-only strength-adjusted RMS over cached pole contrasts'
    report['wall_seconds'] = time.monotonic() - start
    report['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('native_response_shape', 'conditions_shape', 'wall_seconds', 'max_rss_kib')}, indent=2))
    if args.fit_scale:
        print('training_edit_rms_response_scale', report['training_edit_rms_response_scale'])


if __name__ == '__main__':
    main()
