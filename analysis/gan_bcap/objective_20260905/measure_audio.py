"""Apply the existing frozen audio measurements; do not fit a new judge."""
from pathlib import Path
import json
import statistics

from slider_selection.features import AudioMeasurer, write_json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    measurer=AudioMeasurer(ROOT/'analysis/gan_bcap/audio_cpu_fp32_cache','cpu')
    records=[]
    for report in (HERE/'audio-asr.json',HERE/'audio-mmd-asr.json'):
        if not report.exists():continue
        for row in json.loads(report.read_text())['records']:
            if any(r['sha256']==row['sha256'] and r['checkpoint']==row['checkpoint'] and r['role']==row['role'] for r in records):continue
            measured=measurer.measure(Path(row['audio']))
            assert measured['sha256']==row['sha256']
            windows=measured['windows']
            value=dict(audio=row['audio'],sha256=row['sha256'],checkpoint=row['checkpoint'],
                seed=row['seed'],role=row['role'],
                female_margin=statistics.mean(w['concept']['female'] for w in windows),
                production_quality=statistics.mean(w['aesthetics']['PQ'] for w in windows),
                enjoyment=statistics.mean(w['aesthetics']['CE'] for w in windows),measurements=measured)
            records.append(value)
            print(json.dumps({k:v for k,v in value.items() if k not in ('audio','measurements','sha256')}),flush=True)
    write_json(HERE/'audio-frozen-metrics.json',dict(records=records,
        interpretation='Uncalibrated diagnostic scores on two development seeds; not a musical-quality acceptance test.'))


if __name__=='__main__':main()
