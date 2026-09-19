"""Use the existing audio rule with separately versioned uni16 descriptions."""
import argparse
import json
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT))
from analysis.gan_bcap import autonomous_audio as audio

DESCRIPTIONS = {
    'female': ('a song with a feminine sounding lead singing voice','a song with a masculine sounding lead singing voice'),
    'male': ('a song with a masculine sounding lead singing voice','a song with a feminine sounding lead singing voice'),
    'pop': ('contemporary pop music with melodic hooks and polished drums','a song with a plain accompaniment'),
    'hiphop': ('hip hop music with rapped vocals, deep sub bass and trap drums','a song with a plain accompaniment'),
    'rnb': ('R&B music with warm electric piano, syncopated bass and relaxed vocal phrasing','a song with a plain accompaniment'),
    'indie-rock': ('indie rock music with electric guitars, moving bass and a live drum kit','a song with a plain accompaniment'),
    'pop-punk': ('pop punk music with power chords, picked bass and driving live drums','a song with a plain accompaniment'),
    'metal': ('heavy metal music with low distorted guitar riffs and tight double kick drums','a song with a plain accompaniment'),
    'country': ('country music with acoustic guitar, twangy electric guitar and pedal steel','a song with a plain accompaniment'),
    'acoustic-folk': ('acoustic folk music with fingerpicked guitar, upright bass and brushed percussion','a song with a plain accompaniment'),
    'house': ('house music with four on the floor kick, offbeat open hats and rolling bass','a song with a plain accompaniment'),
    'disco-funk': ('disco funk music with rhythmic electric bass, clipped rhythm guitar and dance drums','a song with a plain accompaniment'),
    'kpop': ('K-pop music with sharp synth hooks, precise electronic drums and a glossy chorus','a song with a plain accompaniment'),
    'reggaeton': ('reggaeton music with a dembow drum pattern, sub bass and short keyboard hooks','a song with a plain accompaniment'),
    'afrobeats': ('Afrobeats music with interlocking percussion, syncopated bass and clean guitar plucks','a song with a plain accompaniment'),
    'lofi': ('lo-fi music with mellow electric piano, softly swung drums and warm bass','a song with a plain accompaniment'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',choices=DESCRIPTIONS,required=True)
    p.add_argument('--folders',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audio.CONCEPTS=DESCRIPTIONS
    judge=audio.Judge();records=[]
    rule=dict(audio.RULE,description_version='uni16-descriptions-v1',
        comparison_scope='Compare 600 and 660 within a slider on the same fixtures. Do not rank different sliders or splice into the old gender leaderboard.',
        genre_margin='New genre descriptions use a shared generic accompaniment reference; not a calibrated genre recognition test.')
    for folder in args.folders:
        records.extend(judge.folder(folder,args.id))
        audio.F.write_json(args.output,dict(status='measuring',rule=rule,concept=args.id,records=records))
    audio.F.write_json(args.output,dict(status='complete',rule=rule,concept=args.id,
        descriptions=DESCRIPTIONS,source_sha256=audio.F.file_hash(Path(__file__)),
        records=records,ranking=audio.summarize(records)))


if __name__=='__main__':main()
