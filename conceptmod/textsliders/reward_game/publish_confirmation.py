"""Publish existing confirmation pages and raw WAV links to the LAN listener."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlparse

from .core import ROOT,read,write,sha,IntegrityError


class AudioSources(HTMLParser):
    def __init__(self):super().__init__();self.sources=[]
    def handle_starttag(self,tag,attrs):
        if tag=='audio':self.sources.append(dict(attrs)['src'])


def publish(home,batch,name):
    home=Path(home).resolve()
    if Path(batch).name!=batch or Path(name).name!=name or name in ('.','..'):
        raise IntegrityError('Use single directory names')
    source=home/'confirmation'/batch;page=source/'index.html'
    card=read(source/'scorecard.json');protocol=read(source/'protocol.json')
    if card['batch']!=batch or protocol['name']!=batch:
        raise IntegrityError('Confirmation identity differs from folder')
    known={r['audio']:r['audio_sha256'] for p in (source/'observations').glob('*.json')
        if (r:=read(p)).get('audio_sha256')}
    parser=AudioSources();parser.feed(page.read_text())
    target=ROOT/'eval/listen'/name;target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for url in sorted(set(parser.sources)):
        parsed=urlparse(url)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise IntegrityError('Expected a local raw recording URL')
        relative=Path(unquote(parsed.path));original=(source/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not original.is_relative_to(source) or original.suffix!='.wav':
            raise IntegrityError('Recording URL escapes its confirmation folder')
        expected=known.get(str(original))
        if expected is None or sha(original)!=expected:
            raise IntegrityError('Raw recording differs from retained observation')
        link=target/relative;link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():
            if link.resolve()!=original:raise IntegrityError('Existing publication points to another recording')
        elif link.exists():raise IntegrityError('Existing publication is not a raw recording link')
        else:link.symlink_to(original)
        rows.append(dict(url=url,source=str(original),audio_sha256=expected))
    # The page keeps its original relative audio URLs and all failed outcomes.
    temporary=target/'index.html.pending';temporary.write_bytes(page.read_bytes());temporary.replace(target/'index.html')
    result=dict(source=str(source),page=str(target/'index.html'),source_page_sha256=sha(page),
        published_page_sha256=sha(target/'index.html'),recordings=rows,new_audio_generated=0,
        batch_pass=card['batch_pass'],valid_cases=card['valid_cases'])
    write(target/'publication.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--batch',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    result=publish(a.home,a.batch,a.name)
    print(__import__('json').dumps({k:v for k,v in result.items() if k!='recordings'},indent=2))
