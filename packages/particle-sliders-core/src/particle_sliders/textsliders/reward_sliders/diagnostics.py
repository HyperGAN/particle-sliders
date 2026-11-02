"""Independent intent measurements, never mixed into the CE objective."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import torch

from .specs import WORKSPACE, digest, sha, write_json, read_json
from .reward import scoring_windows


class IntentDiagnostics:
    def __init__(self, cache):
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
        self.clap = self.processor = self.asr = None

    def load(self):
        from transformers import ClapModel, ClapProcessor
        from slider_selection.features import CLAP
        from scripts.lm_score import WhisperBackend
        torch.set_num_threads(4)
        self.processor = ClapProcessor.from_pretrained(CLAP[0], revision=CLAP[1], local_files_only=True)
        self.clap = ClapModel.from_pretrained(CLAP[0], revision=CLAP[1], local_files_only=True).cpu().eval()
        self.asr = WhisperBackend(self.cache/'asr',device='cpu')

    def measure(self, path, family):
        import soundfile as sf
        from torchaudio.functional import resample
        from slider_selection.features import CLAP, lyric_features
        description = family['caption']
        voice_descriptions = ['a song with a feminine sounding lead singing voice',
                              'a song with a masculine sounding lead singing voice',
                              'instrumental music without vocals']
        key = digest([sha(path),description,voice_descriptions,CLAP,'intent-v2-full-disjoint',sha(__file__)])
        cache = self.cache/f'{key}.json'
        if cache.exists():
            return read_json(cache)
        try:
            if self.clap is None:
                self.load()
            data, rate = sf.read(path,dtype='float32',always_2d=True)
            rms = float(np.sqrt(np.mean(data.astype('float64')**2)))
            normalized = data*(.1/max(rms,1e-12))
            mono = torch.from_numpy(normalized.mean(1))
            with torch.inference_mode():
                inputs = self.processor.tokenizer([description]+voice_descriptions, padding=True,
                                                  truncation=True, return_tensors='pt')
                output = self.clap.get_text_features(**inputs)
                text_features = output.pooler_output if hasattr(output,'pooler_output') else output
                embeddings, weights, window_rms = [], [], []
                for start,end in scoring_windows(len(data),rate,True):
                    wav = resample(mono[start:end],rate,48000).numpy()
                    inputs = self.processor.feature_extractor(wav,sampling_rate=48000,return_tensors='pt')
                    output = self.clap.get_audio_features(**inputs)
                    feature = output.pooler_output if hasattr(output,'pooler_output') else output
                    embeddings.append(feature[0].float().numpy()); weights.append(end-start)
                    window_rms.append(float(np.sqrt(np.mean(data[start:end].astype('float64')**2))))
            mean_embedding = np.average(embeddings,axis=0,weights=weights)
            similarities = mean_embedding @ text_features.float().numpy().T
            transcript = []
            # The existing ASR backend accepts only one 30-second model window;
            # explicitly cover the entire raw song instead of silently truncating.
            self.asr._load()
            for start in range(0,len(data),30*rate):
                audio = self.asr._to_16k(data[start:start+30*rate],rate)
                inputs = self.asr._proc(audio,sampling_rate=16000,return_tensors='pt').input_features
                with torch.inference_mode():
                    ids = self.asr._model.generate(inputs,language='en',task='transcribe',do_sample=False,
                                                    num_beams=1,max_new_tokens=256)
                transcript.append(self.asr._proc.batch_decode(ids,skip_special_tokens=True)[0].strip())
            text = ' '.join(transcript)
            sys.path.insert(0,str(WORKSPACE))
            from app.rewriter import _artist_name_hit
            if _artist_name_hit('',text):
                raise ValueError('ASR transcript rejected by name validation; text not persisted')
            lyrics = (dict(transcribed_word_count=len(text.split()),instrumental_word_rate=len(text.split())*60/(len(data)/rate),instrumental=True) if family['voice']=='instrumental'
                      else lyric_features(family['lyrics'],text))
            level_mean=float(np.average(window_rms,weights=weights))
            level_std=float(np.sqrt(np.average((np.array(window_rms)-level_mean)**2,weights=weights)))
            result = dict(valid=True, style_similarity=float(similarities[0]),
                          voice_similarities=dict(zip(['female','male','instrumental'],map(float,similarities[1:]))),
                          lyrics=lyrics, transcript=text, embedding=mean_embedding.tolist(),
                          dynamics=dict(rms_db=float(20*np.log10(max(rms,1e-12))),
                              crest_db=float(20*np.log10(max(float(np.abs(data).max()),1e-12)/max(rms,1e-12))),
                              window_rms_cv=level_std/max(level_mean,1e-12)),
                          protocol=dict(clap=CLAP,description=description,asr_model=self.asr.model_id,
                                        audio_sha256=sha(path),window_seconds=10,asr_window_seconds=30))
        except Exception as exc:
            result = dict(valid=False,error=f'{type(exc).__name__}: {exc}')
        write_json(cache,result)
        return result


def preservation_tolerances(baselines):
    """Freeze two within-family SDs, with explicit measurement-resolution floors."""
    metrics = {
        'PQ': lambda o:o['reward']['diagnostics']['axes']['PQ'],
        'style_similarity': lambda o:o['intent']['style_similarity'],
        'voice_similarity': lambda o:o['intent']['voice_similarities'][o['voice']],
        'phrase_accuracy': lambda o:o['intent']['lyrics']['phrase_accuracy'],
        'clipped_fraction': lambda o:o['reward']['diagnostics']['clipped_fraction'],
        'rms_db': lambda o:o['intent']['dynamics']['rms_db'],
        'crest_db': lambda o:o['intent']['dynamics']['crest_db'],
        'window_rms_cv': lambda o:o['intent']['dynamics']['window_rms_cv'],
        'instrumental_word_rate': lambda o:o['intent']['lyrics']['instrumental_word_rate'],
    }
    floors = dict(PQ=.1,style_similarity=.01,voice_similarity=.01,phrase_accuracy=.05,clipped_fraction=.001,
                  rms_db=2.,crest_db=1.,window_rms_cv=.1,instrumental_word_rate=3.)
    tolerances = {}
    for name,get in metrics.items():
        families = {}
        for row in baselines:
            try: families.setdefault(row['family'],[]).append(get(row))
            except (KeyError,TypeError): pass
        deviations = [float(np.std(v,ddof=1)) for v in families.values() if len(v)>1]
        tolerances[name] = max(floors[name],2*float(np.median(deviations))) if deviations else None
    return dict(rule='two median within-family sample SDs, with predeclared measurement floors',
                scope='pilot short-screen variability; duration/silence/diversity limits are predeclared engineering bounds, not long-song population estimates',
                values=tolerances, duration_relative=.35, silent_fraction_increase=.05,
                diversity_cosine_similarity_increase=.05)
