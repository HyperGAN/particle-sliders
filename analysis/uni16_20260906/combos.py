"""Matched listening examples of two controls together and each separately."""
from contextlib import ExitStack
import html
import json
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT)]
from train import read, write, sha, PAGE


def main():
    import torch
    from safetensors.torch import load_file
    from conceptmod.textsliders import generate_listen as G
    from conceptmod.textsliders.lora import LoRANetwork
    from conceptmod.textsliders.gan_v2.data import validate_prompts
    catalog=read(WORK/'catalog.json');state=read(WORK/'status.json')
    combos=catalog['combinations']+[
        dict(label='Female lead with pop',sliders={'female':.5,'pop':.5}),
        dict(label='Male lead with country',sliders={'male':.5,'country':.5})]
    item=next(x for x in catalog['sliders'] if x['id']=='female')
    fixture=G._load_prompt_row(item['eval_prompts'],2);validate_prompts([fixture])
    output=PAGE/'combos';output.mkdir(parents=True,exist_ok=True)
    weights={key:Path(info['bounded']['weights']) for key,info in state['sliders'].items()}
    spec=dict(combinations=combos,prompts=item['eval_prompts'],prompts_sha256=sha(item['eval_prompts']),
              row=2,seed=515,duration=20,weights={k:dict(path=str(p),sha256=sha(p)) for k,p in weights.items()},
              code_sha256=sha(__file__),interpretation='One new fixture and seed. A listening screen, not evidence of general combination safety or preference.')
    if (output/'spec.json').exists() and read(output/'spec.json')!=spec:raise ValueError('Combination fixtures changed')
    write(output/'spec.json',spec)
    pipe=G._load_pipeline(G.DEFAULT_MODEL_DIR,'cuda:0')
    originals={m:m.forward for m in pipe.language_model.modules() if isinstance(m,torch.nn.Linear)}
    cards=[];records=[]
    for index,combo in enumerate(combos):
        networks={}
        try:
            for key in combo['sliders']:
                net=LoRANetwork(pipe.language_model,multiplier=0.,rank=8,alpha=8.,
                    delimiter='-',target_replace=['Qwen3Attention'],prefix='lora_te',train_method='full').to('cuda:0').eval()
                net.load_state_dict(load_file(str(weights[key])),strict=True)
                networks[key]=net
            cases=[('Off',{}),*[(key,{key:amount}) for key,amount in combo['sliders'].items()],('Together',combo['sliders'])]
            clips=[]
            for case,(label,scales) in enumerate(cases):
                path=output/f'{index:02d}-{case}-{label.lower()}.wav'
                for key,net in networks.items():net.set_lora_slider(scales.get(key,0.))
                if not path.exists():
                    print('Rendering combination',combo['label'],label,flush=True)
                    with torch.inference_mode(),ExitStack() as stack:
                        for net in networks.values():stack.enter_context(net)
                        result=pipe(prompt=fixture.get('neutral') or fixture['target'],lyrics=fixture['lyrics'],
                            audio_duration=20,generator=torch.Generator('cuda:0').manual_seed(515),output='audios')[0]
                    stats=G._write_wav(path,result,int(pipe.sampling_rate),20,accept_short=True,accept_silent=True)
                else:stats=G._inspect_wav(path)
                records.append(dict(combination=combo['label'],case=label,scales=scales,audio=str(path),
                                    sha256=sha(path),inspection=stats))
                clips.append(f'<p>{html.escape(label)}<audio controls preload="none" src="{path.name}"></audio></p>')
            cards.append(f'<section><h2>{html.escape(combo["label"])}</h2><p>{html.escape(str(combo["sliders"]))}</p>'+''.join(clips)+'</section>')
            write(output/'results.json',dict(status='rendering',spec=spec,records=records))
            (output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Style combinations</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;background:#131a20;color:#e6edf1;font:16px system-ui}audio{display:block;width:100%}section{margin:30px 0}a{color:#9bd1ff}</style><h1>Style combinations</h1><p>Same caption, lyrics and seed. Compare each slider alone with both together at the displayed strengths. These are first samples for listening.</p><a href="../index.html">All 16 sliders</a>'''+''.join(cards))
        finally:
            # Nested LoRA wrappers must be removed from the original Linear,
            # not from another wrapper's bound method.
            for module,forward in originals.items():module.forward=forward
            networks.clear();torch.cuda.empty_cache()
    write(output/'results.json',dict(status='complete',spec=spec,records=records))


if __name__=='__main__':main()
