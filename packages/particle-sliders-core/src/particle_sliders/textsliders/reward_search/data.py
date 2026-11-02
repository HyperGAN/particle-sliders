"""Train on windows throughout a shared generated history, retaining its prefix."""
import torch

from ..reward_sliders.capture import ResidualCapture
from ..reward_sliders.data import generation_hidden, end_margins, set_scale
from ..reward_sliders.specs import sha, digest


def slices(row):
    return (slice(row['span_start'],row['span_stop'],row['stride']),
            slice(row['span_start']-1,row['span_stop']-1,row['stride']))


@torch.no_grad()
def prepare_window_rows(lm, observations, teacher, apply_styles, *, device, windows, stride=4):
    rows=[]
    for observation in observations:
        if observation['split']!='train':raise ValueError('Only training families may supply student rows')
        if observation['status']!='complete' or sha(observation['trajectory'])!=observation['trajectory_sha256']:
            raise ValueError('Training trajectory is missing or changed')
        trajectory=torch.load(observation['trajectory'],map_location='cpu',weights_only=True)
        prompt,feedback=trajectory['prompt_embeds'],trajectory['frame_embeds']
        if prompt.shape[0]!=2 or feedback.shape[0]!=2:raise ValueError('Both CFG branches are required')
        boundary=prompt.shape[1]
        if any(not 0<=start<stop<=feedback.shape[1] or (stop-start)%stride for start,stop in windows):
            raise ValueError('Invalid window geometry')
        apply_styles(observation['family'])
        for start,stop in windows:
            for branch in range(2):
                embeds=torch.cat([prompt[branch:branch+1],feedback[branch:branch+1,:stop]],1).to(device)
                baseline=generation_hidden(lm,embeds)
                with ResidualCapture(lm,capture=False,prompt_length=boundary,**teacher.hook_kwargs()):
                    positive=generation_hidden(lm,embeds)
                if not torch.equal(baseline[:,:boundary],positive[:,:boundary]):
                    raise RuntimeError('Generation teacher changed the prompt')
                row=dict(family=observation['family'],observation=observation['id'],branch=branch,
                    seed=observation['seed'],trajectory_sha256=observation['trajectory_sha256'],
                    boundary=boundary,span_start=boundary+start,span_stop=boundary+stop,
                    window_start=start,window_stop=stop,stride=stride,embeds=embeds.cpu())
                span,end_span=slices(row)
                neutral=baseline[:,span].float().cpu()
                real=(positive[:,span].float()-baseline[:,span].float()).cpu()
                if not torch.isfinite(real).all() or float(real.norm())<=0:raise ValueError('Invalid reward target')
                row.update(neutral_span=neutral,real=real,condition=neutral,
                           prompt_teacher=baseline[:,:boundary].cpu(),
                           end_teacher=end_margins(lm,baseline[:,end_span]).cpu(),
                           fixture=digest([observation['trajectory_sha256'],teacher.layer,teacher.coefficient,
                                           branch,start,stop,stride]))
                rows.append(row)
        print(f"PREPARED {observation['id']}: {len(windows)} windows, both CFG branches",flush=True)
    return rows


class WindowStudentForward:
    def __init__(self,lm,network,apply_styles,device):
        self.lm,self.network,self.apply_styles,self.device=lm,network,apply_styles,device

    def __call__(self,row,with_policy=False):
        if with_policy:raise ValueError('No additional policy loss in these two comparison arms')
        self.apply_styles(row['family']);set_scale(self.network,1.)
        hidden=generation_hidden(self.lm,row['embeds'].to(self.device))
        span,end_span=slices(row)
        result=dict(fake=hidden[:,span].float()-row['neutral_span'].to(self.device))
        if torch.is_grad_enabled():result['end']=end_margins(self.lm,hidden[:,end_span])
        return result


@torch.no_grad()
def diagnostics(lm,network,rows,apply_styles,device):
    result=[]
    for family in sorted({r['family'] for r in rows}):
        candidates=[r for r in rows if r['family']==family and r['branch']==0]
        for start in sorted({r['window_start'] for r in candidates}):
            row=next(r for r in candidates if r['window_start']==start)
            apply_styles(family);set_scale(network,1.)
            hidden=generation_hidden(lm,row['embeds'].to(device))
            prompt=hidden[:,:row['boundary']].float();original=row['prompt_teacher'].to(device).float()
            span,_=slices(row)
            fake=hidden[:,span].float()-row['neutral_span'].to(device);real=row['real'].to(device)
            result.append(dict(family=family,window_start=start,
                prompt_relative_rms=float((prompt-original).norm()/original.norm().clamp_min(1e-8)),
                generation_relative_error=float((fake-real).norm()/real.norm()),
                residual_cosine=float(torch.nn.functional.cosine_similarity(fake.flatten(),real.flatten(),dim=0))))
    return result
