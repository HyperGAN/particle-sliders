"""LTX connector GAN orchestration; the shared game owns every update.

Only synthetic CPU fixtures are admitted here. Production text-only loading,
full connector geometry/memory and audiovisual rendering need separate review.
The portable payload is a shared protocol-bound checkpoint, never native LoRA.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

SCHEMA = 'ltx25-shared-gan-cli-v1'
LIVE_GATE = ('LTX production runtime is not qualified: verified staged text-only loader, '
             'full shape/mask, source/license, memory and audiovisual gates remain pending; '
             'only --dummy CPU software fixtures are admitted')
DEFAULT_MODEL = 'Lightricks/LTX-2.5-Diffusers'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dummy', action='store_true')
    p.add_argument('--name', default='ltx25-shared-gan')
    p.add_argument('--model_id', default=DEFAULT_MODEL)
    p.add_argument('--transformer_subfolder', default='transformer')
    p.add_argument('--recipe', default='ltx25_uni_embed', choices=('ltx25_uni_embed', 'embed', 'ltx25_uni_velocity', 'velocity'))
    p.add_argument('--prompts_file')
    p.add_argument('--attributes', default='')
    p.add_argument('--config_file')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=float, default=16.)
    p.add_argument('--te_last_n', type=int, default=4)
    p.add_argument('--lora_up_init_std', type=float, default=.02)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--steps', type=int, default=700, help='Fixed training horizon; 0 loads an inference export')
    p.add_argument('--stop_after', type=int, help='Absolute completed-step boundary within the fixed horizon')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', default='cpu')
    p.add_argument('--encoder_device', default='cpu')
    p.add_argument('--preset', choices=('bispf', 'bispf_optimized'), default='bispf_optimized')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--sampling', choices=('balanced', 'replacement'), default='balanced')
    p.add_argument('--pg_source_profile', choices=('release', 'alz-continuous-v1'), default='release')
    p.add_argument('--penalty_input_domain', choices=('auto', 'continuous'), default='auto')
    p.add_argument('--save_dir', default='models/ltx25-shared-gan')
    p.add_argument('--resume')
    p.add_argument('--load_export')
    p.add_argument('--load_ltx_lora', help='Closed: historical native weights have no shared protocol binding')
    p.add_argument('--sample_scales', default='0,0.5,1')
    p.add_argument('--no_sample', action='store_true', help='Skip response probes; this CLI never writes audiovisual clips')
    p.add_argument('--diag', action='store_true', help='Closed here; use the separate CPU diagnostic fixtures')
    return p.parse_args(argv)


def _validate(args):
    if not args.dummy:
        raise RuntimeError(LIVE_GATE)
    if args.load_ltx_lora:
        raise ValueError('legacy native LoRA lacks shared source/protocol binding; use --load_export or --resume')
    if args.diag or args.config_file or args.attributes:
        raise ValueError('legacy diagnostics/config/attribute overrides are not shared GAN training inputs')
    if args.recipe not in ('ltx25_uni_embed', 'embed'):
        raise ValueError('historical velocity training CLI is closed; only the connector GAN protocol is admitted')
    if args.model_id != DEFAULT_MODEL or args.transformer_subfolder != 'transformer':
        raise ValueError('fixture model identity or transformer subfolder differs')
    if args.device != 'cpu' or args.encoder_device != 'cpu':
        raise ValueError('CPU fixture requires cpu device and encoder_device')
    if args.steps < 0 or args.rank < 1 or not 1 <= args.te_last_n <= 6:
        raise ValueError('invalid horizon, rank or TE layer count')
    if any(not math.isfinite(x) or x <= 0 for x in (args.alpha, args.lr)):
        raise ValueError('alpha and learning rate must be finite and positive')
    if not math.isfinite(args.lora_up_init_std) or args.lora_up_init_std < 0:
        raise ValueError('invalid LoRA initialization')
    if args.resume and args.load_export:
        raise ValueError('choose a full resume or an inference export')
    if args.load_export and args.steps != 0:
        raise ValueError('inference export cannot resume training; set --steps 0')
    if args.steps == 0 and not args.load_export:
        raise ValueError('--steps 0 requires --load_export')
    if args.stop_after is not None and not 0 <= args.stop_after <= args.steps:
        raise ValueError('stop_after must be within the fixed horizon')
    parse_sample_scales(args.sample_scales)


def parse_sample_scales(text):
    scales = [float(x) for x in str(text).split(',') if x.strip()]
    if not scales or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scales):
        raise ValueError('connector response probes require finite unipolar scales in [0,1]')
    return scales


def load_slider_rows(prompts_file=None, cli_attributes=''):
    if cli_attributes:
        raise ValueError('attribute overrides are not part of the connector GAN protocol')
    if prompts_file is None:
        raw = [{'neutral': 'portrait person neutral', 'positive': 'portrait person smile', 'concept_words': 'smile'},
               {'neutral': 'portrait child neutral', 'positive': 'portrait child happy', 'concept_words': 'happy'}]
    else:
        path = Path(prompts_file)
        if path.suffix.lower() == '.json':
            raw = json.loads(path.read_text())
        else:
            import yaml  # Optional YAML reader; JSON needs no extra dependency.
            raw = yaml.safe_load(path.read_text())
        if isinstance(raw, dict):
            shared = raw.get('concept_words', '')
            raw = [dict(item, concept_words=item.get('concept_words', shared))
                   for item in raw.get('prompts', raw.get('rows', []))]
    if not isinstance(raw, list) or not raw:
        raise ValueError('nonempty neutral/positive caption rows required')
    rows = []
    for row in raw:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                           for k in ('neutral', 'positive')):
            raise ValueError('rows require nonempty neutral and positive captions')
        concept = row.get('concept_words', '')
        if not isinstance(concept, str):
            raise ValueError('concept_words must be text')
        rows.append(dict(neutral=row['neutral'], positive=row['positive'], concept_words=concept,
                         unconditional=str(row.get('unconditional', row.get('negative', ''))),
                         target=str(row.get('target', row['neutral'])),
                         attributes=list(row.get('attributes', [])), bare_captions=True))
    return rows


def sample_prompts_from_rows(rows, max_rows=None):
    return list(dict.fromkeys(row['neutral'] for row in rows))[:max_rows]


def build_backend(args):
    _validate(args)
    from .ltx25_backend import LTX25Backend
    return LTX25Backend(dummy=True, lora_rank=args.rank, lora_alpha=args.alpha,
                        te_last_n=args.te_last_n, lora_up_init_std=args.lora_up_init_std)


def create_session(args, *, rows=None, backend=None):
    """Build the actual shared game, binding complete neutral/mask conditions."""
    _validate(args)
    if args.steps == 0:
        raise ValueError('create_session requires a positive training horizon')
    # Gate the native dummy geometry BEFORE any backend allocation/forward.
    from .pg09_text_video import require_continuous_vector
    domain = require_continuous_vector(1026 * 8, 'LTX connector fixture',
                 pg_source_profile=args.pg_source_profile, penalty_input_domain=args.penalty_input_domain)
    import torch
    from .pg09 import ConditionalPG09
    from .pg09_ltx_embed import LTX25ConnectorResponseAdapter, LTXEmbedRow
    from .pg09_readout_critic import ConditionalReadoutCritic
    rows = deepcopy(rows if rows is not None else load_slider_rows(args.prompts_file, args.attributes))
    # Initialize frozen fixtures and critic without advancing caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        backend = backend if backend is not None else build_backend(args)
        if not backend.dummy:
            raise RuntimeError(LIVE_GATE)
        adapter = LTX25ConnectorResponseAdapter(backend, native_shape=(1, 1026, 8),
                  model_identity={'qualification': 'synthetic CPU fixture only', 'seed': args.seed},
                  pg_source_profile=args.pg_source_profile, penalty_input_domain=args.penalty_input_domain)
        prepared = [adapter.prepare(LTXEmbedRow(row['neutral'], row['positive'], row['concept_words'])) for row in rows]
        conditions = torch.cat([torch.cat((r.neutral.flatten(), r.row.common_mask.flatten().float(),
                                          torch.tensor([1., 0.])))[None] for r in prepared])
        critic = ConditionalReadoutCritic(prepared[0].target.numel(), conditions.shape[1])
        game = ConditionalPG09(adapter, prepared, critic, conditions=conditions, particles=None,
                    preset=args.preset, batch_size=args.batch_size, seed=args.seed, total_steps=args.steps,
                    sampling=args.sampling, recipe_overrides={'lr': args.lr}, **domain)
    return adapter, prepared, game, rows


def _session_options(args):
    fields = ('dummy', 'model_id', 'transformer_subfolder', 'recipe', 'rank', 'alpha', 'te_last_n',
              'lora_up_init_std', 'lr', 'steps', 'seed', 'device', 'encoder_device', 'preset', 'batch_size',
              'sampling', 'pg_source_profile', 'penalty_input_domain')
    return {k: getattr(args, k) for k in fields}


def _read_packet(path, kind):
    import torch
    packet = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(packet, dict) or set(packet) != {'schema', 'kind', 'options', 'rows', 'game'}
            or packet['schema'] != SCHEMA or packet['kind'] != kind or not isinstance(packet['options'], dict)):
        raise ValueError('incompatible shared LTX payload; full checkpoint and inference export are distinct')
    defaults = parse_args(['--dummy'])
    if set(packet['options']) != set(_session_options(defaults)):
        raise ValueError('invalid LTX session options')
    return packet


def _probes(adapter, prepared, scales):
    return [{'row': i, 'scale': scale, 'response': adapter.render(row, scale=scale).cpu()}
            for i, row in enumerate(prepared) for scale in scales]


def train(args, backend=None):
    _validate(args)  # Closed routes stop before checkpoint/prompt/native reads.
    import torch
    if args.load_export:
        packet = _read_packet(args.load_export, 'inference')
        restored = parse_args(['--dummy'])
        for key, value in packet['options'].items():
            setattr(restored, key, value)
        adapter, prepared, game, rows = create_session(restored, rows=packet['rows'], backend=backend)
        game.load_export_state_dict(packet['game'])
        result = {'schema': SCHEMA, 'mode': 'inference', 'quality_qualified': False,
                  'audiovisual_render_qualified': False, 'completed_steps': packet['game']['completed_steps']}
    else:
        packet = _read_packet(args.resume, 'training') if args.resume else None
        if packet and packet['options'] != _session_options(args):
            raise ValueError('resume options differ from fixed session contract')
        adapter, prepared, game, rows = create_session(args, backend=backend)
        if packet:
            if rows != packet['rows']:
                raise ValueError('resume captions differ')
            game.load_state_dict(packet['game'])
        stop = args.steps if args.stop_after is None else args.stop_after
        if stop < game.completed_steps:
            raise ValueError('stop boundary precedes resumed checkpoint')
        history = [game.step() for _ in range(game.completed_steps, stop)]
        result = {'schema': SCHEMA, 'mode': 'training', 'completed_steps': game.completed_steps,
                  'history': history, 'quality_qualified': False, 'audiovisual_render_qualified': False,
                  'objective': 'shared GAN only', 'particles': None, 'generator_arm': 'adapter_gan'}
    # Obtain validated payloads/probes before creating any destination artifacts.
    if not args.load_export:
        checkpoint = dict(schema=SCHEMA, kind='training', options=_session_options(args), rows=rows,
                          game=game.state_dict())
        export = dict(checkpoint, kind='inference', game=game.export_state_dict())
    probes = [] if args.no_sample else _probes(adapter, prepared, parse_sample_scales(args.sample_scales))
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.load_export:
        torch.save(checkpoint, out / 'checkpoint.pt')
        torch.save(export, out / 'adapter-export.pt')
    if probes:
        torch.save(probes, out / 'response-probes.pt')
    result['backend'] = adapter.metadata()
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def emit_ltx_samples(*args, **kwargs):
    raise RuntimeError('audiovisual rendering is not qualified; this API only exports connector response probes')


def write_ltx_sample_mp4(*args, **kwargs):
    raise RuntimeError('audiovisual encoding is not qualified; placeholder MP4 files are forbidden')


def main(argv=None):
    return train(parse_args(argv))


if __name__ == '__main__':
    main()
