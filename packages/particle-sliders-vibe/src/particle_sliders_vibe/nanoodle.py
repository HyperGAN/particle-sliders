"""Optional Nanoodle workflow adapters. Judge and model choices stay explicit."""
from copy import deepcopy
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

from . import digest, score_value, file_digest, ProviderRejected


class NanoodleWorkflows:
    """Prompt workflow: Context -> Pairs (JSON list of neutral/positive objects).

    Score workflow: Criterion + Image -> a named Decide score node. Its model
    must be configured explicitly (e.g. the account's Jev route). No fallback
    judge, beauty score, or confidence tiebreak is silently substituted.
    """
    def __init__(self, prompt_graph, score_graph, *, score_node='judge', timeout=120,
                 judge_provider='nanogpt'):
        from nanoodle import Workflow
        self.prompt_graph, self.score_graph = deepcopy(prompt_graph), deepcopy(score_graph)
        nodes = [n for n in score_graph['nodes'] if n['id'] == score_node]
        if len(nodes) != 1 or nodes[0]['type'] != 'decide' or nodes[0]['fields'].get('mode') != 'score':
            raise ValueError('score workflow requires an explicit Decide score node')
        fields = nodes[0]['fields']
        if not fields.get('model') or not fields.get('question'):
            raise ValueError('judge model and criterion question must be explicit')
        self.levels = len([s for s in fields.get('levels', '').splitlines() if s.strip()])
        if not 2 <= self.levels <= 10:
            raise ValueError('judge needs 2 to 10 concrete ordered score levels')
        if judge_provider not in ('nanogpt', 'typesafe'):
            raise ValueError('judge_provider must be nanogpt or typesafe')
        self.score_node, self.timeout, self.judge_provider = score_node, timeout, judge_provider
        self.last_judgment = None
        self.last_rejection = None
        self.last_nanogpt_rejection = None
        self.prompts = Workflow(prompt_graph, http=self._nanogpt_http)
        # Nanoodle owns graph evaluation and its Decide primitive. The optional
        # transport maps that primitive onto TypeSafe's official API, keeping
        # each provider's credential at its own origin.
        self.judge = (Workflow(score_graph, api_key='typesafe-transport', http=self._typesafe_http)
                      if judge_provider == 'typesafe' else Workflow(score_graph, http=self._nanogpt_http))
        self.identity = dict(provider='nanoodle', version=version('nanoodle'),
            adapter_sha256=file_digest(Path(__file__)),
            prompts_sha256=digest(prompt_graph), score_sha256=digest(score_graph), score_node=score_node,
            judge_provider=judge_provider)

    def _nanogpt_http(self, method, url, *, headers=None, body=None, timeout=None):
        from nanoodle.transport import default_http
        response = default_http(method, url, headers=headers, body=body, timeout=timeout)
        if urlsplit(url).hostname == 'nano-gpt.com' and response.status in (400, 401, 402, 403, 404, 413, 422, 429):
            self.last_nanogpt_rejection = response.status
        if urlsplit(url).path == '/api/v1/decisions' and response.status == 200:
            self.last_judgment = json.loads(response.text())
        return response

    def _typesafe_http(self, method, url, *, headers=None, body=None, timeout=None):
        from nanoodle.transport import default_http
        if method != 'POST' or urlsplit(url).path != '/api/v1/decisions':
            raise ValueError('TypeSafe transport accepts only Nanoodle Decide requests')
        key = os.environ.get('TYPESAFE_API_KEY')
        if not key:
            raise ValueError('TYPESAFE_API_KEY is required for the TypeSafe judge')
        payload = typesafe_payload(json.loads(body))
        request_headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        if payload.get('assets'):
            request_headers['TypeSafe-Danger-Zone-Images'] = 'true'
        response = default_http('POST', 'https://api.typesafe.ai/v1/systemone',
            headers=request_headers, body=json.dumps(payload), timeout=timeout)
        if response.status in (400, 401, 402, 403, 404, 413, 422, 429):
            self.last_rejection = response.status
            raise ProviderRejected('TypeSafe', response.status)
        if response.status == 200:
            self.last_judgment = json.loads(response.text())
        return response

    def propose(self, vibe, count, history, *, request_id):
        context = dict(vibe=vibe, count=count, request_id=request_id,
            history=[dict(round=r['index'], prompts=[dict(prompt=p['prompt'], fitness=p['fitness'])
                     for p in r['ranking']]) for r in history])
        self.last_nanogpt_rejection = None
        try:
            result = self.prompts.run({'Context': json.dumps(context)}, timeout=self.timeout)
        except Exception:
            if self.last_nanogpt_rejection is not None:
                raise ProviderRejected('NanoGPT', self.last_nanogpt_rejection) from None
            raise
        if result.errors:
            raise RuntimeError('prompt workflow contains failed nodes')
        text = result['Pairs']
        # A fenced JSON response is accepted; no prose extraction or guessed pairs.
        if text.strip().startswith('```'):
            text = '\n'.join(text.strip().splitlines()[1:-1])
        return json.loads(text)

    def score(self, vibe, output, *, request_id):
        from nanoodle import media_from_file
        self.last_judgment = None
        self.last_rejection = None
        self.last_nanogpt_rejection = None
        try:
            result = self.judge.run({'Criterion': vibe, 'Image': media_from_file(output['path'])}, timeout=self.timeout)
        except Exception:
            if self.last_rejection is not None:
                raise ProviderRejected('TypeSafe', self.last_rejection) from None
            if self.last_nanogpt_rejection is not None:
                raise ProviderRejected('NanoGPT', self.last_nanogpt_rejection) from None
            raise
        if result.errors:
            raise RuntimeError('score workflow contains failed nodes')
        decision = result.nodes[self.score_node].out['decision']
        if self.last_judgment is not None:
            answer = self.last_judgment.get('answers', {}).get('answer', {})
            raw_score = answer.get('score')
            if (answer.get('type') != 'score' or type(raw_score) not in (int, float)
                    or not math.isfinite(raw_score) or not 0 <= raw_score <= self.levels - 1):
                raise ValueError('judge returned no valid measured score')
        # Nanoodle exposes score+1. Use the full decision, never rounded text.
        if decision.get('mode') != 'score' or decision.get('levels') != self.levels:
            raise ValueError('score response does not match the frozen rubric')
        probabilities = [r['p'] for r in decision.get('rows', [])]
        if (len(probabilities) != self.levels
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-4)
                or not math.isclose(sum((i+1)*p for i, p in enumerate(probabilities)), decision['score'], abs_tol=1e-4)):
            raise ValueError('judge returned no valid measured score distribution')
        return score_value(dict(score=(decision['score'] - 1) / (self.levels - 1),
            decision=decision, raw=self.last_judgment, request_id=request_id,
            cost_usd=result.cost_usd if result.cost_exact else None))


def typesafe_payload(decision):
    """Map Nanoodle inline image state to documented TypeSafe asset placeholders."""
    payload = dict(model=decision['model'], questions=decision['questions'])
    state, assets = [], {}
    parts = decision['state'] if isinstance(decision['state'], list) else [decision['state']]
    for part in parts:
        if isinstance(part, str):
            state.append(part)
        elif isinstance(part, dict) and part.get('type') == 'image_url':
            url = part['image_url']['url']
            if not url.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
                raise ValueError('TypeSafe images must be inlined by Nanoodle before dispatch')
            prefix, encoded = url.split(',', 1)
            name = 'vibe_image_' + digest(encoded)[:24]
            assets[name] = dict(type='image', format='png' if 'png' in prefix else 'jpeg', data_base64=encoded)
            state.append(name)
        else:
            raise ValueError('unsupported Nanoodle decision state part')
    payload['state'] = state
    if assets:
        payload['assets'] = assets
    return payload
