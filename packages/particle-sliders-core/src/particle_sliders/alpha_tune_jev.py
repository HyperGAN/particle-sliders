"""Jev observations for finished-image alpha tuning with durable call receipts."""

import base64
from dataclasses import dataclass
import fcntl
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .alpha_search import AlphaObservation
from .candidate_failure import FileRecord, _canonical, _write

JEV_MODEL = "typesafe/jev-1.14"
JEV_URL = "https://api.nano-gpt.com/api/v1/decisions"
_AMBIGUOUS = .5
_FAIL = .8

_FRAME_CHECKS = {
    "unusable_frame": "Is image_1 blank, badly corrupted, or lacking any coherent finished scene?",
    "collage": "Is image_1 split into multiple pictures, frames, panels, or a collage instead of one continuous scene?",
    "duplicated_subject": (
        "Is the main subject in image_1 obviously duplicated by an accidental repeated head, body, or copy? "
        "A coherent scene containing several distinct subjects is acceptable."
    ),
    "broken_anatomy": (
        "Does the main subject in image_1 have visually obvious accidental detached, fused, or misplaced "
        "body parts that break its anatomy? Deliberate, coherent stylization or creature anatomy is acceptable."
    ),
    "incoherent_texture": (
        "Do clearly scrambled or disintegrating textures make a substantial part of image_1 "
        "visually incoherent? Coherent textures, paint, glow, and dramatic lighting are acceptable."
    ),
}

_FRAME_CLARIFICATIONS = {
    "unusable_frame": "Is image_1 recognizable as a coherent finished scene with an identifiable subject?",
    "collage": "Can image_1 be read as one continuous shot with no unintended panels or pasted duplicate views?",
    "duplicated_subject": "Does the main subject read as one coherent figure, with no obvious accidental doubled head or body?",
    "broken_anatomy": (
        "Is the main figure visually coherent enough for a finished illustration? Accept ordinary anime "
        "stylization, simplified forms, heavy clothing, foreshortening and limbs hidden by the crop. "
        "Decline only for a clearly visible detached, fused or misplaced body part."
    ),
    "incoherent_texture": (
        "Do the visible forms in image_1 remain legible? Accept detailed armor, brushwork, lighting and intentional "
        "texture. Decline only if obvious disintegration obscures a substantial part of the composition."
    ),
}
_PAIR_CLARIFICATIONS = {
    "identity_replaced": (
        "Is it visually plausible that image_2 shows the original person in image_1 after an outfit or expression edit? "
        "Accept coherent stylization and small face or hairstyle variations. Decline if the person is visibly replaced."
    ),
    "scene_lost": (
        "Is the original location still recognizable in image_2 from one or more major visible scene anchors? "
        "Accept tighter framing, palette changes, and shifts to individual details."
    ),
}

@dataclass(frozen=True)
class JevAssessment:
    observation: AlphaObservation
    checks: dict[str, float]


@dataclass(frozen=True)
class _Image:
    raw: bytes
    media_type: str
    record: FileRecord

    @classmethod
    def read(cls, record):
        if not isinstance(record, FileRecord):
            raise ValueError("Jev requires a bound finished-image FileRecord")
        raw = Path(record.path).read_bytes()
        if len(raw) != record.bytes or hashlib.sha256(raw).hexdigest() != record.sha256:
            raise ValueError("finished-image bytes differ from the bound FileRecord")
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif raw.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            media_type = "image/webp"
        else:
            raise ValueError("Jev requires a PNG, JPEG, or WebP finished image")
        return cls(raw, media_type, record)

    def identity(self):
        return dict(sha256=self.record.sha256, bytes=self.record.bytes, media_type=self.media_type)

    def content(self):
        data = base64.b64encode(self.raw).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{self.media_type};base64,{data}"}}


def _response_values(body, questions):
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise ValueError("Jev must return typed answers")
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or (model != JEV_MODEL and not model.startswith(JEV_MODEL + "-"))):
        raise ValueError("Jev served an unexpected model")
    answers = body["answers"]
    if set(answers) != set(questions):
        raise ValueError("Jev answer names differ from the request")
    values = {}
    for name, question in questions.items():
        kind = question["type"]
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError(f"Invalid Jev answer type for {name}")
        upper = len(question["criteria"]) - 1 if kind == "score" else 1
        number = answer.get(kind)
        if type(number) not in (int, float) or not 0 <= number <= upper:
            raise ValueError(f"Invalid Jev value for {name}")
        values[name] = float(number) / upper
    return values


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


class JevJudge:
    """Score each image alone and apply separate structure and preservation checks."""

    def __init__(self, receipts: Path, concept: str):
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("a visible concept description is required")
        self.receipts = Path(receipts)
        self.concept = concept.strip()

    def _decide(self, images, state, questions):
        binding = dict(endpoint=JEV_URL, model=JEV_MODEL, state=state, questions=questions,
                       images=[image.identity() for image in images])
        key = _digest(binding)
        self.receipts.mkdir(parents=True, exist_ok=True)
        complete = self.receipts / f"{key}.json"
        pending = self.receipts / f"{key}.pending"
        with (self.receipts / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pending_exists = pending.exists()
            if pending_exists and json.loads(pending.read_text()) != binding:
                raise ValueError("Jev in-flight receipt does not bind the request")
            if complete.exists():
                receipt = json.loads(complete.read_text())
                if (receipt["request"] != binding
                        or receipt["response_sha256"] != _digest(receipt["response"])):
                    raise ValueError("Jev receipt does not bind the request and response")
                if receipt["status"] == "ok":
                    return _response_values(receipt["response"], questions)
                if receipt["status"] not in ("http_error", "invalid_result"):
                    raise ValueError("Jev receipt has an invalid status")
                return None
            if pending_exists:
                return None
            secret = os.environ.get("NANOGPT_API_KEY")
            if not secret:
                raise RuntimeError("NANOGPT_API_KEY is required for uncached Jev observations")
            parts = [state]
            for index, image in enumerate(images, 1):
                parts.extend((f"image_{index}:", image.content()))
            payload = dict(model=JEV_MODEL, state=parts, questions=questions)
            request = Request(JEV_URL, data=_canonical(payload), method="POST",
                              headers={"Content-Type": "application/json"})
            request.add_unredirected_header("Authorization", f"Bearer {secret}")
            _write(pending, binding)
            values = None
            try:
                with urlopen(request, timeout=100) as response:
                    body = json.load(response)
                response_sha256 = _digest(body)
                try:
                    values = _response_values(body, questions)
                    status = "ok"
                except ValueError:
                    status = "invalid_result"
            except HTTPError as error:
                status = "http_error"
                body = {"http_status": error.code}
                response_sha256 = _digest(body)
                error.close()
            except (OSError, HTTPException, ValueError):
                # A request may have reached the service. Keep its marker so
                # interrupted transport never starts a second paid call.
                return None
            _write(complete, dict(request=binding, status=status, response=body, response_sha256=response_sha256))
            pending.unlink()
            return values

    def _frame(self, image):
        questions = {
            "concept": {
                "type": "score",
                "instructions": f"Score the visible strength of this concept in image_1 alone: {self.concept}. "
                                "Use only matching visible evidence. Ignore unrelated changes and general image appeal.",
                "criteria": ["No visible evidence", "Weak visible evidence", "Clear visible evidence",
                             "Pronounced and unmistakable visible evidence"],
            },
            **{name: {"type": "noul", "instructions": question} for name, question in _FRAME_CHECKS.items()},
        }
        return self._decide((image,), "Assess this single finished image from visible evidence.", questions)

    def _pair(self, off, on, preserve):
        state = ("image_1 is the reference and image_2 is the candidate. Compare visible evidence. "
                 f"Preservation context: {preserve}")
        questions = {
            "identity_replaced": {
                "type": "noul",
                "instructions": "Has the recognizable main person in image_1 clearly been replaced by a different "
                                "person in image_2? Coherent changes in outfit, expression, styling, lighting, "
                                "or camera view are acceptable. Answer no if the reference has no main person.",
            },
            "scene_lost": {
                "type": "noul",
                "instructions": "Is the original setting from image_1 almost entirely lost or replaced in image_2? "
                                "Coherent changes in palette, lighting, framing, and individual details are acceptable.",
            },
        }
        return self._decide((off, on), state, questions)

    def _clarify(self, images, state, prompts, checks):
        questions = {f"clear_{name}": {"type": "noul", "instructions": prompt}
                     for name, prompt in prompts.items() if _AMBIGUOUS <= checks.get(name, 0.) < _FAIL}
        if not questions:
            return
        resolved = self._decide(images, state, questions)
        for key, value in (resolved or {}).items():
            if value <= .2 or value >= .8:
                checks[key.removeprefix("clear_")] = 1. - value

    def observe(self, off: FileRecord, on: FileRecord, *, preserve: str) -> JevAssessment:
        """Return independent normalized intensities and named failure probabilities."""
        if not isinstance(preserve, str):
            raise ValueError("preserve must describe the reference as text")
        reference, candidate = _Image.read(off), _Image.read(on)
        original = self._frame(reference)
        edited = self._frame(candidate)
        checks = {key: edited[key] for key in _FRAME_CHECKS} if edited is not None else {}
        comparison = self._pair(reference, candidate, preserve) if original is not None and edited is not None else None
        if comparison is not None:
            checks.update(comparison)
            if not any(value >= _FAIL for value in checks.values()):
                self._clarify((candidate,), "Check the candidate for clear structural defects only.",
                              _FRAME_CLARIFICATIONS, checks)
                if not any(value >= _FAIL for value in checks.values()):
                    self._clarify((reference, candidate), "image_1 is the reference and image_2 is the edit. "
                                  f"Check whether this is a coherent edit of the reference. Preservation context: {preserve}",
                                  _PAIR_CLARIFICATIONS, checks)
        if any(value >= _FAIL for value in checks.values()):
            status = "fail"
        elif comparison is None or any(value >= _AMBIGUOUS for value in checks.values()):
            status = "unmeasured"
        else:
            status = "pass"
        off.verify()
        on.verify()
        observation = AlphaObservation(original["concept"] if original is not None else None,
                                       edited["concept"] if edited is not None else None, status)
        return JevAssessment(observation, checks)
