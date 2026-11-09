"""Persistent catastrophic-output eligibility, separate from any ranking score.

This module is stdlib-only. Products supply genuine finished outputs and the
qualified observer; numerical responses are never interpreted as media here.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


# flock is per open file description, so threads of one process exclude each
# other like processes do. Record the holder so that a nested call from the
# thread running a probe raises instead of waiting for itself.
_HOLDERS, _HOLDERS_LOCK = {}, threading.Lock()


def _holder():
    return os.getpid(), threading.get_ident()


def _in_flight(value):
    return value["status"] == "unmeasured" and any(
        isinstance(row, dict) and row.get("status") == "probe_committed"
        for row in value.get("checkpoints", {}).values())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self):
        if (not isinstance(self.path, str) or not self.path
                or type(self.bytes) is not int or self.bytes < 0
                or not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("complete file identity required")

    @classmethod
    def from_path(cls, path):
        path = Path(path)
        checksum = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
                size += len(block)
        return cls(str(path), size, checksum.hexdigest())

    def verify(self):
        checksum = hashlib.sha256()
        size = 0
        with Path(self.path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
                size += len(block)
        if size != self.bytes or checksum.hexdigest() != self.sha256:
            raise ValueError("actual finished-output bytes differ from the bound source")

    def read_json(self):
        body = Path(self.path).read_bytes()
        if len(body) != self.bytes or hashlib.sha256(body).hexdigest() != self.sha256:
            raise ValueError("file identity differs")
        return json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


@dataclass(frozen=True)
class FinishedOutput:
    output_source: FileRecord
    media: str
    native_contract: dict

    def __post_init__(self):
        if not isinstance(self.output_source, FileRecord) or self.media not in ("image", "text", "audio", "discrete"):
            raise ValueError("finished output with an explicit native media kind required")
        if not isinstance(self.native_contract, dict) or not self.native_contract:
            raise ValueError("minimal native allowed-output semantics required")
        # Candidate settings and expected judgments must not reach the observer.
        forbidden = {"candidate", "candidate_id", "checkpoint", "model", "product", "arm", "lr",
                     "alpha", "loss", "effect_score", "desired_effect", "expected_label", "fixture_role"}
        def visit(value):
            if isinstance(value, dict):
                if set(value) & forbidden:
                    raise ValueError("candidate/label information cannot be supplied to the observer")
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(self.native_contract)
        _canonical(self.native_contract)


@dataclass(frozen=True)
class QualifiedDetector:
    detector_contract: FileRecord
    actual_instrument_source: dict
    actual_qualification: FileRecord | None = None
    judgment_kind: str = "probability"

    def qualification_error(self, media):
        """Missing/invalid/unsupported qualification is never a healthy verdict."""
        try:
            if (not isinstance(self.detector_contract, FileRecord)
                    or not isinstance(self.actual_instrument_source, dict) or not self.actual_instrument_source):
                return "missing detector identity"
            contract = self.detector_contract.read_json()
            if not isinstance(contract, dict):
                return "invalid detector contract metadata"
            if self.judgment_kind not in ("probability", "categorical"):
                return "unknown judgment kind"
            if self.judgment_kind == "probability":
                thresholds = contract["thresholds"]
                if (thresholds["fail_greater_than_or_equal"] != .95
                        or thresholds["pass_less_than_or_equal"] != .05):
                    return "detector thresholds differ from the frozen catastrophic contract"
            if not isinstance(self.actual_qualification, FileRecord):
                return "no actual detector qualification"
            qualified = self.actual_qualification.read_json()
            if not isinstance(qualified, dict):
                return "invalid actual qualification metadata"
            if (qualified.get("schema") != "Universal-catastrophic-detector-actual-qualification-v1"
                    or qualified.get("accepted") is not True or qualified.get("qualified") is not True
                    or qualified.get("detector_contract") != asdict(self.detector_contract)
                    or qualified.get("actual_instrument_source") != self.actual_instrument_source
                    or qualified.get("judgment_kind") != self.judgment_kind):
                return "actual detector qualification does not bind this observer/contract"
            if (type(qualified.get("covered_media")) is not list
                    or media not in qualified["covered_media"]):
                return "actual detector qualification does not cover " + media
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return "unreadable/invalid detector qualification: " + str(exc)
        return None


class CandidateFailure(RuntimeError):
    """Actual qualified catastrophic failure: this candidate stays failed."""


class GuardUnmeasured(RuntimeError):
    """Required output assessment unavailable; not evidence the model is bad."""


class ObservationCache:
    """One durable first per unique output/native contract and observer identity."""
    def __init__(self, root, detector, observe):
        self.root, self.detector, self.observe = Path(root), detector, observe

    def assess(self, output):
        error = self.detector.qualification_error(output.media)
        binding = dict(output_source=asdict(output.output_source), media=output.media,
                       native_contract=output.native_contract,
                       detector_contract=asdict(self.detector.detector_contract),
                       actual_instrument_source=self.detector.actual_instrument_source,
                       judgment_kind=self.detector.judgment_kind)
        # Path/candidate/checkpoint are provenance, not a reason to judge the
        # same physical output again. The adapter sees no candidate metadata.
        identity = dict(binding, output_source={k: binding["output_source"][k] for k in ("bytes", "sha256")})
        key = _digest(identity)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "observer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Admission/configuration errors are diagnostics, not model calls.
            # Retain them separately so later actual qualification can permit
            # the first real call without rerating a completed observation.
            if error is None:
                try:
                    output.output_source.verify()
                except (OSError, ValueError) as exc:
                    error = type(exc).__name__ + ": " + str(exc)
            if error is not None:
                diagnostic = dict(identity=identity, binding=binding, status="unmeasured",
                                  failure_probability=None, categorical_failure=None,
                                  judgment_kind=self.detector.judgment_kind, reason=error,
                                  call_performed=False, actual_qualification=(asdict(self.detector.actual_qualification)
                                                                             if self.detector.actual_qualification else None))
                name = "no-call-" + _digest(diagnostic) + ".json"
                if not (directory / name).exists():
                    _write(directory / name, diagnostic)
                return diagnostic
            target = directory / "observation.json"
            if target.exists():
                saved = json.loads(target.read_text())
                if saved["identity"] != identity:
                    raise ValueError("first-observation identity collision")
                return saved
            call = directory / "call.json"
            raw = None
            invoked = None if call.exists() else False
            if error is None and call.exists():
                error = "committed first call has no durable outcome; no automatic replay"
            if error is None:
                _write(call, dict(status="committed", identity=identity))
                try:
                    invoked = True
                    raw = self.observe(output.output_source, output.media, dict(output.native_contract))
                    output.output_source.verify()
                    _canonical(raw)
                    if not isinstance(raw, dict) or "failure_probability" not in raw:
                        raise ValueError("explicit typed detector result required")
                    probability = raw.get("failure_probability")
                    if self.detector.judgment_kind == "probability":
                        if (type(probability) not in (int, float) or not math.isfinite(probability)
                                or not 0 <= probability <= 1 or raw.get("categorical_failure") is not None):
                            raise ValueError("missing/nonfinite failure probability")
                        expected = "fail" if probability >= .95 else "pass" if probability <= .05 else "unmeasured"
                    else:
                        category = raw.get("categorical_failure")
                        if ("categorical_failure" not in raw or probability is not None
                                or type(category) not in (bool, type(None))):
                            raise ValueError("categorical verdict is not a probability")
                        expected = "unmeasured" if category is None else "fail" if category else "pass"
                    if (raw.get("status") != expected or not isinstance(raw.get("reason"), str)
                            or not raw["reason"].strip()
                            or raw.get("output_source") != asdict(output.output_source)
                            or raw.get("detector_contract") != asdict(self.detector.detector_contract)
                            or raw.get("actual_instrument_source") != self.detector.actual_instrument_source):
                        raise ValueError("observer verdict or source identity differs")
                except Exception as exc:
                    error = type(exc).__name__ + ": " + str(exc)
                    try:
                        _canonical(raw)
                    except (TypeError, ValueError):
                        raw = {"unserializable_response_type": type(raw).__name__}
            result = dict(identity=identity, binding=binding, observer_invoked=invoked,
                          actual_qualification=asdict(self.detector.actual_qualification) if self.detector.actual_qualification else None,
                          status="unmeasured" if error else raw["status"],
                          failure_probability=None if error else raw.get("failure_probability"),
                          categorical_failure=None if error else raw.get("categorical_failure"),
                          judgment_kind=self.detector.judgment_kind,
                          reason=error or raw["reason"], raw_response=raw)
            _write(target, result)
            return result


class CandidateGuard:
    """Arm state outside scientific checkpoints; failed cannot be reset by resume."""
    def __init__(self, path, *, candidate_id, scientific_protocol, media, cache):
        self.path, self.cache = Path(path), cache
        self.binding = dict(candidate_id=candidate_id, scientific_protocol=scientific_protocol,
                            media=media, detector_contract=asdict(cache.detector.detector_contract),
                            actual_instrument_source=cache.detector.actual_instrument_source,
                            judgment_kind=cache.detector.judgment_kind)
        _canonical(self.binding)
        if (not isinstance(candidate_id, str) or not candidate_id
                or not isinstance(scientific_protocol, dict) or not scientific_protocol
                or media not in ("image", "text", "audio", "discrete")):
            raise ValueError("candidate/protocol/native media binding required")
        with self._lock():
            if not self.path.exists():
                _write(self.path, dict(schema="particle-sliders-candidate-failure-state-v1",
                                      binding=self.binding, status="pending", checkpoints={}))
            self._read()

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = self.path.with_name(self.path.name + ".lock")
        key = str(name.resolve())
        if self._held_here(key):
            raise GuardUnmeasured("nested guard call from the thread already holding this candidate's guard")
        with name.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with _HOLDERS_LOCK:
                    _HOLDERS[key] = _holder()
                yield
            finally:
                with _HOLDERS_LOCK:
                    _HOLDERS.pop(key, None)

    def _held_here(self, key=None):
        key = key or str(self.path.with_name(self.path.name + ".lock").resolve())
        with _HOLDERS_LOCK:
            return _HOLDERS.get(key) == _holder()

    def _read(self):
        try:
            value = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise GuardUnmeasured("missing/unreadable candidate failure state") from exc
        if (not isinstance(value, dict) or value.get("schema") != "particle-sliders-candidate-failure-state-v1"
                or value.get("binding") != self.binding):
            raise GuardUnmeasured("candidate failure state cannot be rebound")
        if value.get("status") not in ("pending", "pass", "failed", "unmeasured"):
            raise GuardUnmeasured("invalid candidate guard state")
        return value

    def _settled(self):
        """Lock-free read unless a probe_committed row may belong to a live probe.

        The guard lock is the liveness proof: the kernel releases it when its
        holder dies. Wait for a live prober, then act on its final journal; a
        row still in flight once the lock is ours was interrupted and stays
        unmeasured. The probing thread itself cannot wait for its own probe.
        """
        value = self._read()
        if not _in_flight(value) or self._held_here():
            return value
        with self._lock():
            return self._read()

    def _admit(self, value):
        if value["status"] == "failed":
            raise CandidateFailure("candidate has a retained catastrophic-output failure")
        if value["status"] == "unmeasured":
            raise GuardUnmeasured("required catastrophic-output assessment remains unmeasured"
                                  + (" (probe interrupted, or called from within its own probe)"
                                     if _in_flight(value) else ""))
        error = self.cache.detector.qualification_error(self.binding["media"])
        if error:
            raise GuardUnmeasured(error)

    def require_update(self):
        self._admit(self._settled())

    require_resume = require_update

    def require_selection(self, checkpoint=None):
        value = self._settled()
        self._admit(value)
        assessed = value["checkpoints"].get(_digest(asdict(checkpoint))) if checkpoint is not None else None
        if (value["status"] != "pass" or checkpoint is not None
                and (assessed is None or assessed.get("status") != "pass")):
            raise GuardUnmeasured("no healthy assessment of the selected checkpoint")

    def resolve_unmeasured(self, *, reason, resolved_by):
        """Explicit, journaled exit from unmeasured; never called automatically.

        Moves probe_committed/unmeasured rows verbatim into ``resolutions`` so
        those checkpoints can be assessed again; the candidate returns to pass
        when earlier passing rows remain, else pending. A failure is never
        cleared, and retained ObservationCache firsts are not re-rated.
        """
        if (not isinstance(reason, str) or not reason.strip()
                or not isinstance(resolved_by, str) or not resolved_by.strip()):
            raise ValueError("nonempty resolution reason and resolved_by required")
        with self._lock():
            value = self._read()
            if value["status"] == "failed":
                raise CandidateFailure("candidate has a retained catastrophic-output failure; it cannot be resolved")
            if value["status"] != "unmeasured":
                raise ValueError("only an unmeasured candidate can be resolved, not " + value["status"])
            superseded = {key: row for key, row in value["checkpoints"].items()
                          if row.get("status") in ("probe_committed", "unmeasured")}
            for key in superseded:
                del value["checkpoints"][key]
            resolution = dict(reason=reason, resolved_by=resolved_by,
                              resolved_at=datetime.now(timezone.utc).isoformat(),
                              previous_status=value["status"], superseded_checkpoints=superseded)
            value.setdefault("resolutions", []).append(resolution)
            # Earlier passing assessments are unaffected by a later unknown
            # probe: return to the state held before it, never to a new pass.
            value["status"] = "pass" if any(row.get("status") == "pass" for row in value["checkpoints"].values()) else "pending"
            _write(self.path, value)
            return resolution

    def check_checkpoint(self, checkpoint, probe):
        with self._lock():
            return self._check_checkpoint(checkpoint, probe)

    def _check_checkpoint(self, checkpoint, probe):
        if not isinstance(checkpoint, FileRecord):
            raise TypeError("a committed full checkpoint identity is required before probing")
        self.require_update()
        value = self._read()
        key = _digest(asdict(checkpoint))
        if key in value["checkpoints"]:
            self.require_selection(checkpoint)
            return value["checkpoints"][key]
        # Commit before product rendering. A crash here cannot silently replay
        # unknown completed forwards or become an implicit healthy result.
        value.update(status="unmeasured", latest_checkpoint=asdict(checkpoint))
        value["checkpoints"][key] = dict(status="probe_committed", checkpoint=asdict(checkpoint))
        _write(self.path, value)
        observations = []
        try:
            error = self.cache.detector.qualification_error(self.binding["media"])
            if error:
                raise GuardUnmeasured(error)
            outputs = iter(probe())
            try:
                for output in outputs:
                    if not isinstance(output, FinishedOutput) or output.media != self.binding["media"]:
                        raise GuardUnmeasured("product did not supply genuine finished outputs")
                    observation = self.cache.assess(output)
                    observations.append(observation)
                    if observation["status"] != "pass":
                        break  # Do not request further outputs after fail/unknown.
            finally:
                if callable(getattr(outputs, "close", None)):
                    outputs.close()
            if not observations:
                raise GuardUnmeasured("product did not supply genuine finished outputs")
            status = "failed" if any(o["status"] == "fail" for o in observations) else (
                     "unmeasured" if any(o["status"] != "pass" for o in observations) else "pass")
            row = dict(status=status, checkpoint=asdict(checkpoint), observations=observations)
        except Exception as exc:
            row = dict(status="failed" if any(o["status"] == "fail" for o in observations) else "unmeasured",
                       checkpoint=asdict(checkpoint), observations=observations,
                       error=type(exc).__name__ + ": " + str(exc))
        value["checkpoints"][key] = row
        value["status"] = row["status"]
        _write(self.path, value)
        self.require_selection(checkpoint)
        return row


def _close_preserving(session, exc):
    try:
        session.close()
    except BaseException as cleanup:
        exc.add_note("session.close() also failed: %s: %s" % (type(cleanup).__name__, cleanup))


class GuardedSession:
    """Public wrapper preserving the original session/packet and update math.

    ``checkpoint(session)`` must commit a full training checkpoint and return
    FileRecord. ``probe(session, checkpoint)`` supplies product-owned finished
    outputs, preferably already saved. Neither core nor response vectors supply
    a generic renderer. Observation runs synchronously before the next update.
    """
    def __init__(self, session, guard, *, checkpoint, probe, every):
        if type(every) is not int or every < 1:
            raise ValueError("positive declared output-probe interval required")
        self.session, self.guard = session, guard
        self.save_checkpoint, self.probe, self.every = checkpoint, probe, every

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _boundary(self, *, final=False):
        step = self.session.game.completed_steps
        if final or step % self.every == 0:
            checkpoint = self.save_checkpoint(self.session)
            self.guard.check_checkpoint(checkpoint, lambda: self.probe(self.session, checkpoint))

    def step(self):
        self.guard.require_update()
        metrics = self.session.step()
        try:
            self._boundary()
        except BaseException as exc:
            _close_preserving(self.session, exc)
            raise
        return metrics

    def train(self, *, stop_after=None, on_step=None):
        self.guard.require_update()
        target = self.session.game.recipe.total_steps if stop_after is None else stop_after
        def callback(metrics):
            self.guard.require_update()
            self._boundary(final=self.session.game.completed_steps == target)
            if on_step is not None:
                on_step(metrics)
        return self.session.train(stop_after=stop_after, on_step=callback)

    def restore(self, checkpoint, *, load_packet=None):
        self.guard.require_update()
        if not isinstance(checkpoint, FileRecord) or not callable(load_packet):
            raise GuardUnmeasured("restore requires an exact assessed checkpoint and its product packet loader")
        self.guard.require_selection(checkpoint)
        checkpoint.verify()
        packet = load_packet(checkpoint.path)
        checkpoint.verify()
        return self.session.restore(packet)

    def __enter__(self):
        try:
            self.guard.require_resume()
        except BaseException as exc:
            # ``with`` never calls __exit__ when __enter__ raises.
            if callable(getattr(self.session, "close", None)):
                _close_preserving(self.session, exc)
            raise
        self.session.__enter__()
        return self

    def __exit__(self, *args):
        return self.session.__exit__(*args)


def open_guarded_session(open_session, guard, *, checkpoint, probe, every, resume_checkpoint=None):
    """Check persistent failure before a product opens/loads any native model."""
    if type(every) is not int or every < 1 or not callable(checkpoint) or not callable(probe):
        raise ValueError("valid product checkpoint/probe callbacks and positive interval required")
    guard.require_resume()
    if resume_checkpoint is not None:
        guard.require_selection(resume_checkpoint)
    return GuardedSession(open_session(), guard, checkpoint=checkpoint, probe=probe, every=every)
