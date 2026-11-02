"""Opt-in Ternary-Bonsai-2-27B GGUF readout backend (frozen, text-only).

Pinned model: ``prism-ml/Ternary-Bonsai-2-27B-gguf`` file
``Ternary-Bonsai-2-27B-PTQ1_0.gguf`` (5,946,648,928 bytes on disk) —
Apache-2.0, ungated, derived from ``Qwen/Qwen3.8-27B``. GGUF arch
``qwen35``: 64 blocks, hidden 5120, 24 query / 4 kv heads, 262144
context, vocab 248320. Verified 2026-09-17 against the Hub card plus a
range-fetched GGUF header (magic ``GGUF``, 851 tensors: 402x type 143
ternary, 353x F32 norms/state, 96x type 30; ``prism.hadamard.*`` keys
present). Discard the numbers above if the card disagrees.

Stock llama.cpp and transformers CANNOT load this file: PTQ1_0 (type
143) and PQ2_0 (type 142) sit past upstream's ``GGML_TYPE_COUNT`` so a
stock build refuses them outright, and the weights live in a rotated
basis that needs the fork's Hadamard activation runtime. The required
runtime is the ``PrismML-Eng/llama.cpp`` fork; the run source of truth
is ``PrismML-Eng/Bonsai-demo``. The dev-repo Q2_0 band is the dangerous
one (stock loads it without warning and outputs garbage) and is never
touched here. Text-only: neither mmproj vision pack is loaded, ever.

Readout path: the frozen model is served by the fork's ``llama-server``
with ``--embedding --pooling last``; this backend POSTs ``/embedding``
(and ``/tokenize`` for the context guard) over stdlib HTTP. There is no
gradient path through a GGUF — full LoRA-in-GGUF training is impossible,
so the trainable slider (``bonsai_gguf_particle.py``) is a torch-side
residual head on this readout, trained by the shared particle_bridge
game. ``--dummy`` is the CI / CPU path: deterministic seeded readouts
of the real hidden width, no weights, no binary, no server.

This module never touches Music 3, YuE2, Music Arm B, ``locked_shared``
or any live ``--lm_target`` default.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO

import torch

# Pinned Hub repo + file. PTQ1_0 is the smallest pack (dense trits,
# 1.75 bits/weight). PQ2_0 is NOT accepted here: pin one file so a
# smoke always means the same bytes.
REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
FILENAME = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
FILE_BYTES = 5946648928
FILE_URL = f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}"
BASE_MODEL = "Qwen/Qwen3.8-27B"
MODEL_LICENSE = "Apache-2.0"
# Verified from the GGUF header (see module docstring).
ARCH = "qwen35"
HIDDEN = 5120
LAYERS = 64
CONTEXT = 262144
VOCAB = 248320
N_TENSORS = 851
# Ternary packings live past upstream GGML_TYPE_COUNT: 142 = PQ2_0,
# 143 = PTQ1_0. A stock build refuses both as unknown types.
PTQ1_0_TYPE = 143
PQ2_0_TYPE = 142
TERNARY_TYPE_IDS = (PQ2_0_TYPE, PTQ1_0_TYPE)
# Bands this backend must never fetch or load.
F16_FILENAME = "Ternary-Bonsai-2-27B-F16.gguf"
F16_BYTES = 53808408928
PQ2_0_FILENAME = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
MMPROJ_Q8_FILENAME = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
MMPROJ_BF16_FILENAME = "Ternary-Bonsai-2-27B-mmproj-BF16.gguf"
Q2_0_DEV_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf-dev"
Q2_0_DEV_FILENAME = "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"
# Fork run source of truth + the release this VM smoke-ran.
FORK_URL = "https://github.com/PrismML-Eng/llama.cpp"
DEMO_URL = "https://github.com/PrismML-Eng/Bonsai-demo"
FORK_RELEASE_PIN = "prism-b10685-7dffb15"
FORK_MIN_BUILD = 10658  # prism-v7 rebase: official Q2_0 group-64 + PQ2_0
# Readout: last-token hidden via the fork server embedding endpoint.
EMBED_POOLING = "last"
SERVER_CTX_DEFAULT = 512  # smoke context; text prompts are tens of tokens

DUMMY_SEED = 0xB04A1


class BonsaiGGUFError(RuntimeError):
    """Fail-closed error for anything Bonsai-GGUF (missing fork, bad file)."""


class StockLlamaRejected(BonsaiGGUFError):
    """Raised when a stock (non-fork) llama.cpp build meets ternary weights."""


def _basename(path: str) -> str:
    return os.path.basename(str(path))


def resolve_weights(path: str | os.PathLike) -> str:
    """Accept ONLY the pinned PTQ1_0 file; reject F16 / mmproj / others.

    There is deliberately no override: a Bonsai smoke must always mean
    the same bytes. Raises :class:`BonsaiGGUFError` with a message that
    names the pinned file and why the candidate was refused.
    """
    name = _basename(path)
    if name == FILENAME:
        return str(path)
    if name == F16_FILENAME:
        raise BonsaiGGUFError(
            f"refusing {name}: the 53.8 GB F16 pack is never downloaded "
            f"by this backend; use the pinned {FILENAME} ({FILE_URL})"
        )
    if name in (MMPROJ_Q8_FILENAME, MMPROJ_BF16_FILENAME):
        raise BonsaiGGUFError(
            f"refusing {name}: text-only backend, the vision projector is "
            "never loaded; use the pinned language pack "
            f"{FILENAME}"
        )
    if name == PQ2_0_FILENAME:
        raise BonsaiGGUFError(
            f"refusing {name}: this backend pins {FILENAME} (smallest pack); "
            "PQ2_0 is a legitimate band on the card but is not accepted here "
            "so every smoke means the same bytes"
        )
    if name == Q2_0_DEV_FILENAME or "prism-fork-required" in name:
        raise BonsaiGGUFError(
            f"refusing {name}: the dev-repo Q2_0 band loads as garbage on "
            f"stock llama.cpp without warning; only {FILENAME} from {REPO} "
            "is accepted"
        )
    raise BonsaiGGUFError(
        f"refusing {name}: not the pinned file; expected {FILENAME} "
        f"({FILE_BYTES} bytes) from {REPO}"
    )


def check_weights_size(path: str | os.PathLike) -> int:
    """Stat the pinned file and reject wrong sizes (truncated download)."""
    size = os.path.getsize(path)
    if size != FILE_BYTES:
        raise BonsaiGGUFError(
            f"{_basename(path)} is {size} bytes, expected {FILE_BYTES}; "
            "re-download the pinned PTQ1_0 file (truncated copies fail closed)"
        )
    return size


# ---------------------------------------------------------------------------
# GGUF header probe (no full download needed for the metadata verdict)
# ---------------------------------------------------------------------------

def _read_str(buf: bytes, off: int) -> tuple[str, int]:
    (ln,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = buf[off:off + ln].decode("utf-8", errors="replace")
    return s, off + ln


_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
    5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


def _skip_value(buf: bytes, off: int, t: int) -> int:
    if t == 8:
        _, off = _read_str(buf, off)
        return off
    if t == 9:
        (arr_t,) = struct.unpack_from("<I", buf, off)
        off += 4
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        for _ in range(n):
            off = _skip_value(buf, off, arr_t)
        return off
    return off + struct.calcsize(_SCALAR_FMT[t])


def probe_gguf_head(data: bytes) -> dict[str, Any]:
    """Verdict a GGUF byte prefix (header + metadata + tensor infos).

    Needs enough leading bytes to cover the metadata KV store and the
    851 tensor infos (~11 MB for PTQ1_0: the tokenizer arrays are large).
    Raises :class:`BonsaiGGUFError` / :class:`StockLlamaRejected`.
    """
    if len(data) < 24 or data[:4] != b"GGUF":
        raise BonsaiGGUFError("not a GGUF file (bad magic); refusing to guess")
    (_, n_tensors, n_kv) = struct.unpack_from("<IQQ", data, 4)
    off = 24
    meta: dict[str, Any] = {}
    try:
        for _ in range(n_kv):
            key, off = _read_str(data, off)
            (t,) = struct.unpack_from("<I", data, off)
            off += 4
            if t == 8:
                val, off = _read_str(data, off)
                meta[key] = val
            elif t in _SCALAR_FMT:
                fmt = _SCALAR_FMT[t]
                (val,) = struct.unpack_from(fmt, data, off)
                off += struct.calcsize(fmt)
                meta[key] = val
            else:
                off = _skip_value(data, off, t)
                meta[key] = None
    except struct.error as exc:
        raise BonsaiGGUFError(
            "GGUF prefix too short to cover the metadata store; fetch more "
            f"leading bytes ({exc})"
        ) from exc
    arch = meta.get("general.architecture")
    if arch != ARCH:
        raise BonsaiGGUFError(
            f"GGUF arch is {arch!r}, expected {ARCH!r}; refusing"
        )
    hidden = meta.get(f"{ARCH}.embedding_length")
    layers = meta.get(f"{ARCH}.block_count")
    ctx = meta.get(f"{ARCH}.context_length")
    if hidden != HIDDEN or layers != LAYERS or ctx != CONTEXT:
        raise BonsaiGGUFError(
            f"GGUF shape arch={arch} hidden={hidden} layers={layers} ctx={ctx}; "
            f"expected hidden={HIDDEN} layers={LAYERS} ctx={CONTEXT}"
        )
    if not any(k.startswith("prism.hadamard") for k in meta):
        raise StockLlamaRejected(
            "no prism.hadamard.* metadata: this file is not a rotated-basis "
            "ternary pack, or a stock writer stripped the markers; the fork "
            f"runtime is required ({FORK_URL})"
        )
    tensor_types: set[int] = set()
    try:
        for _ in range(n_tensors):
            _, off = _read_str(data, off)
            (n_dims,) = struct.unpack_from("<I", data, off)
            off += 4 + 8 * n_dims
            (ty,) = struct.unpack_from("<i", data, off)
            off += 4
            off += 8  # tensor data offset
            tensor_types.add(ty)
    except struct.error as exc:
        raise BonsaiGGUFError(
            "GGUF prefix too short to cover the tensor infos; fetch more "
            f"leading bytes ({exc})"
        ) from exc
    if not (tensor_types & set(TERNARY_TYPE_IDS)):
        raise StockLlamaRejected(
            f"no ternary tensor types {TERNARY_TYPE_IDS} found "
            f"(saw {sorted(tensor_types)}); an F16/F32 file or a stock "
            "repack cannot serve as the ternary readout"
        )
    return {
        "arch": arch,
        "hidden": hidden,
        "layers": layers,
        "context": ctx,
        "n_tensors": n_tensors,
        "tensor_types": sorted(tensor_types),
        "has_ternary": True,
        "has_hadamard": True,
    }


def probe_gguf(path: str | os.PathLike) -> dict[str, Any]:
    """Probe a local GGUF file without loading weights (bounded prefix)."""
    resolve_weights(path)
    check_weights_size(path)
    with open(path, "rb") as fh:
        head = fh.read(12 * 1024 * 1024)
    return probe_gguf_head(head)


# ---------------------------------------------------------------------------
# Fork-binary guards
# ---------------------------------------------------------------------------

STOCK_FAILURE_MARKERS = (
    "unknown model type",
    "unknown tensor type",
    "unknown type",
    "unsupported model",
    "unsupported tensor",
    "unsupported type",
    "unsupported architecture",
    "invalid model",
    "error loading model",
    "failed to load model",
    "failed to load the model",
    "exceeds ggml_type_count",
    "ggml_type_count",
)
FORK_LOAD_MARKERS = (
    "model loaded",
    "load: model loaded",
    "llama_model_load_from_file",
)


def check_fork_log(log_text: str) -> None:
    """Fail-closed verdict on a llama-server load log.

    Raises :class:`StockLlamaRejected` on stock failure signatures,
    returns normally on positive fork-load evidence, and raises
    :class:`BonsaiGGUFError` on ambiguous logs (never silently OK).
    """
    low = str(log_text).lower()
    for marker in STOCK_FAILURE_MARKERS:
        if marker in low:
            raise StockLlamaRejected(
                f"llama load log shows {marker!r}: stock llama.cpp cannot run "
                f"ternary PTQ1_0; use the fork ({FORK_URL}, release "
                f"{FORK_RELEASE_PIN}); run source of truth: {DEMO_URL}"
            )
    for marker in FORK_LOAD_MARKERS:
        if marker in low:
            return
    raise BonsaiGGUFError(
        "could not confirm the fork loaded the model (no load marker in "
        "the log); refusing to treat the server as a ternary readout"
    )


def require_fork_binary(bin_path: str | os.PathLike) -> str:
    """Check the server binary exists and reports a usable version.

    Behavioral proof that it is the fork (not stock) comes from
    :func:`check_fork_log` at load time: only the fork can open PTQ1_0.
    """
    path = str(bin_path)
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        raise BonsaiGGUFError(
            f"fork llama-server not found at {path}; fetch the prebuilt "
            f"binary from {FORK_URL}/releases (tag {FORK_RELEASE_PIN}, "
            f"asset *-bin-ubuntu-x64.tar.gz on Linux CPU) or follow {DEMO_URL}"
        )
    try:
        proc = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BonsaiGGUFError(f"could not run {path} --version: {exc}") from exc
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "version" not in out.lower():
        raise BonsaiGGUFError(
            f"{path} --version failed; refusing to guess the runtime"
        )
    return out.strip()


def build_server_cmd(
    bin_path: str | os.PathLike,
    weights: str | os.PathLike,
    *,
    port: int,
    ctx: int = SERVER_CTX_DEFAULT,
    threads: int | None = None,
) -> list[str]:
    """Argv for the fork server as a frozen text-only embedding readout.

    Never passes ``--mmproj`` (text-only) and never touches chat/tool
    flags: this server exists to emit last-token hidden states.
    """
    require_fork_binary(bin_path)
    resolve_weights(weights)
    cmd = [
        str(bin_path), "-m", str(weights),
        "--host", "127.0.0.1", "--port", str(int(port)),
        "--embedding", "--pooling", EMBED_POOLING,
        "-c", str(int(ctx)), "-ngl", "0",
    ]
    if threads is not None:
        cmd += ["-t", str(int(threads))]
    return cmd


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only: no new dependencies for an opt-in backend)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except OSError as exc:
        raise BonsaiGGUFError(
            f"fork server request to {url} failed ({exc}); is llama-server "
            "from the fork running with --embedding --pooling last?"
        ) from exc


def _get_json(url: str, timeout: float = 10.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except OSError as exc:
        raise BonsaiGGUFError(f"fork server GET {url} failed ({exc})") from exc


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class EncodedText:
    ids: list[int]
    text: str


class _HashTokenizer:
    """Deterministic word-hash ids for the dummy path (not a real tokenizer)."""

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in str(text).split():
            digest = hashlib.sha256(word.lower().encode("utf-8")).digest()
            ids.append(int.from_bytes(digest[:4], "little") % (VOCAB - 1) + 1)
        return ids or [1]


class BonsaiGGUFBackend:
    """Frozen ternary-GGUF readout host.

    Live: ``server_url`` must point at the fork's ``llama-server``
    started with ``--embedding --pooling last`` on the pinned PTQ1_0
    file. Dummy: seeded torch readouts of width 5120, no Hub, no
    binary, no server — the CI / CPU path.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        weights: str | os.PathLike | None = None,
        server_url: str | None = None,
        dummy: bool = False,
    ) -> None:
        self.device = torch.device(device if not dummy else "cpu")
        self.dummy = bool(dummy)
        self.server_url = (server_url or "").rstrip("/") or None
        self.weights = str(weights) if weights is not None else None
        if self.dummy:
            self.tokenizer = _HashTokenizer()
        else:
            if self.weights is None or self.server_url is None:
                raise BonsaiGGUFError(
                    "live Bonsai readout needs --weights (pinned PTQ1_0) and "
                    "--server_url (fork llama-server with --embedding "
                    "--pooling last); or pass --dummy for the CPU stand-in"
                )
            resolve_weights(self.weights)
            self._check_server()

    def _check_server(self) -> None:
        assert self.server_url is not None
        try:
            health = _get_json(self.server_url + "/health")
        except BonsaiGGUFError as exc:
            raise BonsaiGGUFError(
                f"fork server at {self.server_url} is not answering: {exc}"
            ) from exc
        if str(health.get("status", "")).lower() not in ("ok", "loading model"):
            raise BonsaiGGUFError(
                f"fork server at {self.server_url} reports {health!r}"
            )

    def encode(self, text: str) -> EncodedText:
        if self.dummy:
            ids = self.tokenizer.encode(text)
        else:
            assert self.server_url is not None
            out = _post_json(
                self.server_url + "/tokenize", {"content": str(text)}
            )
            ids = [int(x) for x in out.get("tokens", [])]
            ids = ids or [0]
        if len(ids) > CONTEXT:
            raise ValueError(f"prompt exceeds context ({len(ids)} > {CONTEXT})")
        return EncodedText(ids=ids, text=str(text))

    def _dummy_readout(self, text: str) -> torch.Tensor:
        seed = (DUMMY_SEED ^ int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
        )) % (2 ** 63)
        gen = torch.Generator().manual_seed(seed)
        vec = torch.randn(HIDDEN, generator=gen)
        return vec.unsqueeze(0).to(self.device)

    def _live_readout(self, text: str) -> torch.Tensor:
        assert self.server_url is not None
        out = _post_json(
            self.server_url + "/embedding", {"content": str(text)},
            timeout=300.0,
        )
        # Fork server shape: [{"index": 0, "embedding": [[...]]}] — one
        # entry per input, with one vector per pooled sequence.
        if isinstance(out, list):
            if len(out) != 1 or not isinstance(out[0], dict):
                raise BonsaiGGUFError(
                    f"server embedding batch has {len(out)} entries, expected 1"
                )
            vec = out[0].get("embedding")
        elif isinstance(out, dict):
            vec = out.get("embedding")
        else:
            raise BonsaiGGUFError(
                f"server embedding has unexpected shape {type(out)}"
            )
        if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
            vec = vec[0]
        if not isinstance(vec, list) or len(vec) != HIDDEN:
            raise BonsaiGGUFError(
                f"server embedding has dim {len(vec) if isinstance(vec, list) else type(vec)}"
                f", expected {HIDDEN}; the readout is not the Bonsai hidden state"
            )
        ten = torch.tensor(vec, dtype=torch.float32, device=self.device)
        if not torch.isfinite(ten).all():
            raise BonsaiGGUFError("non-finite server embedding; refusing")
        return ten.unsqueeze(0)

    @torch.no_grad()
    def teacher_hidden(self, enc: EncodedText) -> torch.Tensor:
        """Frozen base readout (no head attached at rest)."""
        return self.hidden(enc)

    def hidden(self, enc: EncodedText) -> torch.Tensor:
        """Last-token hidden readout ``[1, 5120]``; head scale comes from
        the network's ``scaled()`` context, never from this backend."""
        if self.dummy:
            return self._dummy_readout(enc.text)
        return self._live_readout(enc.text)


def download_weights(
    dest_dir: str | os.PathLike, *, allow_download: bool = False
) -> str:
    """Fetch ONLY the pinned PTQ1_0 file. There is no F16 code path.

    Refuses unless ``allow_download`` is set; verifies the exact byte
    size afterwards (truncated copies fail closed).
    """
    if not allow_download:
        raise BonsaiGGUFError(
            f"refusing to download without --allow_download; pinned file is "
            f"{FILENAME} ({FILE_BYTES} bytes): {FILE_URL}"
        )
    dest = os.path.join(str(dest_dir), FILENAME)
    os.makedirs(str(dest_dir), exist_ok=True)
    req = urllib.request.Request(FILE_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except OSError as exc:
        raise BonsaiGGUFError(f"PTQ1_0 download failed: {exc}") from exc
    check_weights_size(dest)
    return dest
