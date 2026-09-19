#!/usr/bin/env python3
"""Live mixer-desk dashboard for slider training logs.

Tails `*_train.jsonl` (LM / transformer / encoder) as they grow, plus any
matching trainer processes on the GPUs. Stdlib only.

    python scripts/watch_train.py
    python scripts/watch_train.py models/dust-lm-v1-sym-kl-beta05
    python scripts/watch_train.py a.jsonl b.jsonl --once
    python scripts/watch_train.py --recent 8

Keys (live TTY):  q quit  j/k cycle  a all  space pause  r rescan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve()
CONCEPTMOD = HERE.parent.parent
DEFAULT_ROOT = CONCEPTMOD / "models"

TRAIN_SCRIPTS = {
    "train_lm_slider_music3.py": "LM",
    "train_lora_music3.py": "TF",
    "train_encoder_music3.py": "ENC",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
YAML_LABEL_RE = re.compile(r"^(plus_label|minus_label):\s*[\"']?(.+?)[\"']?\s*$")
TQDM_LM_RE = re.compile(
    r"loss\s+([0-9.]+)\s+p%([0-9.]+)\s+n%([0-9.]+)\s+"
    r"c\+\s+(-?[0-9.]+)\s+c-\s+(-?[0-9.]+)\s+col\s+(-?[0-9.]+)"
    r"(?:\s+e±\s+([0-9.]+)/([0-9.]+))?"
)
BLOCKS = "▁▂▃▄▅▆▇█"
BAR_FILL = "█"
BAR_EMPTY = "░"

# 256-color studio palette
AMBER = 214
CYAN = 81
MINT = 85
GREEN = 82
MAGENTA = 213
PINK = 211
RED = 203
GOLD = 220
YELLOW = 227
ORANGE = 208
DIM = 240
MUTED = 245
WHITE = 255
ICE = 159
PURPLE = 177

COLOR = True


# ---------------------------------------------------------------------------
# terminal paint
# ---------------------------------------------------------------------------


def _isatty() -> bool:
    return bool(sys.stdout.isatty())


def enable_color(force: bool | None = None) -> None:
    global COLOR
    if force is not None:
        COLOR = force
        return
    if os.environ.get("NO_COLOR"):
        COLOR = False
        return
    COLOR = _isatty() or os.environ.get("FORCE_COLOR") == "1"


def paint(text: str, fg: int | None = None, *, bg: int | None = None, bold: bool = False, dim: bool = False) -> str:
    if not COLOR or text == "":
        return text
    codes: list[str] = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if fg is not None:
        codes.extend(["38", "5", str(fg)])
    if bg is not None:
        codes.extend(["48", "5", str(bg)])
    if not codes:
        return text
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"


def vislen(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def clip(s: str, n: int) -> str:
    if n <= 0:
        return ""
    if vislen(s) <= n:
        return s
    if n <= 1:
        return "…"
    out: list[str] = []
    width = 0
    i = 0
    while i < len(s) and width < n - 1:
        if s[i] == "\x1b":
            m = ANSI_RE.match(s, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        out.append(s[i])
        width += 1
        i += 1
    out.append("\x1b[0m…" if COLOR else "…")
    return "".join(out)


def pad(s: str, n: int, align: str = "left") -> str:
    s = clip(s, n)
    extra = n - vislen(s)
    if extra <= 0:
        return s
    if align == "right":
        return " " * extra + s
    if align == "center":
        a = extra // 2
        return " " * a + s + " " * (extra - a)
    return s + " " * extra


def term_size(width: int | None = None, height: int | None = None) -> tuple[int, int]:
    fallback = (120, 42)
    tw, th = shutil.get_terminal_size(fallback=fallback)
    if not _isatty():
        tw, th = fallback
    if width:
        tw = width
    if height:
        th = height
    return max(72, tw), max(20, th)


def lerp_good(t: float) -> int:
    """t=0 bad (red) → t=1 good (green)."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    if t < 0.5:
        return RED if t < 0.25 else ORANGE
    if t < 0.8:
        return GOLD
    return GREEN


# ---------------------------------------------------------------------------
# sparks / bars
# ---------------------------------------------------------------------------


def _downsample(xs: list[float], width: int) -> list[float]:
    if width <= 0:
        return []
    if not xs:
        return []
    if len(xs) <= width:
        return xs
    out = []
    last = len(xs) - 1
    for i in range(width):
        out.append(xs[int(round(i * last / (width - 1)))])
    return out


def spark(xs: list[float] | None, width: int, lo: float | None = None, hi: float | None = None) -> str:
    if not xs or width <= 0:
        return " " * max(0, width)
    data = _downsample([float(x) for x in xs if x is not None], width)
    if not data:
        return " " * width
    pad_left = width - len(data)
    if lo is None:
        lo = min(data)
    if hi is None:
        hi = max(data)
    if hi <= lo:
        hi = lo + 1e-9
    chars = []
    for x in data:
        t = (x - lo) / (hi - lo)
        t = 0.0 if t < 0 else 1.0 if t > 1 else t
        chars.append(BLOCKS[int(t * (len(BLOCKS) - 1) + 1e-9)])
    return (" " * pad_left) + "".join(chars)


def hbar(value: float, lo: float, hi: float, width: int, invert: bool = False) -> tuple[str, float]:
    if width <= 0:
        return "", 0.0
    span = hi - lo
    t = 0.5 if span == 0 else (value - lo) / span
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    good = 1.0 - t if invert else t
    filled = int(round(good * width))
    body = BAR_FILL * filled + BAR_EMPTY * (width - filled)
    return body, good


def fmt(x: float | None, digits: int = 3, signed: bool = False) -> str:
    if x is None:
        return "—"
    if abs(x) >= 100:
        return f"{x:.0f}"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(x, spec)


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def fmt_dur(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def shorten_names(names: list[str], n: int) -> list[str]:
    if not names:
        return []
    if max(len(nm) for nm in names) <= n:
        return list(names)
    pfx = names[0]
    for nm in names[1:]:
        i = 0
        while i < len(pfx) and i < len(nm) and pfx[i] == nm[i]:
            i += 1
        pfx = pfx[:i]
    cut = pfx.rfind("-")
    if cut >= 3:
        pfx = pfx[: cut + 1]
    out: list[str] = []
    for nm in names:
        tail = nm[len(pfx) :] if pfx and nm.startswith(pfx) else nm
        label = f"…{tail}" if pfx and tail and tail != nm else nm
        if len(label) > n:
            label = "…" + label[-(n - 1) :]
        out.append(label)
    return out


def mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# file / process discovery
# ---------------------------------------------------------------------------


def _read_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def _read_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def _read_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, _, v = item.partition(b"=")
        out[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return out


def _proc_start(pid: int) -> float | None:
    try:
        return Path(f"/proc/{pid}").stat().st_ctime
    except OSError:
        return None


def parse_argv(tokens: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:]
            if key.startswith("no-"):
                out[key[3:].replace("-", "_")] = False
                i += 1
                continue
            if "=" in key:
                k, _, v = key.partition("=")
                out[k.replace("-", "_")] = v
                i += 1
                continue
            key = key.replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                out[key] = tokens[i + 1]
                i += 2
                continue
            out[key] = True
            i += 1
            continue
        i += 1
    return out


def _script_kind(tokens: list[str]) -> str | None:
    for tok in tokens:
        base = Path(tok).name
        if base in TRAIN_SCRIPTS:
            return TRAIN_SCRIPTS[base]
    return None


@dataclass
class Proc:
    pid: int
    kind: str
    cwd: Path
    args: dict[str, Any]
    name: str
    save_dir: Path
    start: float | None
    cuda_visible: str | None
    jsonl: Path | None = None
    gpu_index: int | None = None
    gpu_mem_mib: float | None = None


def scan_procs() -> list[Proc]:
    found: list[Proc] = []
    try:
        pids = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        return found
    for pid in pids:
        tokens = _read_cmdline(pid)
        if not tokens:
            continue
        kind = _script_kind(tokens)
        if kind is None:
            continue
        cwd = _read_cwd(pid) or Path.cwd()
        args = parse_argv(tokens)
        name = str(args.get("name") or "")
        save = args.get("save_dir")
        save_dir = Path(str(save)) if save else cwd / "models" / (name or "run")
        if not save_dir.is_absolute():
            save_dir = (cwd / save_dir).resolve()
        env = _read_environ(pid)
        proc = Proc(
            pid=pid,
            kind=kind,
            cwd=cwd,
            args=args,
            name=name or save_dir.name,
            save_dir=save_dir,
            start=_proc_start(pid),
            cuda_visible=env.get("CUDA_VISIBLE_DEVICES"),
        )
        jsonls = sorted(save_dir.glob("*_train.jsonl")) if save_dir.is_dir() else []
        if kind == "ENC":
            alt = save_dir / "metrics.jsonl"
            if alt.exists():
                jsonls.append(alt)
        if jsonls:
            # newest / matching name first
            named = [p for p in jsonls if name and name in p.name]
            proc.jsonl = (named or jsonls)[-1]
        found.append(proc)
    return found


@dataclass
class Gpu:
    index: int
    uuid: str
    util: float
    mem_used: float
    mem_total: float
    temp: float
    power: float | None = None


def scan_gpus() -> tuple[list[Gpu], dict[int, tuple[int, float]]]:
    """Return GPUs and pid -> (gpu_index, mem_mib)."""
    gpus: list[Gpu] = []
    pid_map: dict[int, tuple[int, float]] = {}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return gpus, pid_map
    import subprocess

    def _csv(query: str) -> list[list[str]]:
        try:
            raw = subprocess.check_output(
                [smi, f"--query-{query}", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        rows = []
        for line in raw.splitlines():
            if line.strip():
                rows.append([c.strip() for c in line.split(",")])
        return rows

    uuid_to_index: dict[str, int] = {}
    for cols in _csv("gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"):
        if len(cols) < 6:
            continue
        try:
            gpu = Gpu(
                index=int(cols[0]),
                uuid=cols[1],
                util=_num(cols[2]),
                mem_used=_num(cols[3]),
                mem_total=_num(cols[4]),
                temp=_num(cols[5]),
                power=_num(cols[6]) if len(cols) > 6 else None,
            )
        except ValueError:
            continue
        gpus.append(gpu)
        uuid_to_index[gpu.uuid] = gpu.index
    for cols in _csv("compute-apps=pid,gpu_uuid,used_gpu_memory"):
        if len(cols) < 3:
            continue
        try:
            pid = int(cols[0])
            mem = _num(cols[2])
        except ValueError:
            continue
        idx = uuid_to_index.get(cols[1])
        if idx is not None:
            pid_map[pid] = (idx, mem)
    return gpus, pid_map


def _num(s: str) -> float:
    s = s.replace("%", "").replace("MiB", "").replace("W", "").strip()
    if s in {"[N/A]", "N/A", ""}:
        return float("nan")
    return float(s)


# ---------------------------------------------------------------------------
# jsonl tail
# ---------------------------------------------------------------------------


class JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.inode: int | None = None
        self.buf = ""
        self.rows: list[dict[str, Any]] = []
        self.mtime = 0.0

    def poll(self) -> int:
        try:
            st = self.path.stat()
        except OSError:
            return 0
        self.mtime = st.st_mtime
        inode = st.st_ino
        if self.inode is not None and (inode != self.inode or st.st_size < self.offset):
            self.offset = 0
            self.buf = ""
            self.rows = []
        self.inode = inode
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
                self.offset = fh.tell()
        except OSError:
            return 0
        if not chunk:
            return 0
        self.buf += chunk
        parts = self.buf.split("\n")
        self.buf = parts[-1]
        added = 0
        for line in parts[:-1]:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                self.rows.append(row)
                added += 1
        return added


def yaml_labels(path: Path | None) -> tuple[str, str]:
    if path is None or not path.is_file():
        return "", ""
    plus = minus = ""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:80]:
            m = YAML_LABEL_RE.match(line)
            if not m:
                continue
            if m.group(1) == "plus_label":
                plus = m.group(2).strip()
            else:
                minus = m.group(2).strip()
    except OSError:
        return "", ""
    return plus, minus


def load_sidecar(jsonl: Path) -> dict[str, Any]:
    # LM: {name}_train.jsonl → {name}_last.json
    # TF: {stem}_train.jsonl → {stem}_last.json (sometimes)
    name = jsonl.name
    if name.endswith("_train.jsonl"):
        cand = jsonl.with_name(name[: -len("_train.jsonl")] + "_last.json")
    else:
        cand = jsonl.with_name("last.json")
    for path in (cand, jsonl.parent / f"{jsonl.parent.name}_last.json"):
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


# ---------------------------------------------------------------------------
# run model
# ---------------------------------------------------------------------------


def kind_from_row(row: dict[str, Any] | None, name: str) -> str:
    if not row:
        if "-lm-" in name or name.endswith("-lm") or "_lm_" in name:
            return "LM"
        return "TF"
    if "pperc" in row and ("cos_pos" in row or "cos" in row):
        if "cos_pos" in row:
            return "LM"
        return "ENC"
    if "cos" in row or "eval" in row:
        return "TF"
    if "pperc" in row:
        return "ENC"
    return "LM"


@dataclass
class Run:
    key: str
    jsonl: Path
    name: str
    kind: str = "LM"
    tail: JsonlTail = field(init=False)
    proc: Proc | None = None
    sidecar: dict[str, Any] = field(default_factory=dict)
    plus: str = ""
    minus: str = ""
    budget: int | None = None
    recipe: str = ""
    live: bool = False
    samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=24))
    first_attach_step: int | None = None

    def __post_init__(self) -> None:
        self.tail = JsonlTail(self.jsonl)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.tail.rows

    @property
    def last(self) -> dict[str, Any] | None:
        return self.rows[-1] if self.rows else None

    @property
    def step(self) -> int:
        row = self.last
        if not row:
            return 0
        return int(row.get("step") or 0)

    def poll(self) -> None:
        added = self.tail.poll()
        now = time.time()
        if added and self.rows:
            step = self.step
            if self.first_attach_step is None:
                self.first_attach_step = step
            self.samples.append((now, step))
        if self.rows and not self.kind:
            self.kind = kind_from_row(self.rows[0], self.name)

    def rate(self) -> float | None:
        if len(self.samples) >= 2:
            t0, s0 = self.samples[0]
            t1, s1 = self.samples[-1]
            if t1 > t0 and s1 > s0:
                return (s1 - s0) / (t1 - t0)
        if self.proc and self.proc.start and self.step > 0:
            dt = time.time() - self.proc.start
            if dt > 1:
                return self.step / dt
        return None

    def eta(self) -> float | None:
        rate = self.rate()
        if not rate or not self.budget:
            return None
        remain = self.budget - self.step
        if remain < 0:
            return 0.0
        return remain / rate

    def series(self, *keys: str) -> list[float]:
        out: list[float] = []
        for row in self.rows:
            for key in keys:
                if key in row and isinstance(row[key], (int, float)):
                    out.append(float(row[key]))
                    break
        return out

    def eval_series(self, key: str) -> list[tuple[int, float]]:
        pts = []
        for row in self.rows:
            ev = row.get("eval")
            if isinstance(ev, dict) and isinstance(ev.get(key), (int, float)):
                pts.append((int(row.get("step") or 0), float(ev[key])))
        return pts

    def last_eval(self) -> dict[str, Any] | None:
        for row in reversed(self.rows):
            ev = row.get("eval")
            if isinstance(ev, dict):
                return ev
        return None


def metric(row: dict[str, Any] | None, *keys: str) -> float | None:
    if not row:
        return None
    for key in keys:
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def pair_mean(row: dict[str, Any] | None, a: str, b: str) -> float | None:
    va = metric(row, a)
    vb = metric(row, b)
    if va is None and vb is None:
        return None
    if va is None:
        return vb
    if vb is None:
        return va
    return 0.5 * (va + vb)


def health_of(run: Run) -> tuple[int, list[str], float]:
    """Return (0-100, notes, goodness 0-1)."""
    row = run.last
    notes: list[str] = []
    if not row:
        return 0, ["waiting for first step"], 0.0
    step = run.step
    frac = (step / run.budget) if run.budget else 0.0
    cp = metric(row, "cos_pos", "cos")
    cn = metric(row, "cos_neg")
    cos = pair_mean(row, "cos_pos", "cos_neg")
    if cos is None:
        cos = cp
    col = metric(row, "collapse")
    perc = pair_mean(row, "pperc", "nperc")
    edrift = pair_mean(row, "edrift_p", "edrift_n")
    ev = run.last_eval()
    if ev and run.kind == "TF":
        if isinstance(ev.get("cos"), (int, float)):
            cos = float(ev["cos"])
        if isinstance(ev.get("collapse"), (int, float)):
            col = float(ev["collapse"])

    parts: list[tuple[float, float]] = []  # (weight, score 0-1)
    if cos is not None:
        parts.append((0.4, max(0.0, min(1.0, (cos + 0.1) / 1.1))))
        if cos >= 0.95:
            notes.append("cosine locked")
        elif cos >= 0.8:
            notes.append("cosine climbing")
        elif frac > 0.2:
            notes.append("cosine lagging")
    if col is not None:
        parts.append((0.4, max(0.0, min(1.0, (1.0 - col) / 2.0))))
        if col > 0.2:
            notes.append("poles aligned — collapsing")
        elif col > -0.5 and frac > 0.15:
            notes.append("collapse lagging")
        elif col <= -0.9:
            notes.append("poles opposite")
        elif col <= -0.7:
            notes.append("axis opening")
    if perc is not None:
        parts.append((0.2, max(0.0, min(1.0, 1.0 - perc))))
        if perc < 0.22:
            notes.append("residual low")
        elif perc > 0.5 and frac > 0.4:
            notes.append("residual high")
    if edrift is not None and edrift > 0.35:
        notes.append("end-margin drifting")
    if not parts:
        return 0, ["no metrics yet"], 0.0
    wsum = sum(w for w, _ in parts)
    score = sum(w * s for w, s in parts) / wsum
    return int(round(100 * score)), notes[:3], score


def recipe_of(proc: Proc | None, sidecar: dict[str, Any], jsonl: Path) -> str:
    src = dict(sidecar)
    if proc:
        src = {**src, **proc.args}
    bits: list[str] = []
    rank = src.get("rank")
    alpha = src.get("alpha")
    lr = src.get("lr")
    seed = src.get("seed")
    if rank is not None:
        bits.append(f"r{rank}")
    if alpha is not None:
        bits.append(f"α{alpha}")
    if lr is not None:
        bits.append(f"lr {lr}")
    if seed is not None:
        bits.append(f"seed {seed}")
    plan_w = src.get("planreg_weight")
    plan_m = src.get("planreg_mode")
    if plan_w not in (None, "", "0", 0, 0.0, False):
        mode = plan_m or src.get("planreg_mode") or "planreg"
        bits.append(f"plan {mode}×{plan_w}")
    pole = src.get("pole_mode")
    if pole and str(pole) not in {"mse", "None"}:
        bits.append(f"pole {pole}")
    beta = src.get("common_beta")
    if beta not in (None, "", "0", 0, 0.0, False):
        bits.append(f"β={beta}")
    tmode = src.get("target_mode") or ("symmetric" if src.get("symmetric") else None)
    if tmode:
        bits.append(str(tmode))
    xt = src.get("xt_mode")
    if xt:
        bits.append(f"xt {xt}")
    return "  ·  ".join(str(b) for b in bits)


def resolve_budget(proc: Proc | None, sidecar: dict[str, Any]) -> int | None:
    for src in (proc.args if proc else {}, sidecar):
        for key in ("steps", "steps_budget"):
            val = src.get(key)
            if val is None or val is False:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def discover(root: Path, explicit: list[Path], live_window: float) -> tuple[list[Run], list[Gpu]]:
    procs = scan_procs()
    gpus, pid_map = scan_gpus()
    for proc in procs:
        if proc.pid in pid_map:
            proc.gpu_index, proc.gpu_mem_mib = pid_map[proc.pid]
        if proc.jsonl is None and proc.save_dir.is_dir():
            hits = list(proc.save_dir.glob("*_train.jsonl"))
            if hits:
                proc.jsonl = max(hits, key=lambda p: p.stat().st_mtime)

    paths: dict[str, Path] = {}
    proc_by_path: dict[str, Proc] = {}

    def _add(path: Path, proc: Proc | None = None) -> None:
        try:
            path = path.resolve()
        except OSError:
            return
        if path.is_dir():
            hits = list(path.glob("*_train.jsonl")) + list(path.glob("metrics.jsonl"))
            for hit in hits:
                _add(hit, proc)
            return
        if path.suffix == ".log":
            # sibling jsonl next to a stdout capture
            stem = path.name.replace("-train.log", "").replace(".train.log", "")
            sibling = path.with_name(stem) / f"{stem}_train.jsonl"
            if sibling.exists():
                _add(sibling, proc)
            return
        if path.suffix != ".jsonl":
            return
        key = str(path)
        paths[key] = path
        if proc is not None:
            proc_by_path[key] = proc

    now = time.time()
    if explicit:
        for item in explicit:
            _add(item)
        for proc in procs:
            if proc.jsonl is None:
                continue
            try:
                key = str(proc.jsonl.resolve())
            except OSError:
                continue
            if key in paths:
                proc_by_path[key] = proc
    else:
        for proc in procs:
            if proc.jsonl is not None:
                _add(proc.jsonl, proc)
            _add(proc.save_dir, proc)
        if root.is_dir():
            newest: list[tuple[float, Path]] = []
            for path in root.rglob("*_train.jsonl"):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                newest.append((mtime, path))
                if now - mtime <= live_window:
                    _add(path)
            if not paths and newest:
                newest.sort(reverse=True)
                for _, path in newest[:3]:
                    _add(path)

    runs: list[Run] = []
    for key, path in paths.items():
        proc = proc_by_path.get(key)
        name = (proc.name if proc and proc.name else path.parent.name)
        sidecar = load_sidecar(path)
        plus = str(sidecar.get("plus_label") or "")
        minus = str(sidecar.get("minus_label") or "")
        prompts = None
        if proc and proc.args.get("prompts_file"):
            pf = Path(str(proc.args["prompts_file"]))
            if not pf.is_absolute():
                pf = (proc.cwd / pf)
            prompts = pf
        elif sidecar.get("prompts_file"):
            pf = Path(str(sidecar["prompts_file"]))
            if not pf.is_absolute():
                pf = CONCEPTMOD / pf
            prompts = pf
        if not plus or not minus:
            yp, ym = yaml_labels(prompts)
            plus = plus or yp
            minus = minus or ym
        kind = (proc.kind if proc else None) or sidecar.get("kind")
        if kind == "language_model":
            kind = "LM"
        if kind == "transformer":
            kind = "TF"
        if not kind:
            kind = kind_from_row(None, name)
        run = Run(
            key=key,
            jsonl=path,
            name=name,
            kind=str(kind),
            proc=proc,
            sidecar=sidecar,
            plus=plus,
            minus=minus,
            budget=resolve_budget(proc, sidecar),
            recipe=recipe_of(proc, sidecar, path),
            live=bool(proc) or (now - path.stat().st_mtime <= live_window if path.exists() else False),
        )
        try:
            run.poll()
        except OSError:
            pass
        if run.rows:
            run.kind = kind_from_row(run.rows[0], run.name) or run.kind
            if run.budget is None and run.kind == "LM":
                run.budget = 800
        runs.append(run)

    def sort_key(run: Run) -> tuple:
        return (0 if run.live else 1, -(run.tail.mtime or 0), run.name)

    runs.sort(key=sort_key)
    return runs, gpus


def recent_finished(root: Path, limit: int, exclude: set[str]) -> list[dict[str, Any]]:
    if not root.is_dir() or limit <= 0:
        return []
    files = []
    for path in root.rglob("*_train.jsonl"):
        key = str(path.resolve())
        if key in exclude:
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        files.append((st.st_mtime, path))
    files.sort(reverse=True)
    out = []
    for mtime, path in files[: limit * 3]:
        sidecar = load_sidecar(path)
        last = sidecar.get("last") if isinstance(sidecar.get("last"), dict) else None
        if last is None:
            try:
                with path.open("rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    fh.seek(max(0, size - 4096))
                    chunk = fh.read().decode("utf-8", "replace")
                line = [ln for ln in chunk.splitlines() if ln.strip()]
                last = json.loads(line[-1]) if line else None
            except (OSError, json.JSONDecodeError, IndexError):
                last = None
        if not isinstance(last, dict):
            continue
        out.append(
            {
                "name": sidecar.get("name") or path.parent.name,
                "mtime": mtime,
                "step": last.get("step"),
                "budget": sidecar.get("steps") or sidecar.get("steps_budget"),
                "cos": metric(last, "cos_pos", "cos"),
                "cos_neg": metric(last, "cos_neg"),
                "collapse": metric(last, "collapse"),
                "pperc": metric(last, "pperc"),
                "loss": metric(last, "loss"),
            }
        )
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def header_line(width: int, n_live: int, gpus: list[Gpu], paused: bool, clock: str) -> str:
    brand = paint(" LATHE ", fg=0, bg=AMBER, bold=True) + paint(" TRAIN WATCH ", fg=AMBER, bold=True)
    live = paint(f" {n_live} LIVE ", fg=0, bg=GREEN if n_live else DIM, bold=True)
    if paused:
        live += " " + paint(" PAUSED ", fg=0, bg=ORANGE, bold=True)
    bits = [brand, live, paint(clock, fg=ICE)]
    for gpu in gpus:
        util_c = lerp_good(1.0 - (gpu.util / 100.0) * 0.3) if gpu.util == gpu.util else MUTED
        temp_c = RED if gpu.temp >= 90 else ORANGE if gpu.temp >= 84 else MINT
        mem = f"{gpu.mem_used/1024:.1f}/{gpu.mem_total/1024:.0f}G" if gpu.mem_total == gpu.mem_total else ""
        pwr = f" {gpu.power:.0f}W" if gpu.power == gpu.power else ""
        bits.append(
            paint(f" GPU{gpu.index} ", fg=0, bg=CYAN, bold=True)
            + paint(f" {gpu.util:.0f}% {mem} ", fg=util_c)
            + paint(f"{gpu.temp:.0f}°{pwr}", fg=temp_c)
        )
    line = "  ".join(bits)
    return pad(line, width)


def _box_top(width: int, title: str, tag: str) -> str:
    inner = width - 2
    # ┌─ title ──── tag ─┐  (title/tag keep their own color)
    prefix = paint("─ ", fg=DIM)
    midpad = paint(" ", fg=DIM)
    suffix = paint(" ─", fg=DIM)
    tag = clip(tag, max(12, inner // 2))
    title = clip(title, max(4, inner - vislen(tag) - 8))
    fill = inner - (vislen(prefix) + vislen(title) + vislen(midpad) + vislen(tag) + vislen(suffix))
    return (
        paint("┌", fg=DIM)
        + prefix
        + title
        + midpad
        + paint("─" * max(0, fill), fg=DIM)
        + tag
        + suffix
        + paint("┐", fg=DIM)
    )


def _box_bot(width: int) -> str:
    return paint("└" + "─" * (width - 2) + "┘", fg=DIM)


def _box_row(width: int, body: str) -> str:
    inner = width - 4
    return paint("│ ", fg=DIM) + pad(body, inner) + paint(" │", fg=DIM)


def pulse(live: bool) -> str:
    if not live:
        return paint(" DONE ", fg=0, bg=DIM, bold=True)
    on = (time.time() % 1.0) < 0.55
    if on:
        return paint(" ● LIVE ", fg=0, bg=GREEN, bold=True)
    return paint(" ● LIVE ", fg=GREEN, bold=True)


def fader(progress: float, width: int) -> str:
    progress = 0.0 if progress < 0 else 1.0 if progress > 1 else progress
    filled = int(round(progress * max(0, width)))
    return paint(BAR_FILL * filled, fg=CYAN) + paint(BAR_EMPTY * (width - filled), fg=DIM)


def metric_row(
    width: int,
    label: str,
    value: str,
    bar: str,
    spark_s: str,
    note: str = "",
    value_fg: int = WHITE,
) -> str:
    # label 8 | value 10 | bar | spark | note
    lab = paint(f"{label:<9}", fg=MUTED)
    val = paint(pad(value, 10, "right"), fg=value_fg, bold=True)
    rest = "  " + bar + "  " + spark_s
    if note:
        rest += "  " + paint(note, fg=DIM)
    return lab + val + rest


def render_run(run: Run, width: int, detail: bool) -> list[str]:
    row = run.last
    step = run.step
    budget = run.budget
    pct = (step / budget) if budget else 0.0
    prog = f"{step}/{budget}" if budget else f"{step}"
    rate = run.rate()
    rate_s = f"{rate:.2f}/s" if rate else "—/s"
    eta_s = f"ETA {fmt_dur(run.eta())}" if run.live else ""
    gpu = ""
    if run.proc and run.proc.gpu_index is not None:
        mem = f" {run.proc.gpu_mem_mib/1024:.1f}G" if run.proc.gpu_mem_mib else ""
        gpu = f"GPU{run.proc.gpu_index}{mem}"
    pid = f"pid {run.proc.pid}" if run.proc else ""
    tag_bits = [pulse(run.live), paint(run.kind, fg=PURPLE, bold=True), paint(prog, fg=ICE, bold=True)]
    if budget:
        tag_bits.append(paint(f"{100*pct:.0f}%", fg=CYAN))
    if run.live:
        tag_bits.append(paint(rate_s, fg=MINT))
        if eta_s:
            tag_bits.append(paint(eta_s, fg=GOLD))
    if gpu:
        tag_bits.append(paint(gpu, fg=MUTED))
    tag = "  ".join(tag_bits)
    title = paint(run.name, fg=WHITE, bold=True)
    lines = [_box_top(width, title, tag)]

    axis = ""
    if run.plus or run.minus:
        axis = paint(run.minus or "−", fg=CYAN) + paint("  ↔  ", fg=DIM) + paint(run.plus or "+", fg=ORANGE)
    meta = "   ".join(x for x in (axis, paint(run.recipe, fg=MUTED), paint(pid, fg=DIM)) if x)
    lines.append(_box_row(width, meta))

    # progress fader
    inner = width - 4
    fade_w = max(10, min(40, inner - 22))
    lines.append(_box_row(width, "         " + fader(pct, fade_w) + "  " + paint(prog, fg=ICE)))

    if not row:
        lines.append(_box_row(width, paint("waiting for the first jsonl step…", fg=YELLOW)))
        lines.append(_box_bot(width))
        return lines

    health, notes, good = health_of(run)
    spark_w = max(16, min(48, inner - 48))
    bar_w = 18 if inner > 90 else 12

    loss = metric(row, "loss")
    cp = metric(row, "cos_pos", "cos")
    cn = metric(row, "cos_neg")
    col = metric(row, "collapse")
    pp = metric(row, "pperc")
    np_ = metric(row, "nperc")
    ep = metric(row, "edrift_p")
    en = metric(row, "edrift_n")
    pdp = metric(row, "pdrift_p")
    pdn = metric(row, "pdrift_n")
    ev = run.last_eval()

    def row_out(label: str, value: str, val: float | None, lo: float, hi: float, xs: list[float], invert: bool, note: str) -> None:
        if val is None and not xs:
            return
        body, g = hbar(val if val is not None else 0.0, lo, hi, bar_w, invert=invert)
        fg = lerp_good(g) if val is not None else MUTED
        bar = paint(body, fg=fg)
        sp = paint(spark(xs, spark_w, lo=lo, hi=hi), fg=fg)
        lines.append(_box_row(width, metric_row(inner, label, value, bar, sp, note, value_fg=fg)))

    if loss is not None:
        xs = run.series("loss")
        window = xs[-max(spark_w * 3, 80) :]
        if len(window) >= 12:
            ordered = sorted(window)
            lo = ordered[len(ordered) // 10]
            hi = ordered[-1 - len(ordered) // 10]
        else:
            lo = min(window) if window else 0.0
            hi = max(window) if window else 1.0
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        # invert: lower loss is greener, but don't pin 0
        body, _ = hbar(loss, lo, hi, bar_w, invert=True)
        fg = CYAN
        sp = paint(spark(xs, spark_w, lo=lo, hi=hi), fg=CYAN)
        lines.append(
            _box_row(
                width,
                metric_row(inner, "loss", fmt(loss, 4), paint(body, fg=fg), sp, "", value_fg=CYAN),
            )
        )

    cos_xs_p = run.series("cos_pos") or run.series("cos")
    cos_xs_n = run.series("cos_neg")
    cos_show = pair_mean(row, "cos_pos", "cos_neg")
    if cos_show is None:
        cos_show = cp
    if cos_show is not None:
        note = ""
        if cp is not None and cn is not None:
            note = f"c+ {fmt(cp, 2)}  c− {fmt(cn, 2)}"
        elif cp is not None:
            note = f"cos {fmt(cp, 3)}"
        row_out("cosine", fmt(cos_show, 3), cos_show, -0.2, 1.0, cos_xs_p or [], False, note)
        if detail and cos_xs_n:
            row_out("c−", fmt(cn, 3), cn, -0.2, 1.0, cos_xs_n, False, "")

    if col is not None:
        note = "want −1"
        if col > 0:
            note = "⚠ same direction"
        elif col > -0.5:
            note = "still correlated"
        row_out("collapse", fmt(col, 2, signed=True), col, -1.0, 1.0, run.series("collapse"), True, note)

    if pp is not None or np_ is not None:
        perc = pair_mean(row, "pperc", "nperc")
        xs = run.series("pperc")
        note = ""
        if pp is not None and np_ is not None:
            note = f"p {fmt_pct(pp)}  n {fmt_pct(np_)}  want 0"
        row_out("residual", fmt_pct(perc), perc if perc is not None else 0.0, 0.0, 1.2, xs, True, note)

    if ep is not None or en is not None:
        ed = pair_mean(row, "edrift_p", "edrift_n")
        xs = run.series("edrift_p")
        note = f"p {fmt(ep, 3)}  n {fmt(en, 3)}"
        hi = max(0.4, max(xs) * 1.2) if xs else 0.4
        row_out("edrift", fmt(ed, 3), ed, 0.0, hi, xs, True, note)

    if pdp is not None and (abs(pdp) > 1e-8 or (pdn is not None and abs(pdn) > 1e-8)):
        pd = pair_mean(row, "pdrift_p", "pdrift_n")
        xs = run.series("pdrift_p")
        hi = max(abs(x) for x in xs) * 1.2 if xs else 0.01
        row_out("planreg", fmt(pd, 4), pd if pd is not None else 0.0, 0.0, max(hi, 1e-4), xs, True, "")

    if ev and run.kind == "TF":
        ev_cos = ev.get("cos")
        ev_col = ev.get("collapse")
        ev_mag = ev.get("mag")
        ev_proj = ev.get("proj_abs")
        bits = []
        if isinstance(ev_cos, (int, float)):
            bits.append(f"cos {ev_cos:.3f}")
        if isinstance(ev_col, (int, float)):
            bits.append(f"col {ev_col:+.2f}")
        if isinstance(ev_mag, (int, float)):
            bits.append(f"mag {ev_mag:.2f}")
        if isinstance(ev_proj, (int, float)):
            bits.append(f"proj {ev_proj:.3f}")
        if bits:
            lines.append(_box_row(width, paint("eval     ", fg=MUTED) + paint("  ".join(bits), fg=GOLD, bold=True) + paint("  (fixed probe)", fg=DIM)))

    # health
    hb, hg = hbar(health / 100.0, 0.0, 1.0, 16, invert=False)
    hfg = lerp_good(good)
    note = paint(" · ", fg=DIM).join(paint(n, fg=MUTED) for n in notes) if notes else ""
    lines.append(
        _box_row(
            width,
            paint("health   ", fg=MUTED)
            + paint(f"{health:3d}  ", fg=hfg, bold=True)
            + paint(hb, fg=hfg)
            + "  "
            + note,
        )
    )
    lines.append(_box_bot(width))
    return lines


def render_compare(runs: list[Run], width: int) -> list[str]:
    live = [r for r in runs if r.last]
    if len(live) < 2:
        return []
    inner = width - 4
    lines = [_box_top(width, paint("A/B", fg=GOLD, bold=True), paint("lower residual · more negative collapse · higher cosine", fg=DIM))]
    names = [r.name for r in live]
    col_w = max(14, (inner - 12) // len(live))
    short = shorten_names(names, max(12, col_w - 1))
    header = pad(paint("metric", fg=MUTED), 12)
    for s in short:
        header += pad(paint(s, fg=ICE, bold=True), col_w)
    lines.append(_box_row(width, header))

    def cells(label: str, vals: list[float | None], better: str, fmt_one) -> None:
        usable = [(i, v) for i, v in enumerate(vals) if v is not None]
        best_i = None
        if usable:
            if better == "high":
                best_i = max(usable, key=lambda iv: iv[1])[0]
            elif better == "low":
                best_i = min(usable, key=lambda iv: iv[1])[0]
            elif better == "neg":
                best_i = min(usable, key=lambda iv: iv[1])[0]
        row_s = pad(paint(label, fg=MUTED), 12)
        for i, v in enumerate(vals):
            text = fmt_one(v)
            fg = GREEN if i == best_i else WHITE
            row_s += pad(paint(text, fg=fg, bold=i == best_i), col_w)
        lines.append(_box_row(width, row_s))

    cells("cosine", [pair_mean(r.last, "cos_pos", "cos_neg") or metric(r.last, "cos_pos", "cos") for r in live], "high", lambda v: fmt(v, 3))
    cells("collapse", [metric(r.last, "collapse") for r in live], "neg", lambda v: fmt(v, 2, signed=True))
    cells("residual", [pair_mean(r.last, "pperc", "nperc") for r in live], "low", lambda v: fmt_pct(v))
    cells("loss", [metric(r.last, "loss") for r in live], "low", lambda v: fmt(v, 4))
    cells("edrift", [pair_mean(r.last, "edrift_p", "edrift_n") for r in live], "low", lambda v: fmt(v, 3))
    cells("health", [health_of(r)[0] for r in live], "high", lambda v: "—" if v is None else str(v))
    cells("step", [float(r.step) for r in live], "high", lambda v: str(int(v)) if v is not None else "—")
    lines.append(_box_bot(width))
    return lines


def render_recent(rows: list[dict[str, Any]], width: int) -> list[str]:
    if not rows:
        return []
    lines = [_box_top(width, paint("recent", fg=MUTED), paint("finished jsonl", fg=DIM))]
    for rec in rows:
        cos = rec.get("cos")
        col = rec.get("collapse")
        perc = rec.get("pperc")
        step = rec.get("step")
        budget = rec.get("budget")
        prog = f"{step}/{budget}" if step and budget else (str(step) if step else "")
        age = time.time() - rec["mtime"]
        age_s = fmt_dur(age) + " ago"
        name = rec["name"] if len(rec["name"]) <= 34 else "…" + rec["name"][-33:]
        bits = [
            pad(paint(name, fg=WHITE), 36),
            pad(paint(prog, fg=ICE), 11, "right"),
            pad(paint(f"c+ {fmt(cos, 2)}", fg=lerp_good((cos or 0))), 10),
            pad(
                paint(
                    f"col {fmt(col, 2, signed=True)}",
                    fg=lerp_good((1 - (col or 0)) / 2) if col is not None else MUTED,
                ),
                12,
            ),
        ]
        if perc is not None:
            bits.append(pad(paint(f"p% {fmt_pct(perc)}", fg=MUTED), 12))
        bits.append(paint(age_s, fg=DIM))
        lines.append(_box_row(width, "  ".join(bits)))
    lines.append(_box_bot(width))
    return lines


def help_line(width: int) -> str:
    keys = "q quit   j/k cycle   a all runs   space pause   r rescan"
    return pad(paint(keys, fg=DIM), width)


def render_board(
    runs: list[Run],
    gpus: list[Gpu],
    recent: list[dict[str, Any]],
    width: int,
    height: int,
    focus: int,
    show_all: bool,
    paused: bool,
    fit: bool = True,
) -> str:
    clock = time.strftime("%H:%M:%S")
    n_live = sum(1 for r in runs if r.live)
    lines = [header_line(width, n_live, gpus, paused, clock), ""]

    if not runs:
        lines.append(paint("  no live training jsonl found under models/", fg=YELLOW))
        lines.append(paint("  pass a path, or start a trainer — watching for *_train.jsonl", fg=DIM))
    else:
        visible: list[Run]
        if show_all or len(runs) <= 3:
            visible = runs
        else:
            visible = [runs[focus % len(runs)]]
            # always keep other live runs as compare partners
            for run in runs:
                if run is not visible[0] and run.live and run not in visible:
                    visible.append(run)
                    if len(visible) >= 3:
                        break
        detail = len(visible) == 1
        for run in visible:
            lines.extend(render_run(run, width, detail=detail))
            lines.append("")
        if len(visible) >= 2:
            lines.extend(render_compare(visible, width))
            lines.append("")

    if recent:
        lines.extend(render_recent(recent, width))
        lines.append("")
    lines.append(help_line(width))

    if fit:
        if len(lines) > height:
            lines = lines[:height]
        while len(lines) < height:
            lines.append("")
        lines = lines[:height]
    return "\n".join(pad(line, width) for line in lines)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


class Tty:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old = None
        self.alt = False

    def enter(self) -> None:
        if not _isatty():
            return
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        self.alt = True
        if self.fd is not None:
            try:
                import termios
                import tty

                self.old = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except (termios.error, OSError):
                self.old = None

    def leave(self) -> None:
        if self.fd is not None and self.old is not None:
            try:
                import termios

                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except (termios.error, OSError):
                pass
        if self.alt:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            self.alt = False

    def key(self, timeout: float) -> str | None:
        if self.fd is None:
            time.sleep(timeout)
            return None
        try:
            ready, _, _ = select.select([self.fd], [], [], timeout)
        except (OSError, ValueError):
            time.sleep(timeout)
            return None
        if not ready:
            return None
        try:
            ch = os.read(self.fd, 8).decode("utf-8", "ignore")
        except OSError:
            return None
        return ch


def default_root() -> Path:
    cwd = Path.cwd()
    candidates = [
        CONCEPTMOD / "models",
        cwd / "sliders-conceptmod" / "models",
        cwd / "models",
    ]
    ranked: list[tuple[int, Path]] = []
    for cand in candidates:
        if not cand.is_dir():
            continue
        try:
            has = next(cand.glob("*_train.jsonl"), None) or next(cand.rglob("*_train.jsonl"), None)
        except OSError:
            has = None
        ranked.append((1 if has is not None else 0, cand))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else DEFAULT_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path, help="jsonl files, run dirs, or train.log captures")
    p.add_argument("--root", type=Path, default=None, help="models/ directory to scan (default: auto)")
    p.add_argument("--once", action="store_true", help="print one frame and exit")
    p.add_argument("--interval", type=float, default=0.5, help="refresh seconds (default 0.5)")
    p.add_argument("--live-window", type=float, default=20.0, help="jsonl mtime seconds to count as live")
    p.add_argument("--recent", type=int, default=6, help="finished runs to list in the footer")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    enable_color(False if args.no_color else None)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    root = args.root or default_root()
    explicit = [p if p.is_absolute() else Path.cwd() / p for p in args.paths]
    tty = Tty()
    focus = 0
    show_all = True
    paused = False
    stop = False
    cache_runs: list[Run] = []
    cache_index: dict[str, Run] = {}
    last_scan = 0.0
    last_gpu = 0.0
    gpus: list[Gpu] = []

    def handle_stop(_signum=None, _frame=None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    def merge_discover() -> tuple[list[Run], list[Gpu]]:
        fresh, found_gpus = discover(root, explicit, args.live_window)
        merged: list[Run] = []
        seen = set()
        for run in fresh:
            prev = cache_index.get(run.key)
            if prev is None:
                cache_index[run.key] = run
                merged.append(run)
            else:
                prev.proc = run.proc
                prev.live = run.live
                prev.budget = run.budget or prev.budget
                prev.recipe = run.recipe or prev.recipe
                prev.plus = run.plus or prev.plus
                prev.minus = run.minus or prev.minus
                prev.kind = run.kind or prev.kind
                merged.append(prev)
            seen.add(run.key)
        # keep recently live runs that just finished (jsonl still useful)
        now = time.time()
        for key, run in list(cache_index.items()):
            if key in seen:
                continue
            age = now - (run.tail.mtime or 0)
            if age < 60:
                run.live = False
                merged.append(run)
            else:
                cache_index.pop(key, None)
        merged.sort(key=lambda r: (0 if r.live else 1, -r.tail.mtime, r.name))
        return merged, found_gpus

    def frame() -> str:
        nonlocal last_scan, last_gpu, cache_runs, gpus
        now = time.time()
        if now - last_scan > 2.0 or not cache_runs:
            cache_runs, gpus = merge_discover()
            last_scan = now
            last_gpu = now
        elif now - last_gpu > 1.0:
            gpus = scan_gpus()[0]
            last_gpu = now
        for run in cache_runs:
            run.poll()
            if run.proc and not Path(f"/proc/{run.proc.pid}").exists():
                run.live = False
                run.proc = None
        w, h = term_size(args.width, args.height)
        exclude = {r.key for r in cache_runs}
        recent = recent_finished(root, args.recent, exclude) if not explicit else []
        show = show_all or len(cache_runs) <= 2
        return render_board(
            cache_runs,
            gpus,
            recent,
            w,
            h,
            focus,
            show,
            paused,
            fit=not args.once and _isatty(),
        )

    if args.once or not _isatty():
        sys.stdout.write(frame() + "\n")
        return 0

    tty.enter()
    try:
        last_draw = ""
        while not stop:
            if not paused:
                drawn = frame()
            else:
                drawn = last_draw or frame()
            if drawn != last_draw:
                sys.stdout.write("\x1b[H" + drawn)
                sys.stdout.flush()
                last_draw = drawn
            key = tty.key(args.interval if not paused else 0.1)
            if not key:
                continue
            if key in {"q", "Q", "\x03"}:
                break
            if key in {" "}:
                paused = not paused
                last_draw = ""
            if key in {"r", "R"}:
                last_scan = 0.0
                last_draw = ""
            if key in {"a", "A"}:
                show_all = not show_all
                last_draw = ""
            if key in {"j", "n", "\x1b[B"}:
                if cache_runs:
                    focus = (focus + 1) % len(cache_runs)
                    show_all = False
                    last_draw = ""
            if key in {"k", "p", "\x1b[A"}:
                if cache_runs:
                    focus = (focus - 1) % len(cache_runs)
                    show_all = False
                    last_draw = ""
            if key.isdigit() and key != "0":
                idx = int(key) - 1
                if cache_runs and idx < len(cache_runs):
                    focus = idx
                    show_all = False
                    last_draw = ""
    finally:
        tty.leave()
    return 0


if __name__ == "__main__":
    sys.exit(main())
