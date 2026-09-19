#!/usr/bin/env python3
"""Threaded listen browser for eval/listen.

Serves the whole listen tree on 0.0.0.0. Featured sets come from
eval/listen/queue.json (re-read on every home page). Everything else is
listed newest-first. Pages are generated from the wavs on disk, so a
render in flight shows up without a rebuild.

    python scripts/listen_server.py
    python scripts/listen_server.py --port 8888 --root eval/listen
"""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import time
import wave
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LISTEN = ROOT / "eval" / "listen"
DEFAULT_PORT = 8888

CSS = """
:root { color-scheme: dark; }
body { font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 1.5rem 2rem 4rem;
       background: #12131a; color: #e7e7ee; }
a { color: #8ab4ff; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.35rem; margin: 0 0 .2rem; }
p.sub, .note, .metrics, .meta, .empty { color: #9a9aab; }
p.sub { margin: 0 0 1.4rem; max-width: 72rem; }
nav.toc { display: flex; flex-wrap: wrap; gap: .4rem; margin: 0 0 1.5rem; }
nav.toc a { font: 12px/1.3 ui-monospace, monospace; background: #232536; color: #cfd0e0;
            padding: .25rem .55rem; border-radius: 999px; }
section { border: 1px solid #2a2c39; border-radius: 10px; padding: 1rem 1.25rem;
          margin-bottom: 1.1rem; background: #191b24; }
section.waiting { opacity: .72; }
h2 { font-size: 1.05rem; margin: 0 0 .15rem; }
h2 .when { font-size: .75rem; font-weight: 600; background: #7fd7a5; color: #12131a;
           padding: .1rem .45rem; border-radius: 5px; margin-right: .45rem;
           font-family: ui-sans-serif, system-ui, sans-serif; vertical-align: middle; }
h3 { font-size: .92rem; margin: 1rem 0 .45rem; font-family: ui-monospace, monospace; color: #d7d7e4; }
.note { margin: .2rem 0 .7rem; max-width: 72rem; }
.metrics { font-size: .82rem; font-family: ui-monospace, monospace; margin: 0 0 .6rem; }
.row { display: flex; justify-content: space-between; align-items: baseline; gap: .75rem; flex-wrap: wrap; }
button { font: 12px/1.2 ui-sans-serif, system-ui, sans-serif; background: #2a2c39; color: #e7e7ee;
         border: 1px solid #3a3c4d; border-radius: 6px; padding: .25rem .55rem; cursor: pointer; }
button:hover { background: #34364a; }
.clips { display: flex; flex-wrap: wrap; gap: .7rem; }
.clip { flex: 1 1 240px; min-width: 220px; max-width: 420px; background: #12131a;
        border: 1px solid #2a2c39; border-radius: 8px; padding: .7rem .8rem; }
.clip.playing { border-color: #7fd7a5; }
.clip .name { display: block; font: 12px/1.3 ui-monospace, monospace; color: #d7d7e4; margin-bottom: .15rem;
              word-break: break-all; }
.clip .meta { font: 11px/1.3 ui-monospace, monospace; margin: 0 0 .35rem; }
.clip.fresh { border-color: #7fd7a5; }
.clip .tag { font-size: .68rem; font-weight: 600; padding: .08rem .4rem; border-radius: 4px; margin-left: .3rem; }
.tag.fresh { background: #7fd7a5; color: #12131a; }
.tag.slider { background: #2a2c39; color: #cfd0e0; }
.tag.ref { background: #1e2a3a; color: #8ab4ff; }
.tag.solo { background: #3a2a12; color: #f0c674; }
.tag.cum { background: #2a1e33; color: #d4a0e8; }
.tag.base { background: #1e3324; color: #7fd7a5; }
.tag.hot { background: #e07a7a; color: #12131a; }
.empty { font-size: .85rem; font-family: ui-monospace, monospace; margin: .3rem 0 .2rem; }
audio { width: 100%; }
table { border-collapse: collapse; font: 12px/1.4 ui-monospace, monospace; margin: .4rem 0 .8rem; }
th, td { text-align: left; padding: .15rem .6rem .15rem 0; color: #b9b9c8; }
th { color: #9a9aab; font-weight: 600; }
.index { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: .45rem; }
.index a { display: block; background: #191b24; border: 1px solid #2a2c39; border-radius: 8px;
           padding: .55rem .7rem; color: #e7e7ee; }
.index a:hover { border-color: #4a4c5d; text-decoration: none; }
.index .n { color: #9a9aab; font-size: .78rem; }
h3 .mtime, h2 .mtime { font-size: .75rem; font-weight: 500; color: #9a9aab;
                       font-family: ui-monospace, monospace; margin-left: .5rem; }
"""

JS = """
(function () {
  const ordered = (sec) => [...sec.querySelectorAll("audio")].sort(
    (a, b) => Number(a.dataset.seq) - Number(b.dataset.seq)
  );
  const sections = document.querySelectorAll("[data-playlist]");
  sections.forEach((sec) => {
    const audios = [...sec.querySelectorAll("audio")];
    audios.forEach((a) => {
      a.addEventListener("play", () => {
        audios.forEach((b) => { if (b !== a) b.pause(); });
        sec.querySelectorAll(".clip").forEach((c) => c.classList.remove("playing"));
        a.closest(".clip")?.classList.add("playing");
      });
      a.addEventListener("ended", () => {
        const seq = ordered(sec);
        const n = seq[seq.indexOf(a) + 1];
        if (n) { n.currentTime = 0; n.play(); }
      });
    });
    sec.querySelector("[data-play-seq]")?.addEventListener("click", () => {
      const seq = ordered(sec);
      if (!seq[0]) return;
      audios.forEach((a) => { a.pause(); a.currentTime = 0; });
      seq[0].play();
    });
  });
  setInterval(() => {
    if (![...document.querySelectorAll("audio")].some((a) => !a.paused)) location.reload();
  }, 10000);
})();
"""


FRESH_SEC = 5 * 60


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / float(rate) if rate else None
    except Exception:
        return None


def fmt_mtime(ts: float, now: float | None = None) -> str:
    now = time.time() if now is None else now
    dt = datetime.fromtimestamp(ts)
    today = datetime.fromtimestamp(now).date()
    clock = dt.strftime("%H:%M") if dt.date() == today else dt.strftime("%m-%d %H:%M")
    delta = max(0.0, now - ts)
    if delta < 60:
        rel = f"{int(delta)}s ago"
    elif delta < 3600:
        rel = f"{int(delta / 60)}m ago"
    elif delta < 86400:
        rel = f"{int(delta / 3600)}h ago"
    else:
        rel = f"{int(delta / 86400)}d ago"
    return f"{rel} · {clock}"


def folder_stats(folder: Path) -> tuple[int, float]:
    if not folder.is_dir():
        return 0, 0.0
    n = 0
    latest = folder.stat().st_mtime
    for p in folder.rglob("*.wav"):
        n += 1
        latest = max(latest, p.stat().st_mtime)
    return n, latest


def wav_sort_key(name: str) -> tuple:
    stem = Path(name).stem
    if stem.startswith("00_") or stem.endswith("_base") or stem == "base":
        return (0, stem)
    if stem.startswith("solo_"):
        return (1, stem)
    if stem.startswith("cum_"):
        return (2, stem)
    if "REF" in stem:
        return (4, stem)
    return (3, stem)


def clip_tag(stem: str) -> str:
    if stem.startswith("solo_"):
        return "solo"
    if stem.startswith("cum_"):
        return "cum"
    if "base" in stem.lower() or "zero" in stem.lower():
        return "base"
    if "REF" in stem:
        return "ref"
    return "slider"


def load_queue(listen_root: Path) -> list[dict]:
    path = listen_root / "queue.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def load_triage(folder: Path) -> dict[str, dict]:
    path = folder / "triage.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, dict] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        clip = str(row.get("clip") or "")
        if clip:
            out[clip] = row
    return out


def count_wavs(folder: Path) -> int:
    return folder_stats(folder)[0]


def folder_wavs(folder: Path) -> list[Path]:
    wavs = list(folder.glob("*.wav"))
    wavs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return wavs


def play_seq(wavs: list[Path]) -> dict[Path, int]:
    ordered = sorted(wavs, key=lambda p: wav_sort_key(p.name))
    return {p: i for i, p in enumerate(ordered)}


def rel_url(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "/" + rel


def clip_html(
    wav: Path,
    root: Path,
    triage: dict[str, dict] | None = None,
    *,
    seq: int = 0,
    now: float | None = None,
) -> str:
    stem = wav.stem
    tag = clip_tag(stem)
    dur = wav_duration(wav)
    dur_s = f"{dur:.1f}s" if dur is not None else "?"
    mtime = wav.stat().st_mtime
    now = time.time() if now is None else now
    fresh = (now - mtime) < FRESH_SEC
    extra = ""
    row = (triage or {}).get(stem)
    if row:
        bits = []
        if "rms" in row:
            bits.append(f"rms {row['rms']}")
        if "centroid_hz" in row:
            bits.append(f"cent {row['centroid_hz']}")
        if "bpm" in row:
            bits.append(f"bpm {row['bpm']}")
        extra = " · ".join(str(b) for b in bits)
        extra = f" · {html.escape(extra)}" if extra else ""
        rms = row.get("rms")
        if isinstance(rms, (int, float)) and rms > 0.35:
            tag = "hot"
    src = html.escape(rel_url(wav, root))
    fresh_cls = " fresh" if fresh else ""
    fresh_tag = '<span class="tag fresh">new</span>' if fresh else ""
    return (
        f'<div class="clip{fresh_cls}"><span class=name>{html.escape(wav.name)}'
        f'<span class="tag {tag}">{tag}</span>{fresh_tag}</span>'
        f'<p class=meta>{dur_s} · {html.escape(fmt_mtime(mtime, now))}{extra}</p>'
        f'<audio controls preload=none data-seq="{seq}" src="{src}"></audio></div>'
    )


def set_html(title: str, folder: Path, root: Path, *, waiting_label: str | None = None) -> str:
    n, latest = folder_stats(folder) if folder.is_dir() else (0, 0.0)
    stamp = f'<span class=mtime>{html.escape(fmt_mtime(latest))}</span>' if n else ""
    parts = [f"<div data-playlist><div class=row><h3>{html.escape(title)}{stamp}</h3>"]
    wavs = folder_wavs(folder) if folder.is_dir() else []
    if wavs:
        parts.append('<button type=button data-play-seq>play in order</button></div>')
        triage = load_triage(folder)
        seq = play_seq(wavs)
        now = time.time()
        parts.append('<div class=clips>')
        parts.extend(clip_html(w, root, triage, seq=seq[w], now=now) for w in wavs)
        parts.append("</div>")
        md = folder / "LISTEN.md"
        if md.exists():
            first = next((ln.strip() for ln in md.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
            if first:
                parts.append(f'<p class=metrics>{html.escape(first.lstrip("# ").strip())}</p>')
    else:
        parts.append("</div>")
        label = waiting_label or "no wavs yet"
        parts.append(f'<p class=empty>waiting — {html.escape(label)}</p>')
    parts.append("</div>")
    return "\n".join(parts)


def featured_html(item: dict, root: Path) -> str:
    rel = str(item.get("path") or "")
    folder = (root / rel).resolve()
    try:
        folder.relative_to(root.resolve())
    except ValueError:
        return ""
    title = str(item.get("title") or rel)
    when = str(item.get("when") or "")
    note = str(item.get("note") or "")
    children = [str(c) for c in (item.get("children") or [])]
    n, latest = folder_stats(folder) if folder.is_dir() else (0, 0.0)
    waiting = n == 0
    parts = [f'<section id="{html.escape(rel)}" class="{"waiting" if waiting else ""}">']
    when_h = f'<span class=when>{html.escape(when)}</span>' if when else ""
    stamp = f'<span class=mtime>{html.escape(fmt_mtime(latest))}</span>' if n else ""
    parts.append(
        f"<h2>{when_h}{html.escape(title)} <a href='/{html.escape(rel)}/'>/{html.escape(rel)}/</a>{stamp}</h2>"
    )
    if note:
        parts.append(f'<p class=note>{html.escape(note)}</p>')
    parts.append(f'<p class=metrics>{n} wavs</p>')
    if children:
        kids = [(folder / child, child) for child in children]
        kids.sort(key=lambda kv: folder_stats(kv[0])[1], reverse=True)
        for sub, child in kids:
            parts.append(set_html(child, sub, root, waiting_label=f"{rel}/{child}"))
    else:
        parts.append(set_html(rel, folder, root, waiting_label=rel))
    parts.append("</section>")
    return "\n".join(parts)


def index_html(root: Path) -> str:
    queue = load_queue(root)
    featured_paths = {str(item.get("path") or "") for item in queue}
    dirs = []
    for p in root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        n, latest = folder_stats(p)
        if n == 0 and p.name not in featured_paths:
            continue
        dirs.append((latest, p.name, n))
    dirs.sort(reverse=True)

    parts = [
        "<!doctype html><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width, initial-scale=1">',
        "<title>listen</title>",
        f"<style>{CSS}</style>",
        "<h1>listen</h1>",
        "<p class=sub>Featured queue from <code>queue.json</code>. "
        "Clips and folders sort newest-first. Play-in-order still follows filename order "
        "(−2 → +2, solos before cumulative). "
        "Page reloads every 10s while nothing is playing.</p>",
    ]
    if queue:
        parts.append('<nav class=toc>')
        for item in queue:
            rel = html.escape(str(item.get("path") or ""))
            title = html.escape(str(item.get("title") or rel))
            parts.append(f'<a href="#{rel}">{title}</a>')
        parts.append("</nav>")
        for item in queue:
            parts.append(featured_html(item, root))
    parts.append("<section><h2>all sets</h2><div class=index>")
    now = time.time()
    for latest, name, n in dirs:
        stamp = fmt_mtime(latest, now) if n else "waiting"
        parts.append(
            f'<a href="/{html.escape(name)}/"><code>{html.escape(name)}</code>'
            f'<div class=n>{n} wavs · {html.escape(stamp)}</div></a>'
        )
    parts.append("</div></section>")
    parts.append(f"<script>{JS}</script>")
    return "\n".join(parts)


def dir_page(folder: Path, root: Path) -> str:
    rel = folder.relative_to(root).as_posix()
    n, latest = folder_stats(folder)
    stamp = f'<span class=mtime>{html.escape(fmt_mtime(latest))}</span>' if n else ""
    subs = [p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")]
    subs.sort(key=lambda p: folder_stats(p)[1], reverse=True)
    parts = [
        "<!doctype html><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(rel)}</title>",
        f"<style>{CSS}</style>",
        f'<p class=sub><a href="/">listen</a> / {html.escape(rel)}</p>',
        f"<h1>{html.escape(rel)}{stamp}</h1>",
        "<section data-playlist>",
    ]
    wavs = folder_wavs(folder)
    if wavs:
        parts.append('<div class=row><h2>clips</h2><button type=button data-play-seq>play in order</button></div>')
        triage = load_triage(folder)
        seq = play_seq(wavs)
        now = time.time()
        parts.append('<div class=clips>')
        parts.extend(clip_html(w, root, triage, seq=seq[w], now=now) for w in wavs)
        parts.append("</div>")
    if subs:
        parts.append("<h2>folders</h2><div class=index>")
        now = time.time()
        for sub in subs:
            sn, slatest = folder_stats(sub)
            stamp_s = fmt_mtime(slatest, now) if sn else "waiting"
            parts.append(
                f'<a href="/{html.escape(sub.relative_to(root).as_posix())}/">'
                f'<code>{html.escape(sub.name)}</code>'
                f'<div class=n>{sn} wavs · {html.escape(stamp_s)}</div></a>'
            )
        parts.append("</div>")
        for sub in subs:
            parts.append(set_html(sub.name, sub, root))
    if not wavs and not subs:
        parts.append('<p class=empty>waiting — no wavs yet</p>')
    parts.append("</section>")
    parts.append(f"<script>{JS}</script>")
    return "\n".join(parts)


class ListenHandler(SimpleHTTPRequestHandler):
    listen_root: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.listen_root), **kwargs)

    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _local_path(self, path: str) -> Path | None:
        parsed = urlparse(path)
        raw = unquote(parsed.path)
        rel = posixpath.normpath(raw).lstrip("/")
        root = self.listen_root.resolve()
        link = root / rel
        dest = link.resolve()
        try:
            dest.relative_to(root)
        except ValueError:
            # Research pages explicitly link immutable recordings kept in analysis.
            # Permit those individual WAV links, not external directory browsing.
            if (link.is_symlink() and link.suffix.lower() == ".wav"
                    and link.parent.resolve().is_relative_to(root)
                    and dest.is_relative_to((ROOT / "analysis").resolve())
                    and dest.suffix.lower() == ".wav" and dest.is_file()):
                return dest
            return None
        return dest

    def translate_path(self, path: str) -> str:
        dest = self._local_path(path)
        if dest is None:
            raise ValueError("Path is outside the listening tree")
        return str(dest)

    def send_head(self):
        if self._local_path(self.path) is None:
            self.send_error(404, "File not found")
            return None
        return super().send_head()

    def list_directory(self, path: str):
        folder = Path(path)
        root = self.listen_root.resolve()
        if folder.resolve() == root:
            self._send_html(index_html(root))
        else:
            self._send_html(dir_page(folder, root))
        return None


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_LISTEN)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    ListenHandler.listen_root = root
    httpd = Server((args.bind, args.port), ListenHandler)
    print(f"listen server {args.bind}:{args.port}  root={root}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
