"""SQLite is the durable source of truth for jobs, images, reviews and ownership."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .contracts import canonical


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, group_id TEXT NOT NULL, priority INTEGER NOT NULL,
                    status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, payload TEXT NOT NULL,
                    error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL, updated REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS job_queue ON jobs(status,priority,created);
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
                    metadata TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                    data TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    sha256 TEXT PRIMARY KEY, variation TEXT NOT NULL, path TEXT NOT NULL,
                    metadata TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL,
                    step INTEGER NOT NULL DEFAULT 0, metrics TEXT NOT NULL DEFAULT '[]', error TEXT);
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, image_id TEXT NOT NULL,
                    data TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS controls (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            for key, value in dict(paused=False, render_burst=0, training_updates=20, owner=None).items():
                db.execute("INSERT OR IGNORE INTO controls VALUES (?,?)", (key, canonical(value)))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def emit(self, db, kind, data):
        db.execute("INSERT INTO events(kind,data,created) VALUES (?,?,?)", (kind, canonical(data), time.time()))

    def enqueue(self, payloads, group=None):
        group = group or uuid.uuid4().hex
        ids = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = [r[0] for r in db.execute("SELECT id FROM jobs WHERE group_id=? ORDER BY priority", (group,))]
            if existing:
                return dict(ids=existing, group=group)
            priority = db.execute("SELECT COALESCE(MAX(priority),0) FROM jobs").fetchone()[0]
            for payload in payloads:
                ident = uuid.uuid4().hex
                now = time.time()
                priority += 1
                db.execute("INSERT INTO jobs(id,group_id,priority,status,payload,created,updated) VALUES (?,?,?,?,?,?,?)",
                           (ident, group, priority, "queued", canonical(payload), now, now))
                ids.append(ident)
            self.emit(db, "queue", dict(ids=ids, group=group))
        return dict(ids=ids, group=group)

    @staticmethod
    def unpack(row):
        if row is None:
            return None
        value = dict(row)
        for key in ("payload", "metadata", "data", "config", "metrics"):
            if key in value:
                value[key] = json.loads(value[key])
        return value

    def jobs(self, limit=200):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute(
                """SELECT * FROM jobs WHERE status IN ('queued','running','cancelling') OR id IN
                   (SELECT id FROM jobs WHERE status NOT IN ('queued','running','cancelling') ORDER BY updated DESC LIMIT ?)
                   ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'cancelling' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, priority""", (limit,))]

    def job(self, ident):
        with self.connect() as db:
            return self.unpack(db.execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone())

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY priority,created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated=? WHERE id=?", (time.time(), row["id"]))
            self.emit(db, "job", dict(id=row["id"], status="running"))
            return self.unpack(row)

    def recover(self):
        """Only the coordinator holding the OS lock may invoke recovery."""
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued',progress=0,error='Worker restarted; recovered',updated=? WHERE status='running'", (time.time(),))
            db.execute("UPDATE jobs SET status='cancelled',updated=? WHERE status='cancelling'", (time.time(),))
            db.execute("UPDATE training_runs SET status='queued' WHERE status='running'")
            self.emit(db, "recovery", {})

    def progress(self, ident, progress):
        with self.connect() as db:
            db.execute("UPDATE jobs SET progress=?,updated=? WHERE id=? AND status='running'", (progress, time.time(), ident))
            self.emit(db, "progress", dict(id=ident, progress=progress))

    def cancel(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            if row[0] in ("queued", "running"):
                status = "cancelled" if row[0] == "queued" else "cancelling"
                db.execute("UPDATE jobs SET status=?,updated=? WHERE id=?", (status, time.time(), ident))
                self.emit(db, "job", dict(id=ident, status=status))

    def retry(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            if row[0] not in ("failed", "cancelled"):
                raise ValueError("Only failed or cancelled jobs can be retried")
            db.execute("UPDATE jobs SET status='queued',error=NULL,progress=0,updated=? WHERE id=?", (time.time(), ident))
            self.emit(db, "queue", dict(retry=ident))

    def reorder(self, ids):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            actual = {r[0] for r in db.execute("SELECT id FROM jobs WHERE status='queued'")}
            if len(ids) != len(set(ids)) or set(ids) != actual:
                raise ValueError("Reorder must contain each currently queued job exactly once")
            for position, ident in enumerate(ids):
                db.execute("UPDATE jobs SET priority=? WHERE id=?", (position, ident))
            self.emit(db, "queue", dict(order=ids))

    def finish(self, ident, path, metadata):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = db.execute("SELECT status FROM jobs WHERE id=?", (ident,)).fetchone()[0]
            if status in ("cancelled", "cancelling"):
                db.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (ident,))
                self.emit(db, "job", dict(id=ident, status="cancelled"))
                return False
            if status != "running":
                raise ValueError("Only a running job can finish")
            relative = str(Path(path).resolve().relative_to(self.path.parent.resolve()))
            db.execute("INSERT INTO images VALUES (?,?,?,?,?)", (ident, ident, relative, canonical(metadata), time.time()))
            db.execute("UPDATE jobs SET status='completed',progress=1,error=NULL,updated=? WHERE id=?", (time.time(), ident))
            self.emit(db, "image", dict(id=ident))
            return True

    def fail(self, ident, error, cancelled=False):
        with self.connect() as db:
            status = "cancelled" if cancelled else "failed"
            db.execute("UPDATE jobs SET status=?,error=?,updated=? WHERE id=?", (status, str(error), time.time(), ident))
            self.emit(db, "job", dict(id=ident, status=status, error=str(error)))

    def history(self, limit=100, offset=0):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute("SELECT * FROM images ORDER BY created DESC LIMIT ? OFFSET ?", (limit, offset))]

    def image(self, ident):
        with self.connect() as db:
            value = self.unpack(db.execute("SELECT * FROM images WHERE id=?", (ident,)).fetchone())
            if value:
                value["path"] = str(self.path.parent / value["path"])
            return value

    def events(self, after):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 100", (after,))]

    def controls(self):
        with self.connect() as db:
            return {r[0]: json.loads(r[1]) for r in db.execute("SELECT * FROM controls")}

    def control(self, **values):
        with self.connect() as db:
            for key, value in values.items():
                db.execute("INSERT OR REPLACE INTO controls VALUES (?,?)", (key, canonical(value)))
            self.emit(db, "controls", values)

    def catalog(self):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute("SELECT * FROM checkpoints ORDER BY variation,created DESC")]

    def register_checkpoint(self, path, variation):
        from safetensors import safe_open
        from .contracts import VARIATIONS, file_hash
        from .particles import FORMAT
        path = Path(path).resolve()
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = json.loads(f.metadata()["anima"])
        if variation not in VARIATIONS or metadata["format"] != FORMAT or metadata["weights"] != "ema":
            raise ValueError("Studio accepts native immutable EMA snapshots")
        sha = file_hash(path)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO checkpoints VALUES (?,?,?,?,?)", (sha, variation, str(path), canonical(metadata), time.time()))
            self.emit(db, "catalog", dict(sha256=sha))
        return sha

    def add_run(self, ident, config):
        with self.connect() as db:
            db.execute("INSERT INTO training_runs(id,config,status) VALUES (?,?,'queued')", (ident, canonical(config)))
            self.emit(db, "training", dict(id=ident, status="queued"))

    def runs(self):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute("SELECT * FROM training_runs ORDER BY id")]

    def update_run(self, ident, status, step, metrics=None, error=None):
        with self.connect() as db:
            if metrics is not None:
                old = json.loads(db.execute("SELECT metrics FROM training_runs WHERE id=?", (ident,)).fetchone()[0])
                # Re-probing a committed step replaces its chart point.
                old = [m for m in old if m["step"] != metrics["step"]] + [metrics]
                db.execute("UPDATE training_runs SET metrics=? WHERE id=?", (canonical(old), ident))
            db.execute("UPDATE training_runs SET status=?,step=?,error=? WHERE id=?", (status, step, error, ident))
            self.emit(db, "training", dict(id=ident, status=status, step=step))

    def review(self, ident, data):
        if self.image(ident) is None:
            raise KeyError(ident)
        with self.connect() as db:
            db.execute("INSERT INTO reviews(image_id,data,created) VALUES (?,?,?)", (ident, canonical(data), time.time()))
            self.emit(db, "review", dict(id=ident))

    def reviews(self):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute("SELECT * FROM reviews ORDER BY id")]
