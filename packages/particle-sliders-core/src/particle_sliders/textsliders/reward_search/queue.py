"""Atomic GPU work claims; each matched seed bundle stays on one GPU."""
import json
import os
import re
import uuid
import socket
import sqlite3
import time
from pathlib import Path


def connect(run):
    db=sqlite3.connect(Path(run)/'queue.sqlite',timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, stage TEXT, payload TEXT, status TEXT, gpu INTEGER, updated REAL, error TEXT)')
    # Serialize migration with claims; a pre-owner database is retained.
    db.execute('BEGIN IMMEDIATE')
    if 'owner' not in {row[1] for row in db.execute('PRAGMA table_info(jobs)')}:
        db.execute('ALTER TABLE jobs ADD COLUMN owner TEXT')
    db.commit()
    return db


def _owner():
    tail = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(host=socket.gethostname(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=os.getpid(), start=tail[19])


def _owner_alive(owner):
    if (not isinstance(owner, dict) or set(owner) != {'host', 'boot', 'pid', 'start'}
            or type(owner['pid']) is not int or owner['pid'] <= 0
            or any(not isinstance(owner[key], str) or not owner[key] for key in ('host', 'boot', 'start'))
            or re.fullmatch(r'[0-9]+', owner['start']) is None):
        return None  # Legacy/foreign/corrupt claims need explicit reconciliation.
    try:
        if str(uuid.UUID(owner['boot'])) != owner['boot']:
            return None
    except ValueError:
        return None
    try:
        current = _owner()
    except OSError:
        return None
    try:
        if owner['host'] != current['host']:
            return None
        if owner['boot'] != current['boot']:
            return False
        tail = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return tail[19] == owner['start'] and tail[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return None


def _recover(db, stage=None):
    sql = "SELECT id,owner FROM jobs WHERE status='running'"
    rows = db.execute(sql + (' AND stage=?' if stage is not None else ''),
                      (stage,) if stage is not None else ()).fetchall()
    result = dict(failed=[], unverified=[])
    for ident, encoded in rows:
        try:
            owner = json.loads(encoded) if encoded else None
        except (TypeError, ValueError):
            owner = None
        alive = _owner_alive(owner)
        if alive is False:
            # Never rerender an abandoned bundle: it may already have output.
            db.execute("UPDATE jobs SET status='failed',updated=?,error=? WHERE id=? AND status='running'",
                       (time.time(), 'Claim owner exited; retained without automatic reroll', ident))
            result['failed'].append(ident)
        elif alive is None:
            result['unverified'].append(ident)
    return result


def recover_abandoned(run, stage=None):
    """Terminalize proven-dead local claims, preserving every job/output."""
    with connect(run) as db:
        db.execute('BEGIN IMMEDIATE')
        return _recover(db, stage)


def add(run, stage, jobs):
    with connect(run) as db:
        for job in jobs:
            ident=f"{stage}/{job['family']['family']}/{job['seed']}"
            payload=json.dumps(job,sort_keys=True)
            previous=db.execute('SELECT payload FROM jobs WHERE id=?',(ident,)).fetchone()
            if previous and previous[0]!=payload:
                raise ValueError('Queued experiment changed: '+ident)
            db.execute('INSERT OR IGNORE INTO jobs (id,stage,payload,status,gpu,updated,error) VALUES (?,?,?,?,?,?,?)',
                       (ident,stage,payload,'pending',None,time.time(),None))


def claim(run, gpu):
    with connect(run) as db:
        db.execute('BEGIN IMMEDIATE')
        _recover(db)
        row=db.execute("SELECT id,stage,payload FROM jobs WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone()
        if row is None:return None
        db.execute("UPDATE jobs SET status='running',gpu=?,updated=?,owner=? WHERE id=?",
                   (gpu,time.time(),json.dumps(_owner(),sort_keys=True),row[0]))
        return dict(id=row[0],stage=row[1],payload=json.loads(row[2]))


def finish(run, ident, error=None):
    with connect(run) as db:
        db.execute('BEGIN IMMEDIATE')
        changed = db.execute("UPDATE jobs SET status=?,updated=?,error=? WHERE id=? AND status='running' AND owner=?",
                   ('failed' if error else 'complete',time.time(),error,ident,json.dumps(_owner(),sort_keys=True)))
        if changed.rowcount != 1:
            raise ValueError('Only the current claim owner can finish a running bundle')


def counts(run, stage=None):
    with connect(run) as db:
        if stage is None:rows=db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status')
        else:rows=db.execute('SELECT status,COUNT(*) FROM jobs WHERE stage=? GROUP BY status',(stage,))
        return dict(rows)
