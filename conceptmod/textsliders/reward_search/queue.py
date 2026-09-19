"""Atomic GPU work claims; each matched seed bundle stays on one GPU."""
import json
import sqlite3
import time
from pathlib import Path


def connect(run):
    db=sqlite3.connect(Path(run)/'queue.sqlite',timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, stage TEXT, payload TEXT, status TEXT, gpu INTEGER, updated REAL, error TEXT)')
    return db


def add(run, stage, jobs):
    with connect(run) as db:
        for job in jobs:
            ident=f"{stage}/{job['family']['family']}/{job['seed']}"
            payload=json.dumps(job,sort_keys=True)
            previous=db.execute('SELECT payload FROM jobs WHERE id=?',(ident,)).fetchone()
            if previous and previous[0]!=payload:
                raise ValueError('Queued experiment changed: '+ident)
            db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?)',
                       (ident,stage,payload,'pending',None,time.time(),None))


def claim(run, gpu):
    with connect(run) as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT id,stage,payload FROM jobs WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone()
        if row is None:return None
        db.execute("UPDATE jobs SET status='running',gpu=?,updated=? WHERE id=?",(gpu,time.time(),row[0]))
        return dict(id=row[0],stage=row[1],payload=json.loads(row[2]))


def finish(run, ident, error=None):
    with connect(run) as db:
        db.execute('UPDATE jobs SET status=?,updated=?,error=? WHERE id=?',
                   ('failed' if error else 'complete',time.time(),error,ident))


def counts(run, stage=None):
    with connect(run) as db:
        if stage is None:rows=db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status')
        else:rows=db.execute('SELECT status,COUNT(*) FROM jobs WHERE stage=? GROUP BY status',(stage,))
        return dict(rows)
