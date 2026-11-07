"""Conservative local Linux process identity; unknown owners are never dead proof."""
from __future__ import annotations

import os
from pathlib import Path
import re
import socket
import uuid


def valid_identity(identity) -> bool:
    if (not isinstance(identity, dict) or set(identity) != {'host', 'boot', 'pid', 'start'}
            or type(identity['pid']) is not int or identity['pid'] <= 0
            or any(not isinstance(identity[key], str) or not identity[key]
                   for key in ('host', 'boot', 'start'))
            or re.fullmatch(r'[0-9]+', identity['start']) is None):
        return False
    try:
        return str(uuid.UUID(identity['boot'])) == identity['boot']
    except ValueError:
        return False


def _stat(pid):
    # comm may contain spaces and ')'; fields after its final ')' start at state.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    state, start = fields[0], fields[19]
    if re.fullmatch(r'[0-9]+', start) is None:
        raise ValueError('invalid process start stamp')
    return state, start


def capture_process_identity(pid=None):
    """Return exact host/boot/PID/start proof, or None when it cannot be read."""
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid <= 0:
        return None
    try:
        state, start = _stat(pid)
        identity = dict(host=socket.gethostname(),
                        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        pid=pid, start=start)
        # An owned child may already be a zombie before its launcher reads
        # /proc. Preserve that exact identity; liveness classifies it as dead.
        return identity if valid_identity(identity) else None
    except (OSError, ValueError, IndexError):
        return None


def process_liveness(identity):
    """Return 'live', 'dead', or 'unknown', without signaling any process.

    Only strict local metadata can establish death. A foreign host, corrupt
    record, unreadable /proc or unsupported environment remains unknown.
    """
    if not valid_identity(identity):
        return 'unknown'
    current = capture_process_identity()
    if current is None or identity['host'] != current['host']:
        return 'unknown'
    if identity['boot'] != current['boot']:
        return 'dead'
    try:
        state, start = _stat(identity['pid'])
        return 'live' if start == identity['start'] and state not in ('Z', 'X') else 'dead'
    except FileNotFoundError:
        return 'dead'
    except (OSError, ValueError, IndexError):
        return 'unknown'
