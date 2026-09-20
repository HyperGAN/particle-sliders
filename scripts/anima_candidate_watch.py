"""Bound the cost of a temporary, disposable GPU migration probe."""
import json
from pathlib import Path
import sys
import time

from anima_runpod import api

path = Path(sys.argv[1])
while True:
    state = json.loads(path.read_text())
    if state.get("promoted") or state.get("terminated"):
        break
    if time.time() >= state["deadline"]:
        # No Studio requests or training runs are admitted to this candidate.
        # Its inputs are copies of verified local artifacts; probe outputs are
        # disposable. Promotion transfers ownership to the full backup watcher.
        assert state["probe_only"]
        api("DELETE", "pods/" + state["id"])
        state.update(terminated=True, reason="Migration probe deadline", terminated_at=time.time())
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)
        break
    time.sleep(30)
