"""Render a fixed review group under bounded manual pause, then restore training."""
import argparse
import json
import signal
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    args = parser.parse_args()
    if not 1 <= args.expected <= 100 or not 1 <= args.timeout <= 600:
        raise ValueError("Review bursts are bounded to 100 images and ten minutes")

    def call(path, data=None):
        request = urllib.request.Request(args.url + "/api/" + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(request, timeout=30))

    targets = [j for j in call("jobs") if j["group_id"] in args.group]
    if len(targets) != args.expected:
        raise ValueError("Expected review group is missing or outside recent job history")
    if any(j["status"] in ("failed", "cancelled", "cancelling") for j in targets):
        raise ValueError("Resolve failed or cancelled group jobs before starting a burst")
    if all(j["status"] == "completed" for j in targets):
        print("Review group is already complete")
        return
    ids = {j["id"] for j in targets}
    prior = call("status")["controls"]["paused"]

    def interrupted(*_):
        raise KeyboardInterrupt("Review burst interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        call("training/pause", {"paused": True})
        for attempt in range(10):
            queued = [j for j in call("jobs") if j["status"] == "queued"]
            try:
                call("queue/reorder", {"ids": [j["id"] for j in queued if j["id"] in ids]
                     + [j["id"] for j in queued if j["id"] not in ids]})
                break
            except urllib.error.HTTPError as error:
                if error.code != 422 or attempt == 9:
                    raise
                time.sleep(1)
        deadline, previous = time.monotonic() + args.timeout, -1
        while time.monotonic() < deadline:
            jobs = {j["id"]: j for j in call("jobs") if j["id"] in ids}
            for ident in ids - jobs.keys():
                jobs[ident] = call("jobs/" + ident)
            if any(j["status"] in ("failed", "cancelled", "cancelling") for j in jobs.values()):
                raise RuntimeError("A review render failed or was cancelled")
            done = sum(j["status"] == "completed" for j in jobs.values())
            if done // 8 != previous or done == args.expected:
                print(f"Completed {done} / {args.expected}", flush=True)
                previous = done // 8
            if done == args.expected:
                return
            time.sleep(2)
        raise TimeoutError("Review burst exceeded its deadline")
    finally:
        print("Restored training pause state:", call("training/pause", {"paused": prior}), flush=True)


if __name__ == "__main__":
    main()
