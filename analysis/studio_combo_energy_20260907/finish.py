"""Complete reporting as soon as both persistent study services finish."""
import json
import time
from common import WORK


def main():
    while True:
        states = []
        for filename in ('renders.json', 'measurements.json'):
            path = WORK / filename
            states.append(json.loads(path.read_text())['status'] if path.exists() else 'pending')
        if all(s == 'complete' for s in states):
            break
        if any(s in ('complete_with_errors', 'interrupted_cuda_error') for s in states):
            raise RuntimeError('Study needs error review: ' + repr(states))
        time.sleep(10)
    from summarize import main as summarize
    from audit import main as audit
    summarize()
    audit()


if __name__ == '__main__':
    main()
