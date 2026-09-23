"""
Tails logs/api_requests.log (JSON lines) and re-emits each entry as a
Combined Log Format line on stdout, for GoAccess to consume.

GoAccess can parse raw JSON logs directly via a custom log-format mapping,
but that mapping is fragile in practice (see e.g. allinurl/goaccess#2699,
where a user hit the same kind of date-parsing failures this project's
timestamp format would also risk). Converting to Combined Log Format --
GoAccess's most mature, best-tested input format -- avoids that risk
entirely, at the cost of this one small conversion step.

Run with: python3 src/scripts/tail_to_clf.py logs/api_requests.log
"""

import json
import sys
import time
from datetime import datetime


def to_clf_line(entry: dict) -> str:
    ts = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S")
    clf_time = ts.strftime("%d/%b/%Y:%H:%M:%S +0000")
    method = entry.get("method", "GET")
    path = entry.get("path", "/")
    status = entry.get("status", 200)
    return f'127.0.0.1 - - [{clf_time}] "{method} {path} HTTP/1.1" {status} 0'


def follow(path: str):
    with open(path) as f:
        f.seek(0, 2)  # start at end, like tail -f
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                print(to_clf_line(entry), flush=True)
            except (KeyError, ValueError):
                continue  # malformed/unexpected entry shape -- skip rather than crash the whole stream


if __name__ == "__main__":
    follow(sys.argv[1])
