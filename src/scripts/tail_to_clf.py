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
import os
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


def _open_at_start(path: str):
    """A fresh (re)open, positioned at the beginning -- see follow()'s own
    comment on why this script never seeks to end."""
    f = open(path)
    return f, os.fstat(f.fileno()).st_ino


def follow(path: str):
    # No seek-to-end: GoAccess's --real-time-html needs at least one line on
    # its very first read or it exits with "No input data was provided nor
    # there's data to restore" rather than waiting -- confirmed by testing,
    # this crash-looped log-viewer every time it restarted with no traffic
    # since its last start. Replaying existing history from the beginning
    # avoids that cold start and is a better default anyway (a fresh
    # dashboard showing nothing until new traffic arrives isn't more correct,
    # just emptier).
    f, inode = _open_at_start(path)
    try:
        while True:
            line = f.readline()
            if not line:
                # api_requests.log is a RotatingFileHandler (10MB cap, see
                # logging_config.py) -- under real load it rotates (renamed
                # to .log.1, a fresh empty file takes its place) well within
                # a long-running tail. Python's open() follows the INODE,
                # not the path, so once that happens this handle just keeps
                # reading the old, now-disconnected file forever: EOF every
                # time, silently, no error -- confirmed directly, a live
                # load test rotated the log 4 times in under 20 minutes and
                # the dashboard went stuck at zero throughput with no sign
                # why. Comparing the path's CURRENT inode to the one this
                # handle was opened with detects exactly that: a changed
                # inode means rotation happened, so reopen fresh (from the
                # start, same reasoning as the cold-start case above) rather
                # than silently going stale.
                try:
                    current_inode = os.stat(path).st_ino
                except FileNotFoundError:
                    time.sleep(0.5)
                    continue
                if current_inode != inode:
                    f.close()
                    f, inode = _open_at_start(path)
                    continue
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
    finally:
        f.close()


if __name__ == "__main__":
    follow(sys.argv[1])
