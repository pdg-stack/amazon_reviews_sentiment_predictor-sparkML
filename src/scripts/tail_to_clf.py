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

--once reads whatever's currently in the file, top to bottom, and exits
instead of waiting for new lines -- used by the cloud deployment's
periodic-regeneration loop (see docker-compose.cloud.yml) instead of
follow()'s continuous tail. See that file's comment for why: GoAccess's
--real-time-html mechanism only ever rewrites its output file in response
to a connected WebSocket client, which nginx's static-file serving of
that output (see docker/nginx.conf's /logs/ comment) never provides --
confirmed directly, the file sat byte-for-byte unchanged through an entire
live load test's worth of real traffic. A one-shot full-file batch
re-render, re-run on a timer, doesn't depend on a WebSocket at all.
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
    # GoAccess's built-in --log-format=COMBINED is specifically
    # `%h %^[%d:%t %^] "%r" %s %b "%R" "%u"` (see /etc/goaccess/goaccess.conf
    # in the image) -- it needs the trailing quoted referrer + user-agent
    # fields too, even if there's nothing real to put in them, or its parser
    # rejects the whole line ("Token for '%b' specifier is NULL") -- this
    # project doesn't track either value, so "-" (the standard CLF "absent
    # field" placeholder) stands in for both.
    return f'127.0.0.1 - - [{clf_time}] "{method} {path} HTTP/1.1" {status} 0 "-" "-"'


def _open_at_start(path: str):
    """A fresh (re)open, positioned at the beginning -- see follow()'s own
    comment on why this script never seeks to end."""
    f = open(path)
    return f, os.fstat(f.fileno()).st_ino


def dump_once(path: str) -> None:
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                print(to_clf_line(entry), flush=True)
            except (KeyError, ValueError):
                continue


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
    args = sys.argv[1:]
    if args and args[0] == "--once":
        dump_once(args[1])
    else:
        follow(args[0])
