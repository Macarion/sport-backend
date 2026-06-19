#!/usr/bin/env python
"""
Read UWB UART lines locally and upload raw lines to sport-backend.

Example:
  python uwb_serial_forwarder.py --port COM6 --baud 921600 --uid 20123456
"""
import argparse
import json
import sys
import time
from datetime import datetime
from typing import Iterable, List, Optional

import requests

try:
    import serial
except ImportError:
    serial = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward UWB serial raw lines to sport-backend.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM6.")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate.")
    parser.add_argument("--backend", default="http://localhost:8090", help="Backend base URL.")
    parser.add_argument("--uid", default="", help="Optional student UID for this run.")
    parser.add_argument("--session-id", default="", help="Optional run/session id.")
    parser.add_argument("--source", default="uwb_serial_forwarder", help="Source name stored in DB.")
    parser.add_argument("--batch-size", type=int, default=5, help="Upload this many lines per request.")
    parser.add_argument("--flush-ms", type=int, default=20, help="Upload pending lines after this many milliseconds.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print lines instead of uploading.")
    return parser.parse_args()


def iter_serial_lines(port: str, baud: int, read_timeout: float) -> Iterable[Optional[str]]:
    if serial is None:
        raise RuntimeError("Missing dependency: install pyserial first, for example `pip install pyserial`.")
    with serial.Serial(port=port, baudrate=baud, timeout=read_timeout) as ser:
        while True:
            raw = ser.readline()
            if not raw:
                yield None
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                yield text


def post_lines(session: requests.Session, args: argparse.Namespace, lines: List[str]) -> None:
    if args.dry_run:
        for line in lines:
            print(line)
        return

    url = args.backend.rstrip("/") + "/api/uwb/ingest/"
    payload = {
        "uid": args.uid,
        "session_id": args.session_id,
        "source": args.source,
        "lines": lines,
        "return_records": False,
    }
    response = session.post(url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    result = response.json()
    print(
        json.dumps(
            {
                "saved_count": result.get("saved_count", 0),
                "invalid_count": result.get("invalid_count", 0),
                "last_id": result.get("last_id", 0),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not args.session_id:
        args.session_id = datetime.now().strftime("uwb_%Y%m%d_%H%M%S")
    batch: List[str] = []
    session = requests.Session()
    batch_size = max(1, args.batch_size)
    flush_seconds = max(1, args.flush_ms) / 1000
    read_timeout = min(max(flush_seconds / 2, 0.01), 0.2)
    last_flush_at = time.monotonic()
    print(
        f"Forwarding {args.port} @ {args.baud} -> {args.backend.rstrip('/')}/api/uwb/ingest/ "
        f"(session_id={args.session_id}, batch_size={batch_size}, flush_ms={args.flush_ms})",
        flush=True,
    )

    while True:
        try:
            for line in iter_serial_lines(args.port, args.baud, read_timeout):
                now = time.monotonic()
                if line:
                    batch.append(line)
                if batch and (len(batch) >= batch_size or now - last_flush_at >= flush_seconds):
                    try:
                        post_lines(session, args, batch)
                    except requests.exceptions.ReadTimeout as exc:
                        print(
                            f"Forwarder timeout: {exc}. Dropping {len(batch)} pending lines because the server may have saved them.",
                            file=sys.stderr,
                            flush=True,
                        )
                        session.close()
                        session = requests.Session()
                    batch = []
                    last_flush_at = time.monotonic()
        except KeyboardInterrupt:
            if batch:
                post_lines(session, args, batch)
            print("Stopped.", flush=True)
            return 0
        except Exception as exc:
            print(f"Forwarder error: {exc}", file=sys.stderr, flush=True)
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
