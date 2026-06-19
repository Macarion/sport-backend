#!/usr/bin/env python
"""
Read UWB serial lines and send raw data to sport-backend for parsing/storage.

This script intentionally does not parse UWB payloads. The backend endpoint
`/api/uwb/ingest/` parses raw lines, builds JSON, and stores database rows.

Example:
  python uwb_serial_to_backend.py --port COM24 --baud 921600 --backend http://localhost:8090 --uid 20123456
"""
import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from typing import List

import requests

try:
    import serial
except ImportError:
    serial = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read UWB serial data and send raw lines to sport-backend.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM24.")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate.")
    parser.add_argument("--backend", default="http://localhost:8090", help="Backend base URL.")
    parser.add_argument("--uid", default="", help="Optional default athlete/student UID.")
    parser.add_argument("--session-id", default="", help="Optional run/session id. Defaults to current datetime.")
    parser.add_argument("--source", default="uwb_serial_to_backend", help="Source name stored in DB.")
    parser.add_argument("--batch-size", type=int, default=5, help="Upload when this many raw lines are pending.")
    parser.add_argument("--flush-ms", type=int, default=10, help="Upload pending lines after this many milliseconds.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--queue-size", type=int, default=10000, help="Maximum raw lines buffered between serial and HTTP.")
    parser.add_argument("--dry-run", action="store_true", help="Print raw lines without uploading.")
    return parser.parse_args()


def make_session_id() -> str:
    return datetime.now().strftime("uwb_%Y%m%d_%H%M%S")


def serial_reader(args: argparse.Namespace, outbox: queue.Queue, stop_event: threading.Event, stats: dict) -> None:
    if serial is None:
        raise RuntimeError("Missing dependency: install pyserial first, for example `pip install pyserial`.")

    with serial.Serial(port=args.port, baudrate=args.baud, timeout=0.02) as ser:
        while not stop_event.is_set():
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            stats["read_count"] += 1
            try:
                outbox.put(line, timeout=0.05)
            except queue.Full:
                stats["drop_count"] += 1


def post_lines(http: requests.Session, args: argparse.Namespace, lines: List[str]) -> dict:
    if args.dry_run:
        for line in lines:
            print(line, flush=True)
        return {"saved_count": 0, "invalid_count": 0, "last_id": 0, "dry_run": True}

    payload = {
        "uid": args.uid,
        "session_id": args.session_id,
        "source": args.source,
        "lines": lines,
        "return_records": False,
    }
    url = args.backend.rstrip("/") + "/api/uwb/ingest/"
    response = http.post(url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    return response.json()


def uploader(args: argparse.Namespace, inbox: queue.Queue, stop_event: threading.Event, stats: dict) -> None:
    http = requests.Session()
    batch: List[str] = []
    batch_size = max(1, args.batch_size)
    flush_seconds = max(1, args.flush_ms) / 1000.0
    last_flush_at = time.monotonic()

    while not stop_event.is_set() or not inbox.empty() or batch:
        timeout = max(0.001, flush_seconds / 2)
        try:
            line = inbox.get(timeout=timeout)
            batch.append(line)
        except queue.Empty:
            pass

        now = time.monotonic()
        should_flush = batch and (len(batch) >= batch_size or now - last_flush_at >= flush_seconds or stop_event.is_set())
        if not should_flush:
            continue

        current = batch
        batch = []
        try:
            result = post_lines(http, args, current)
            stats["upload_count"] += len(current)
            print(
                json.dumps(
                    {
                        "sent_count": len(current),
                        "saved_count": result.get("saved_count", 0),
                        "invalid_count": result.get("invalid_count", 0),
                        "last_id": result.get("last_id", 0),
                        "queued": inbox.qsize(),
                        "dropped": stats["drop_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except requests.exceptions.ReadTimeout as exc:
            stats["timeout_count"] += 1
            print(
                f"Upload timeout: {exc}. The backend may still have saved this batch; continuing.",
                file=sys.stderr,
                flush=True,
            )
            http.close()
            http = requests.Session()
        except Exception as exc:
            stats["error_count"] += 1
            print(f"Upload error: {exc}. Requeueing {len(current)} lines.", file=sys.stderr, flush=True)
            for line in reversed(current):
                try:
                    inbox.put_nowait(line)
                except queue.Full:
                    stats["drop_count"] += 1
            time.sleep(0.2)
        finally:
            last_flush_at = time.monotonic()

    http.close()


def main() -> int:
    args = parse_args()
    if not args.session_id:
        args.session_id = make_session_id()

    stats = {
        "read_count": 0,
        "upload_count": 0,
        "drop_count": 0,
        "timeout_count": 0,
        "error_count": 0,
    }
    outbox: queue.Queue = queue.Queue(maxsize=max(100, args.queue_size))
    stop_event = threading.Event()

    print(
        f"Reading {args.port} @ {args.baud}, forwarding raw lines to "
        f"{args.backend.rstrip()}/api/uwb/ingest/ "
        f"(session_id={args.session_id}, batch_size={args.batch_size}, flush_ms={args.flush_ms})",
        flush=True,
    )

    reader = threading.Thread(target=serial_reader, args=(args, outbox, stop_event, stats), daemon=True)
    writer = threading.Thread(target=uploader, args=(args, outbox, stop_event, stats), daemon=True)
    reader.start()
    writer.start()

    try:
        while reader.is_alive() and writer.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        writer.join(timeout=max(2.0, args.timeout + 1.0))

    print(json.dumps({"status": "stopped", **stats}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
