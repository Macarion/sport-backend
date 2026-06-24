import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from django.http import StreamingHttpResponse
from django.db import connection, transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.renderers import BaseRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import UwbRawRecord, UwbSession, UwbSessionTagBinding, UwbTagBinding, UwbTrackPoint
from .uwb_parser import build_unified_record, parse_uwb_line


_UWB_TABLE_READY = False
_UWB_STREAM_SUBSCRIBERS: set[queue.Queue] = set()
_UWB_STREAM_LOCK = threading.Lock()
_UWB_STREAM_SENTINEL = object()


class ServerSentEventRenderer(BaseRenderer):
    media_type = "text/event-stream"
    format = "event-stream"
    charset = None

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return data


def ensure_uwb_table() -> None:
    global _UWB_TABLE_READY
    if _UWB_TABLE_READY:
        return
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uwb_raw_record (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                uid VARCHAR(64) NULL,
                session_id VARCHAR(64) NULL,
                source VARCHAR(64) NOT NULL DEFAULT 'serial_forwarder',
                raw_line LONGTEXT NOT NULL,
                record_type VARCHAR(20) NOT NULL DEFAULT 'UNKNOWN',
                tag_id INT NULL,
                cycle INT NULL,
                timestamp_us BIGINT NULL,
                timestamp_ms BIGINT NULL,
                parsed_json JSON NULL,
                unified_json JSON NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                INDEX idx_uwb_created_at (created_at),
                INDEX idx_uwb_tag_created (tag_id, created_at),
                INDEX idx_uwb_uid_created (uid, created_at),
                INDEX idx_uwb_session_created (session_id, created_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uwb_session (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL UNIQUE,
                name VARCHAR(128) NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                started_at DATETIME(6) NULL,
                ended_at DATETIME(6) NULL,
                anchors_json JSON NULL,
                remark VARCHAR(255) NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                INDEX idx_uwb_session_status (status),
                INDEX idx_uwb_session_started (started_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uwb_tag_binding (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                tag_id INT NOT NULL UNIQUE,
                uid VARCHAR(64) NOT NULL,
                username VARCHAR(128) NULL,
                runner_group VARCHAR(128) NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                INDEX idx_uwb_binding_uid (uid)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uwb_session_tag_binding (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                tag_id INT NOT NULL,
                uid VARCHAR(64) NOT NULL,
                username VARCHAR(128) NULL,
                runner_group VARCHAR(128) NULL,
                bound_at DATETIME(6) NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                UNIQUE KEY uq_uwb_session_tag (session_id, tag_id),
                INDEX idx_uwb_session_binding_uid (session_id, uid)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uwb_track_point (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                raw_record_id BIGINT NULL,
                session_id VARCHAR(64) NULL,
                tag_id INT NOT NULL,
                uid VARCHAR(64) NULL,
                username VARCHAR(128) NULL,
                runner_group VARCHAR(128) NULL,
                timestamp_ms BIGINT NULL,
                cycle INT NULL,
                x DOUBLE NULL,
                y DOUBLE NULL,
                z DOUBLE NULL,
                d0 DOUBLE NULL,
                d1 DOUBLE NULL,
                d2 DOUBLE NULL,
                d3 DOUBLE NULL,
                speed DOUBLE NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                UNIQUE KEY uq_uwb_track_raw (raw_record_id),
                INDEX idx_uwb_track_session_tag_id (session_id, tag_id, id),
                INDEX idx_uwb_track_session_tag_time (session_id, tag_id, timestamp_ms),
                INDEX idx_uwb_track_uid_time (uid, timestamp_ms)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
    ensure_column("uwb_session", "anchors_json", "JSON NULL")
    _UWB_TABLE_READY = True


def ensure_column(table_name: str, column_name: str, definition: str) -> None:
    with connection.cursor() as cursor:
        existing = {
            column.name
            for column in connection.introspection.get_table_description(cursor, table_name)
        }
    if column_name in existing:
        return
    with connection.cursor() as cursor:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def sse_encode(event: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def sse_comment(text: str) -> str:
    return f": {text}\n\n"


def publish_uwb_records(records: List[UwbRawRecord]) -> None:
    if not records:
        return
    payload = {
        "type": "records",
        "last_id": records[-1].id,
        "session_id": records[-1].session_id,
        "count": len(records),
        "records": [record_to_payload(record) for record in records],
    }
    with _UWB_STREAM_LOCK:
        subscribers = list(_UWB_STREAM_SUBSCRIBERS)
    for subscriber in subscribers:
        try:
            if subscriber.full():
                subscriber.get_nowait()
            subscriber.put_nowait(payload)
        except queue.Full:
            pass
        except Exception:
            remove_uwb_subscriber(subscriber)


def add_uwb_subscriber() -> queue.Queue:
    subscriber: queue.Queue = queue.Queue(maxsize=50)
    with _UWB_STREAM_LOCK:
        _UWB_STREAM_SUBSCRIBERS.add(subscriber)
    return subscriber


def remove_uwb_subscriber(subscriber: queue.Queue) -> None:
    with _UWB_STREAM_LOCK:
        _UWB_STREAM_SUBSCRIBERS.discard(subscriber)
    try:
        subscriber.put_nowait(_UWB_STREAM_SENTINEL)
    except Exception:
        pass


def payload_after_id(payload: Dict[str, Any], after_id: int) -> Optional[Dict[str, Any]]:
    records = []
    for record in payload.get("records") or []:
        try:
            record_id = int(record.get("id") or 0)
        except (TypeError, ValueError):
            record_id = 0
        if record_id > after_id:
            records.append(record)
    if not records:
        return None
    filtered = dict(payload)
    filtered["records"] = records
    filtered["count"] = len(records)
    filtered["last_id"] = records[-1].get("id")
    filtered["session_id"] = records[-1].get("session_id")
    return filtered


def records_payload(records: List[UwbRawRecord]) -> Dict[str, Any]:
    return {
        "type": "records",
        "last_id": records[-1].id,
        "session_id": records[-1].session_id,
        "count": len(records),
        "records": [record_to_payload(record) for record in records],
    }


def iter_uwb_events(subscriber: queue.Queue, after_id: int = 0):
    yield sse_comment("uwb stream connected")
    max_sent_id = max(0, after_id)
    if max_sent_id:
        records = list(UwbRawRecord.objects.filter(id__gt=max_sent_id).order_by("id")[:500])
        if records:
            payload = records_payload(records)
            yield sse_encode("uwb", payload)
            max_sent_id = records[-1].id
    last_heartbeat = time.monotonic()
    try:
        while True:
            try:
                payload = subscriber.get(timeout=10)
            except queue.Empty:
                yield sse_comment("heartbeat")
                last_heartbeat = time.monotonic()
                continue
            if payload is _UWB_STREAM_SENTINEL:
                break
            filtered_payload = payload_after_id(payload, max_sent_id)
            if not filtered_payload:
                continue
            payload = filtered_payload
            yield sse_encode("uwb", payload)
            try:
                max_sent_id = max(max_sent_id, int(payload.get("last_id") or max_sent_id))
            except (TypeError, ValueError):
                pass
            now = time.monotonic()
            if now - last_heartbeat > 10:
                yield sse_comment("heartbeat")
                last_heartbeat = now
    finally:
        remove_uwb_subscriber(subscriber)


def coerce_lines(payload: Dict[str, Any]) -> List[str]:
    if isinstance(payload.get("lines"), list):
        return [str(line).strip() for line in payload["lines"] if str(line).strip()]
    line = payload.get("raw_line") or payload.get("line") or payload.get("raw") or payload.get("data")
    return [str(line).strip()] if line else []


def latest_id(uid: Optional[str] = None) -> int:
    qs = UwbRawRecord.objects.all()
    if uid:
        qs = qs.filter(uid=str(uid))
    record = qs.order_by("-id").first()
    return record.id if record else 0


def normalize_session_id(value: Any) -> str:
    return str(value or "").strip()


def default_session_id(now=None) -> str:
    current = timezone.localtime(now or timezone.now())
    return current.strftime("uwb_%Y%m%d_%H%M%S")


def latest_active_session_id() -> str:
    record = (
        UwbRawRecord.objects.exclude(session_id__isnull=True)
        .exclude(session_id="")
        .order_by("-id")
        .only("session_id")
        .first()
    )
    if record and record.session_id:
        return record.session_id
    session = UwbSession.objects.filter(status="active").order_by("-updated_at", "-started_at", "-created_at").first()
    return session.session_id if session else ""


def resolve_session_id(value: Any = None, *, reuse_active: bool = True) -> str:
    session_id = normalize_session_id(value)
    if session_id:
        return session_id
    if reuse_active:
        active_id = latest_active_session_id()
        if active_id:
            return active_id
    return default_session_id()


def touch_session(
    session_id: str,
    status_value: str = "active",
    name: Optional[str] = None,
    anchors: Optional[List[Dict[str, float]]] = None,
) -> None:
    session_id = normalize_session_id(session_id)
    if not session_id:
        return
    session, created = UwbSession.objects.get_or_create(
        session_id=session_id,
        defaults={
            "name": name or session_id,
            "status": status_value,
            "started_at": timezone.now() if status_value == "active" else None,
            "ended_at": timezone.now() if status_value == "stopped" else None,
            "anchors_json": anchors,
        },
    )
    if created:
        return
    update_fields = ["status", "updated_at"]
    session.status = status_value
    if name:
        session.name = name
        update_fields.append("name")
    if anchors:
        session.anchors_json = anchors
        update_fields.append("anchors_json")
    if status_value == "active":
        if not session.started_at:
            session.started_at = timezone.now()
            update_fields.append("started_at")
        session.ended_at = None
        update_fields.append("ended_at")
    elif status_value == "stopped":
        session.ended_at = timezone.now()
        update_fields.append("ended_at")
    session.save(update_fields=sorted(set(update_fields)))


def session_to_dict(session: UwbSession, count: int = 0) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "name": session.name or session.session_id,
        "status": session.status,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "anchors": session.anchors_json,
        "record_count": count,
    }


def binding_to_dict(binding: UwbTagBinding) -> Dict[str, Any]:
    data = {
        "tagId": binding.tag_id,
        "uid": binding.uid,
        "username": binding.username or "",
        "group": binding.runner_group or "",
    }
    session_id = getattr(binding, "session_id", None)
    if session_id:
        data["session_id"] = session_id
    return data


def get_binding_for_tag(tag_id: Optional[int], session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if tag_id is None:
        return None
    if session_id:
        session_binding = UwbSessionTagBinding.objects.filter(session_id=session_id, tag_id=tag_id).first()
        if session_binding:
            return binding_to_dict(session_binding)
    binding = UwbTagBinding.objects.filter(tag_id=tag_id).first()
    return binding_to_dict(binding) if binding else None


def save_session_binding(session_id: str, binding: Dict[str, Any]) -> None:
    session_id = normalize_session_id(session_id)
    if not session_id or binding.get("tagId") is None:
        return
    UwbSessionTagBinding.objects.update_or_create(
        session_id=session_id,
        tag_id=binding["tagId"],
        defaults={
            "uid": binding["uid"],
            "username": binding.get("username") or "",
            "runner_group": binding.get("group") or "",
            "bound_at": timezone.now(),
        },
    )


def apply_binding_to_unified(unified: Dict[str, Any], binding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not binding:
        return unified
    tag_id = binding.get("tagId")
    tag_name = f"Tag{tag_id}" if tag_id is not None else unified.get("skeletal_point", {}).get("name", "")
    unified["uid"] = str(binding.get("uid") or unified.get("uid") or "")
    unified["username"] = binding.get("username") or unified.get("username") or ""
    unified["runnerGroup"] = binding.get("group") or unified.get("runnerGroup") or ""
    skeletal = dict(unified.get("skeletal_point") or {})
    skeletal["tagId"] = tag_id
    skeletal["name"] = tag_name
    unified["skeletal_point"] = skeletal
    uwb = dict(unified.get("uwb") or {})
    uwb["tagId"] = tag_id
    uwb["tagName"] = tag_name
    unified["uwb"] = uwb
    return unified


def distance_value(distances: Optional[List[Optional[float]]], index: int) -> Optional[float]:
    if not isinstance(distances, list) or index >= len(distances):
        return None
    value = distances[index]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def compute_track_speed(session_id: str, tag_id: int, position: Optional[Dict[str, float]], timestamp_ms: Optional[int]) -> Optional[float]:
    if not position or not timestamp_ms:
        return None
    previous = (
        UwbTrackPoint.objects.filter(session_id=session_id, tag_id=tag_id, x__isnull=False, y__isnull=False)
        .order_by("-id")
        .first()
    )
    if not previous or not previous.timestamp_ms:
        return None
    dt = max((int(timestamp_ms) - int(previous.timestamp_ms)) / 1000, 0)
    if dt <= 0:
        return None
    step = ((position["x"] - previous.x) ** 2 + (position["y"] - previous.y) ** 2) ** 0.5
    if step < 0 or step > 20:
        return None
    return step / dt


def save_track_point(
    record: UwbRawRecord,
    parsed: Dict[str, Any],
    unified: Dict[str, Any],
    session_id: str,
    binding: Optional[Dict[str, Any]],
    anchors: List[Dict[str, float]],
) -> Optional[UwbTrackPoint]:
    if parsed.get("type") != "RNG" or parsed.get("tagId") is None:
        return None
    distances = normalize_distances(parsed.get("distances"))
    position = parsed.get("position") if isinstance(parsed.get("position"), dict) else None
    if not position:
        position = solve_replay_position(anchors, distances)
    tag_id = int(parsed["tagId"])
    timestamp_ms = unified.get("timestamp") or record.timestamp_ms
    speed = compute_track_speed(session_id, tag_id, position, timestamp_ms)
    uid = str(binding.get("uid") if binding else unified.get("uid") or "") or None
    username = binding.get("username") if binding else unified.get("username") or ""
    runner_group = binding.get("group") if binding else unified.get("runnerGroup") or ""
    return UwbTrackPoint.objects.create(
        raw_record_id=record.id,
        session_id=session_id or None,
        tag_id=tag_id,
        uid=uid,
        username=username or None,
        runner_group=runner_group or None,
        timestamp_ms=timestamp_ms,
        cycle=parsed.get("cycle"),
        x=position.get("x") if position else None,
        y=position.get("y") if position else None,
        z=position.get("z") if position else None,
        d0=distance_value(distances, 0),
        d1=distance_value(distances, 1),
        d2=distance_value(distances, 2),
        d3=distance_value(distances, 3),
        speed=speed,
    )


def save_line(line: str, payload: Dict[str, Any]) -> Optional[UwbRawRecord]:
    parsed = parse_uwb_line(line)
    if not parsed:
        return None

    source = str(payload.get("source") or "serial_forwarder")
    uid = payload.get("uid")
    username = payload.get("username")
    session_id = resolve_session_id(payload.get("session_id"), reuse_active=True)
    raw_anchors = payload.get("anchors")
    anchors = normalize_anchor_list(raw_anchors)
    touch_session(session_id, "active", payload.get("session_name"), anchors if isinstance(raw_anchors, list) else None)
    binding = get_binding_for_tag(parsed.get("tagId"), session_id)
    unified = build_unified_record(parsed, line, uid=uid, username=username, source=source)
    unified = apply_binding_to_unified(unified, binding)
    if binding:
        uid = binding.get("uid")
        username = binding.get("username")

    record = UwbRawRecord.objects.create(
        uid=str(uid) if uid not in (None, "") else None,
        session_id=session_id or None,
        source=source,
        raw_line=line,
        record_type=str(parsed.get("type") or "UNKNOWN"),
        tag_id=parsed.get("tagId"),
        cycle=parsed.get("cycle"),
        timestamp_us=parsed.get("timestampUs"),
        timestamp_ms=unified.get("timestamp"),
        parsed_json=parsed,
        unified_json=unified,
    )
    save_track_point(record, parsed, unified, session_id, binding, anchors)
    return record


def fetch_bindings_for_tags(tag_ids: List[int], session_id: str) -> Dict[int, Dict[str, Any]]:
    bindings: Dict[int, Dict[str, Any]] = {}
    if session_id:
        for item in UwbSessionTagBinding.objects.filter(session_id=session_id, tag_id__in=tag_ids):
            bindings[item.tag_id] = binding_to_dict(item)
    missing = [tag_id for tag_id in tag_ids if tag_id not in bindings]
    if missing:
        for item in UwbTagBinding.objects.filter(tag_id__in=missing):
            bindings[item.tag_id] = binding_to_dict(item)
    return bindings


def make_track_point(
    record: UwbRawRecord,
    parsed: Dict[str, Any],
    unified: Dict[str, Any],
    session_id: str,
    binding: Optional[Dict[str, Any]],
    anchors: List[Dict[str, float]],
) -> Optional[UwbTrackPoint]:
    if parsed.get("type") != "RNG" or parsed.get("tagId") is None:
        return None
    distances = normalize_distances(parsed.get("distances"))
    position = parsed.get("position") if isinstance(parsed.get("position"), dict) else None
    if not position:
        position = solve_replay_position(anchors, distances)
    uid = str(binding.get("uid") if binding else unified.get("uid") or "") or None
    username = binding.get("username") if binding else unified.get("username") or ""
    runner_group = binding.get("group") if binding else unified.get("runnerGroup") or ""
    return UwbTrackPoint(
        raw_record_id=record.id,
        session_id=session_id or None,
        tag_id=int(parsed["tagId"]),
        uid=uid,
        username=username or None,
        runner_group=runner_group or None,
        timestamp_ms=unified.get("timestamp") or record.timestamp_ms,
        cycle=parsed.get("cycle"),
        x=position.get("x") if position else None,
        y=position.get("y") if position else None,
        z=position.get("z") if position else None,
        d0=distance_value(distances, 0),
        d1=distance_value(distances, 1),
        d2=distance_value(distances, 2),
        d3=distance_value(distances, 3),
        speed=None,
    )


def save_lines(lines: List[str], payload: Dict[str, Any]) -> tuple[List[UwbRawRecord], List[str]]:
    source = str(payload.get("source") or "serial_forwarder")
    base_uid = payload.get("uid")
    base_username = payload.get("username")
    session_id = resolve_session_id(payload.get("session_id"), reuse_active=True)
    raw_anchors = payload.get("anchors")
    anchors = normalize_anchor_list(raw_anchors)
    touch_session(session_id, "active", payload.get("session_name"), anchors if isinstance(raw_anchors, list) else None)

    parsed_items: List[Dict[str, Any]] = []
    invalid: List[str] = []
    for line in lines:
        parsed = parse_uwb_line(line)
        if parsed:
            parsed_items.append({"line": line, "parsed": parsed})
        else:
            invalid.append(line)
    if not parsed_items:
        return [], invalid

    tag_ids = sorted(
        {
            int(item["parsed"]["tagId"])
            for item in parsed_items
            if item["parsed"].get("tagId") is not None
        }
    )
    bindings = fetch_bindings_for_tags(tag_ids, session_id)
    prepared: List[Dict[str, Any]] = []
    raw_records: List[UwbRawRecord] = []
    for item in parsed_items:
        parsed = item["parsed"]
        binding = bindings.get(parsed.get("tagId"))
        uid = base_uid
        username = base_username
        unified = build_unified_record(parsed, item["line"], uid=uid, username=username, source=source)
        unified = apply_binding_to_unified(unified, binding)
        if binding:
            uid = binding.get("uid")
            username = binding.get("username")
        record = UwbRawRecord(
            uid=str(uid) if uid not in (None, "") else None,
            session_id=session_id or None,
            source=source,
            raw_line=item["line"],
            record_type=str(parsed.get("type") or "UNKNOWN"),
            tag_id=parsed.get("tagId"),
            cycle=parsed.get("cycle"),
            timestamp_us=parsed.get("timestampUs"),
            timestamp_ms=unified.get("timestamp"),
            parsed_json=parsed,
            unified_json=unified,
        )
        raw_records.append(record)
        prepared.append({"record": record, "parsed": parsed, "unified": unified, "binding": binding})

    with transaction.atomic():
        saved_records = list(UwbRawRecord.objects.bulk_create(raw_records, batch_size=500))

    if saved_records and any(record.id is None for record in saved_records):
        saved_records = list(reversed(list(UwbRawRecord.objects.filter(session_id=session_id).order_by("-id")[: len(saved_records)])))

    for index, record in enumerate(saved_records):
        prepared[index]["record"] = record

    publish_uwb_records(saved_records)

    track_points = [
        point
        for point in (
            make_track_point(item["record"], item["parsed"], item["unified"], session_id, item["binding"], anchors)
            for item in prepared
        )
        if point is not None
    ]
    if track_points:
        with transaction.atomic():
            UwbTrackPoint.objects.bulk_create(track_points, batch_size=500, ignore_conflicts=True)

    return saved_records, invalid


def process_direct_db_records(after_id: int, limit: int, payload: Dict[str, Any]) -> tuple[List[UwbRawRecord], int]:
    source = str(payload.get("source") or "direct_db")
    base_uid = payload.get("uid")
    base_username = payload.get("username")
    raw_anchors = payload.get("anchors")
    anchors = normalize_anchor_list(raw_anchors)
    session_id = normalize_session_id(payload.get("session_id"))

    qs = (
        UwbRawRecord.objects.filter(id__gt=after_id, unified_json__isnull=True)
        .exclude(record_type="INVALID")
        .order_by("id")
    )
    raw_records = list(qs[:limit])
    processed: List[UwbRawRecord] = []
    invalid_count = 0

    for record in raw_records:
        parsed = parse_uwb_line(record.raw_line)
        if not parsed:
            record.record_type = "INVALID"
            record.parsed_json = {"error": "parse_failed", "raw_line": record.raw_line}
            record.source = record.source or source
            record.save(update_fields=["record_type", "parsed_json", "source"])
            invalid_count += 1
            continue

        current_session_id = normalize_session_id(record.session_id) or session_id or resolve_session_id(None, reuse_active=True)
        touch_session(current_session_id, "active", None, anchors if isinstance(raw_anchors, list) else None)
        binding = get_binding_for_tag(parsed.get("tagId"), current_session_id)
        uid = record.uid or base_uid
        username = base_username
        unified = build_unified_record(parsed, record.raw_line, uid=uid, username=username, source=record.source or source)
        unified = apply_binding_to_unified(unified, binding)
        if binding:
            uid = binding.get("uid")
            username = binding.get("username")

        record.uid = str(uid) if uid not in (None, "") else None
        record.session_id = current_session_id or None
        record.source = record.source or source
        record.record_type = str(parsed.get("type") or "UNKNOWN")
        record.tag_id = parsed.get("tagId")
        record.cycle = parsed.get("cycle")
        record.timestamp_us = parsed.get("timestampUs")
        record.timestamp_ms = unified.get("timestamp")
        record.parsed_json = parsed
        record.unified_json = unified
        record.save(
            update_fields=[
                "uid",
                "session_id",
                "source",
                "record_type",
                "tag_id",
                "cycle",
                "timestamp_us",
                "timestamp_ms",
                "parsed_json",
                "unified_json",
            ]
        )
        save_track_point(record, parsed, unified, current_session_id, binding, anchors)
        processed.append(record)

    if processed:
        publish_uwb_records(processed)

    return processed, invalid_count


def record_to_payload(record: UwbRawRecord) -> Dict[str, Any]:
    data = dict(record.unified_json or {})
    data["id"] = record.id
    data["session_id"] = record.session_id
    data["raw"] = data.get("raw") or record.raw_line
    data["raw_line"] = record.raw_line
    data["created_at"] = record.created_at.isoformat() if record.created_at else None
    return data


def backfill_binding(binding: Dict[str, Any]) -> int:
    tag_id = binding.get("tagId")
    if tag_id is None:
        return 0
    updated = 0
    qs = UwbRawRecord.objects.filter(tag_id=tag_id).order_by("id")
    for record in qs.iterator(chunk_size=200):
        unified = dict(record.unified_json or {})
        record.unified_json = apply_binding_to_unified(unified, binding)
        record.uid = str(binding.get("uid") or record.uid or "")
        record.save(update_fields=["uid", "unified_json"])
        updated += 1
    UwbTrackPoint.objects.filter(tag_id=tag_id).update(
        uid=str(binding.get("uid") or ""),
        username=binding.get("username") or "",
        runner_group=binding.get("group") or "",
    )
    return updated


def backfill_session_binding(session_id: str, binding: Dict[str, Any]) -> int:
    session_id = normalize_session_id(session_id)
    tag_id = binding.get("tagId")
    if not session_id or tag_id is None:
        return 0
    save_session_binding(session_id, binding)
    updated = 0
    qs = UwbRawRecord.objects.filter(session_id=session_id, tag_id=tag_id).order_by("id")
    for record in qs.iterator(chunk_size=200):
        unified = dict(record.unified_json or {})
        record.unified_json = apply_binding_to_unified(unified, binding)
        record.uid = str(binding.get("uid") or record.uid or "")
        record.save(update_fields=["uid", "unified_json"])
        updated += 1
    UwbTrackPoint.objects.filter(session_id=session_id, tag_id=tag_id).update(
        uid=str(binding.get("uid") or ""),
        username=binding.get("username") or "",
        runner_group=binding.get("group") or "",
    )
    return updated


def normalize_anchor_list(raw_anchors: Any) -> List[Dict[str, float]]:
    if not isinstance(raw_anchors, list):
        raw_anchors = [
            {"id": "A0", "x": 0, "y": 0, "z": 0},
            {"id": "A1", "x": 20, "y": 0, "z": 0},
            {"id": "A2", "x": 20, "y": 10, "z": 0},
            {"id": "A3", "x": 0, "y": 10, "z": 0},
        ]
    anchors = []
    for index, item in enumerate(raw_anchors[:4]):
        try:
            anchors.append(
                {
                    "id": str(item.get("id") or f"A{index}"),
                    "x": float(item.get("x") or 0),
                    "y": float(item.get("y") or 0),
                    "z": float(item.get("z") or 0),
                }
            )
        except (AttributeError, TypeError, ValueError):
            anchors.append({"id": f"A{index}", "x": 0.0, "y": 0.0, "z": 0.0})
    while len(anchors) < 4:
        anchors.append({"id": f"A{len(anchors)}", "x": 0.0, "y": 0.0, "z": 0.0})
    return anchors


def normalize_distances(raw: Any) -> Optional[List[Optional[float]]]:
    if not isinstance(raw, list):
        return None
    values: List[Optional[float]] = []
    for item in raw[:4]:
        if item is None:
            values.append(None)
            continue
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(None)
    while len(values) < 4:
        values.append(None)
    return values


def solve_replay_position(anchors: List[Dict[str, float]], distances: Optional[List[Optional[float]]]) -> Optional[Dict[str, float]]:
    if not distances or sum(distance is not None for distance in distances) < 3:
        return None
    points = [{"x": a["x"], "y": a["y"], "z": a["z"]} for a in anchors]
    solved = solve_trilateration(points, distances, ["x", "y", "z"]) or solve_trilateration(points, distances, ["x", "y"])
    if not solved:
        return None
    if "z" not in solved:
        solved["z"] = sum(point["z"] for point in points) / len(points)
    return {"x": solved["x"], "y": solved["y"], "z": solved["z"]}


def track_point_distances(point: UwbTrackPoint) -> List[Optional[float]]:
    return [point.d0, point.d1, point.d2, point.d3]


def track_point_position(point: UwbTrackPoint, anchors: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if point.x is not None and point.y is not None:
        return {"x": point.x, "y": point.y, "z": point.z if point.z is not None else 0}
    return solve_replay_position(anchors, track_point_distances(point))


def make_replay_series_map(tag_ids: List[int], session_id: str = "") -> Dict[int, Dict[str, Any]]:
    bindings: Dict[int, Dict[str, Any]] = {}
    if session_id:
        bindings.update(
            {
                item.tag_id: binding_to_dict(item)
                for item in UwbSessionTagBinding.objects.filter(session_id=session_id, tag_id__in=tag_ids)
            }
        )
    for item in UwbTagBinding.objects.filter(tag_id__in=tag_ids):
        bindings.setdefault(item.tag_id, binding_to_dict(item))

    series_map: Dict[int, Dict[str, Any]] = {}
    for tag_id in tag_ids:
        binding = bindings.get(tag_id)
        series_map[tag_id] = {
            "tagId": tag_id,
            "tagName": f"Tag{tag_id}",
            "uid": binding.get("uid") if binding else "",
            "username": binding.get("username") if binding else "",
            "group": binding.get("group") if binding else "",
            "points": [],
        }
    return series_map


def append_track_point(series_map: Dict[int, Dict[str, Any]], flat_records: List[Dict[str, Any]], point: UwbTrackPoint, anchors: List[Dict[str, float]]) -> None:
    position = track_point_position(point, anchors)
    if not position:
        return
    tag_id = int(point.tag_id)
    replay_point = {
        **position,
        "at": int(point.timestamp_ms or 0),
        "recordId": point.id,
        "rawRecordId": point.raw_record_id,
        "frame": point.cycle,
        "distances": track_point_distances(point),
        "speed": point.speed,
    }
    if point.uid and not series_map[tag_id].get("uid"):
        series_map[tag_id]["uid"] = point.uid
    if point.username and not series_map[tag_id].get("username"):
        series_map[tag_id]["username"] = point.username
    if point.runner_group and not series_map[tag_id].get("group"):
        series_map[tag_id]["group"] = point.runner_group
    series_map[tag_id]["points"].append(replay_point)
    flat_records.append({"tagId": tag_id, **replay_point})


def solve_trilateration(points: List[Dict[str, float]], distances: List[Optional[float]], axes: List[str]) -> Optional[Dict[str, float]]:
    try:
        ref_index = next(index for index, distance in enumerate(distances) if distance is not None)
    except StopIteration:
        return None
    p0 = points[ref_index]
    r0 = distances[ref_index]
    if r0 is None:
        return None
    rows = []
    values = []
    for index, distance in enumerate(distances):
        if index == ref_index or distance is None:
            continue
        point = points[index]
        rows.append([2 * (point[axis] - p0[axis]) for axis in axes])
        values.append(r0 * r0 - distance * distance + squared_norm(point, axes) - squared_norm(p0, axes))
    if len(rows) < len(axes):
        return None
    normal_a = []
    normal_b = []
    for row in range(len(axes)):
        normal_a.append([sum(item[row] * item[col] for item in rows) for col in range(len(axes))])
        normal_b.append(sum(item[row] * values[index] for index, item in enumerate(rows)))
    solved = solve_linear_system(normal_a, normal_b)
    if not solved:
        return None
    return {axis: solved[index] for index, axis in enumerate(axes)}


def squared_norm(point: Dict[str, float], axes: List[str]) -> float:
    return sum(point[axis] * point[axis] for axis in axes)


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    n = len(vector)
    a = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-8:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        for item in range(col, n + 1):
            a[col][item] /= div
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for item in range(col, n + 1):
                a[row][item] -= factor * a[col][item]
    return [a[row][n] for row in range(n)]


class UwbStartView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        uid = request.query_params.get("uid") or request.query_params.get("username")
        reuse_active = str(request.query_params.get("reuse_active") or "").lower() in ("1", "true", "yes")
        session_id = resolve_session_id(request.query_params.get("session_id"), reuse_active=reuse_active)
        touch_session(session_id, "active", request.query_params.get("session_name"))
        return Response(
            {
                "status": "started",
                "uid": uid,
                "session_id": session_id,
                "last_id": latest_id(uid),
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )


class UwbStopView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        uid = request.query_params.get("uid") or request.query_params.get("username")
        session_id = normalize_session_id(request.query_params.get("session_id")) or latest_active_session_id()
        if session_id:
            touch_session(session_id, "stopped")
        return Response(
            {
                "status": "stopped",
                "uid": uid,
                "session_id": session_id,
                "last_id": latest_id(uid),
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )


class UwbIngestView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        ensure_uwb_table()
        payload = request.data if isinstance(request.data, dict) else {}
        lines = coerce_lines(payload)
        if not lines:
            return Response({"error": "missing raw_line or lines"}, status=status.HTTP_400_BAD_REQUEST)

        saved, invalid = save_lines(lines, payload)

        response_status = status.HTTP_201_CREATED if saved else status.HTTP_400_BAD_REQUEST
        response_data = {
            "saved_count": len(saved),
            "invalid_count": len(invalid),
            "last_id": saved[-1].id if saved else latest_id(payload.get("uid")),
            "session_id": saved[-1].session_id if saved else normalize_session_id(payload.get("session_id")),
            "invalid_lines": invalid[:20],
        }
        if payload.get("return_records") is True:
            response_data["records"] = [record_to_payload(record) for record in saved]
        return Response(response_data, status=response_status)


class UwbStreamView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    renderer_classes = [ServerSentEventRenderer]

    def get(self, request):
        ensure_uwb_table()
        try:
            after_id = int(request.query_params.get("after_id") or 0)
        except (TypeError, ValueError):
            after_id = 0
        subscriber = add_uwb_subscriber()
        response = StreamingHttpResponse(iter_uwb_events(subscriber, after_id), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response


class UwbFetchIncDataView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        uid = request.query_params.get("uid") or request.query_params.get("username")
        try:
            after_id = int(request.query_params.get("after_id") or 0)
        except (TypeError, ValueError):
            after_id = 0
        try:
            limit = int(request.query_params.get("limit") or 100)
        except (TypeError, ValueError):
            limit = 100
        limit = max(1, min(limit, 500))

        qs = UwbRawRecord.objects.all()
        if uid:
            qs = qs.filter(uid=str(uid))
        session_id = normalize_session_id(request.query_params.get("session_id"))
        if session_id:
            qs = qs.filter(session_id=session_id)

        if after_id > 0:
            records = list(qs.filter(id__gt=after_id).order_by("id")[:limit])
        else:
            records = list(reversed(list(qs.order_by("-id")[:limit])))
        last_id = records[-1].id if records else after_id
        return Response(
            {
                "uid": uid,
                "session_id": session_id,
                "after_id": after_id,
                "last_id": last_id,
                "count": len(records),
                "records": [record_to_payload(record) for record in records],
            },
            status=status.HTTP_200_OK,
        )


class UwbDirectDbFetchIncDataView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        uid = request.query_params.get("uid") or request.query_params.get("username")
        try:
            after_id = int(request.query_params.get("after_id") or 0)
        except (TypeError, ValueError):
            after_id = 0
        try:
            limit = int(request.query_params.get("limit") or 100)
        except (TypeError, ValueError):
            limit = 100
        limit = max(1, min(limit, 500))

        payload: Dict[str, Any] = {
            "uid": uid,
            "username": request.query_params.get("username"),
            "source": "direct_db",
            "session_id": request.query_params.get("session_id"),
        }
        anchors_text = request.query_params.get("anchors")
        if anchors_text:
            try:
                payload["anchors"] = json.loads(anchors_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                payload["anchors"] = None

        _, invalid_count = process_direct_db_records(after_id, limit, payload)

        qs = UwbRawRecord.objects.filter(id__gt=after_id, unified_json__isnull=False)
        if uid:
            qs = qs.filter(uid=str(uid))
        session_id = normalize_session_id(request.query_params.get("session_id"))
        if session_id:
            qs = qs.filter(session_id=session_id)

        if after_id > 0:
            records = list(qs.order_by("id")[:limit])
        else:
            records = list(reversed(list(qs.order_by("-id")[:limit])))
        last_id = records[-1].id if records else after_id
        return Response(
            {
                "uid": uid,
                "session_id": session_id,
                "after_id": after_id,
                "last_id": last_id,
                "count": len(records),
                "processed_invalid_count": invalid_count,
                "records": [record_to_payload(record) for record in records],
            },
            status=status.HTTP_200_OK,
        )


class UwbLatestView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        uid = request.query_params.get("uid") or request.query_params.get("username")
        qs = UwbRawRecord.objects.all()
        if uid:
            qs = qs.filter(uid=str(uid))
        record = qs.order_by("-id").first()
        if not record:
            return Response({"record": None, "last_id": 0}, status=status.HTTP_200_OK)
        return Response({"record": record_to_payload(record), "last_id": record.id}, status=status.HTTP_200_OK)


class UwbBindingView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        bindings = [binding_to_dict(item) for item in UwbTagBinding.objects.all().order_by("tag_id")]
        return Response({"bindings": bindings}, status=status.HTTP_200_OK)

    def post(self, request):
        ensure_uwb_table()
        payload = request.data if isinstance(request.data, dict) else {}
        session_id = normalize_session_id(payload.get("session_id")) or latest_active_session_id()
        raw_bindings = payload.get("bindings")
        if not isinstance(raw_bindings, list):
            return Response({"error": "bindings must be a list"}, status=status.HTTP_400_BAD_REQUEST)

        normalized: List[Dict[str, Any]] = []
        errors: List[str] = []
        seen = set()
        for index, item in enumerate(raw_bindings, start=1):
            if not isinstance(item, dict):
                errors.append(f"第 {index} 条绑定格式错误")
                continue
            try:
                tag_id = int(item.get("tagId"))
            except (TypeError, ValueError):
                errors.append(f"第 {index} 条缺少 TagID")
                continue
            uid = str(item.get("uid") or "").strip()
            if not uid:
                errors.append(f"Tag{tag_id} 缺少 UID")
                continue
            if tag_id in seen:
                errors.append(f"Tag{tag_id} 重复绑定")
                continue
            seen.add(tag_id)
            normalized.append(
                {
                    "tagId": tag_id,
                    "uid": uid,
                    "username": str(item.get("username") or "").strip(),
                    "group": str(item.get("group") or "").strip(),
                }
            )

        if errors:
            return Response({"error": "绑定保存失败", "errors": errors}, status=status.HTTP_400_BAD_REQUEST)

        UwbTagBinding.objects.exclude(tag_id__in=[item["tagId"] for item in normalized]).delete()
        if session_id:
            UwbSessionTagBinding.objects.filter(session_id=session_id).exclude(tag_id__in=[item["tagId"] for item in normalized]).delete()
        backfilled = 0
        for item in normalized:
            UwbTagBinding.objects.update_or_create(
                tag_id=item["tagId"],
                defaults={
                    "uid": item["uid"],
                    "username": item["username"],
                    "runner_group": item["group"],
                },
            )
            if session_id:
                backfilled += backfill_session_binding(session_id, item)
            else:
                backfilled += backfill_binding(item)

        bindings = [binding_to_dict(item) for item in UwbTagBinding.objects.all().order_by("tag_id")]
        return Response(
            {"bindings": bindings, "session_id": session_id, "backfilled_count": backfilled},
            status=status.HTTP_200_OK,
        )


class UwbSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ensure_uwb_table()
        counts: Dict[str, int] = {}
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, COUNT(*)
                FROM uwb_raw_record
                WHERE session_id IS NOT NULL AND session_id <> ''
                GROUP BY session_id
                """
            )
            for session_id, count in cursor.fetchall():
                counts[str(session_id)] = int(count)

        existing_ids = set(UwbSession.objects.values_list("session_id", flat=True))
        missing_ids = [session_id for session_id in counts.keys() if session_id not in existing_ids]
        for session_id in missing_ids:
            UwbSession.objects.create(session_id=session_id, name=session_id, status="unknown")

        sessions = [
            session_to_dict(session, counts.get(session.session_id, 0))
            for session in UwbSession.objects.all().order_by("-started_at", "-created_at")
        ]
        return Response({"sessions": sessions}, status=status.HTTP_200_OK)


class UwbReplayView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ensure_uwb_table()
        payload = request.data if isinstance(request.data, dict) else {}
        raw_tag_ids = payload.get("tag_ids") or payload.get("tagIds") or []
        if not isinstance(raw_tag_ids, list):
            return Response({"error": "tag_ids must be a list"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            tag_ids = [int(tag_id) for tag_id in raw_tag_ids]
        except (TypeError, ValueError):
            return Response({"error": "tag_ids must contain numbers"}, status=status.HTTP_400_BAD_REQUEST)
        tag_ids = sorted(set(tag_ids))
        if not tag_ids:
            return Response({"series": [], "count": 0}, status=status.HTTP_200_OK)

        anchors = normalize_anchor_list(payload.get("anchors"))
        try:
            limit = int(payload.get("limit") or 5000)
        except (TypeError, ValueError):
            limit = 5000
        limit = max(1, min(limit, 20000))

        session_id = str(payload.get("session_id") or "").strip()
        qs = UwbTrackPoint.objects.filter(tag_id__in=tag_ids)
        if session_id:
            qs = qs.filter(session_id=session_id)
        points = list(qs.order_by("id")[:limit])
        series_map = make_replay_series_map(tag_ids, session_id)
        flat_records: List[Dict[str, Any]] = []
        for point in points:
            append_track_point(series_map, flat_records, point, anchors)

        if not flat_records:
            raw_qs = UwbRawRecord.objects.filter(tag_id__in=tag_ids, record_type="RNG")
            if session_id:
                raw_qs = raw_qs.filter(session_id=session_id)
            records = list(raw_qs.order_by("id").only("id", "tag_id", "timestamp_ms", "unified_json")[:limit])
            for record in records:
                unified = dict(record.unified_json or {})
                uwb = unified.get("uwb") or {}
                distances = normalize_distances(uwb.get("distances"))
                position = solve_replay_position(anchors, distances)
                if not position:
                    continue
                tag_id = int(record.tag_id)
                point = {
                    **position,
                    "at": int(unified.get("timestamp") or record.timestamp_ms or 0),
                    "recordId": record.id,
                    "frame": unified.get("frame"),
                    "distances": distances,
                }
                series_map[tag_id]["points"].append(point)
                flat_records.append({"tagId": tag_id, **point})

        series = [series_map[tag_id] for tag_id in tag_ids]
        return Response(
            {
                "series": series,
                "count": len(flat_records),
                "anchors": anchors,
            },
            status=status.HTTP_200_OK,
        )


class UwbReplayWindowView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ensure_uwb_table()
        payload = request.data if isinstance(request.data, dict) else {}
        raw_tag_ids = payload.get("tag_ids") or payload.get("tagIds") or []
        if not isinstance(raw_tag_ids, list):
            return Response({"error": "tag_ids must be a list"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            tag_ids = sorted(set(int(tag_id) for tag_id in raw_tag_ids))
        except (TypeError, ValueError):
            return Response({"error": "tag_ids must contain numbers"}, status=status.HTTP_400_BAD_REQUEST)
        if not tag_ids:
            return Response({"series": [], "records": [], "has_more": False, "next_cursor_id": 0}, status=status.HTTP_200_OK)

        try:
            cursor_id = int(payload.get("cursor_id") or payload.get("cursorId") or 0)
        except (TypeError, ValueError):
            cursor_id = 0
        try:
            limit = int(payload.get("limit") or 1000)
        except (TypeError, ValueError):
            limit = 1000
        limit = max(1, min(limit, 3000))

        session_id = normalize_session_id(payload.get("session_id"))
        anchors = normalize_anchor_list(payload.get("anchors"))
        qs = UwbTrackPoint.objects.filter(tag_id__in=tag_ids, id__gt=cursor_id)
        if session_id:
            qs = qs.filter(session_id=session_id)
        points = list(qs.order_by("id")[: limit + 1])
        has_more = len(points) > limit
        points = points[:limit]

        series_map = make_replay_series_map(tag_ids, session_id)
        flat_records = []
        next_cursor_id = cursor_id
        for point in points:
            next_cursor_id = max(next_cursor_id, int(point.id))
            append_track_point(series_map, flat_records, point, anchors)

        if not flat_records and cursor_id == 0:
            raw_qs = UwbRawRecord.objects.filter(tag_id__in=tag_ids, record_type="RNG", id__gt=cursor_id)
            if session_id:
                raw_qs = raw_qs.filter(session_id=session_id)
            records = list(raw_qs.order_by("id").only("id", "tag_id", "timestamp_ms", "unified_json")[: limit + 1])
            has_more = len(records) > limit
            records = records[:limit]
            for record in records:
                next_cursor_id = max(next_cursor_id, int(record.id))
                unified = dict(record.unified_json or {})
                uwb = unified.get("uwb") or {}
                distances = normalize_distances(uwb.get("distances"))
                position = solve_replay_position(anchors, distances)
                if not position:
                    continue
                tag_id = int(record.tag_id)
                replay_point = {
                    **position,
                    "at": int(unified.get("timestamp") or record.timestamp_ms or 0),
                    "recordId": record.id,
                    "frame": unified.get("frame"),
                    "distances": distances,
                }
                series_map[tag_id]["points"].append(replay_point)
                flat_records.append({"tagId": tag_id, **replay_point})

        return Response(
            {
                "series": [series_map[tag_id] for tag_id in tag_ids],
                "records": flat_records,
                "count": len(flat_records),
                "next_cursor_id": next_cursor_id,
                "has_more": has_more,
                "session_id": session_id,
            },
            status=status.HTTP_200_OK,
        )
