import re
import time
from typing import Any, Dict, List, Optional


DEFAULT_ANCHORS = [
    {"type": "anchor", "id": "A0", "x": 0, "y": 0, "z": 0},
    {"type": "anchor", "id": "A1", "x": 20, "y": 0, "z": 0},
    {"type": "anchor", "id": "A2", "x": 20, "y": 10, "z": 0},
    {"type": "anchor", "id": "A3", "x": 0, "y": 10, "z": 0},
]


def parse_uwb_line(line: str) -> Optional[Dict[str, Any]]:
    text = (line or "").strip()
    if not text:
        return None

    rng = re.match(r"^RNG,Tag(\d+),(\d+),(\d+),(.+)$", text, re.I)
    if rng:
        distances = [_normalize_distance(value) for value in rng.group(4).split(",")[:4]]
        while len(distances) < 4:
            distances.append(None)
        return {
            "type": "RNG",
            "tagId": int(rng.group(1)),
            "cycle": int(rng.group(2)),
            "timestampUs": int(rng.group(3)),
            "distances": distances,
        }

    motion = re.match(r"^MOT,Tag(\d+),(\d+),(\d+),(.+)$", text, re.I)
    if motion:
        payload = motion.group(4)
        return {
            "type": "MOT",
            "tagId": int(motion.group(1)),
            "cycle": int(motion.group(2)),
            "timestampUs": int(motion.group(3)),
            "motionPayload": payload,
            "motionData": parse_motion_payload(payload),
        }

    return None


def build_unified_record(
    parsed: Dict[str, Any],
    raw_line: str,
    uid: Optional[str] = None,
    username: Optional[str] = None,
    source: str = "serial_forwarder",
) -> Dict[str, Any]:
    tag_id = parsed.get("tagId")
    tag_name = f"Tag{tag_id}" if tag_id is not None else ""
    timestamp_ms = _timestamp_us_to_ms(parsed.get("timestampUs")) or int(time.time() * 1000)
    distances = parsed.get("distances") if isinstance(parsed.get("distances"), list) else None
    motion_data = parsed.get("motionData") if isinstance(parsed.get("motionData"), dict) else None

    skeletal_point = {
        "type": "uwb_tag" if parsed.get("type") == "RNG" else "imu_motion",
        "tagId": tag_id,
        "name": tag_name,
        "position": parsed.get("position"),
        "distances": distances_to_object(distances),
        "imu": motion_data,
    }

    return {
        "uid": str(uid if uid not in (None, "") else tag_id if tag_id is not None else tag_name),
        "username": username or "",
        "runnerGroup": "",
        "frame": int(parsed.get("cycle") or parsed.get("frame") or 0),
        "score": None,
        "timestamp": timestamp_ms,
        "painting": build_painting_payload(distances, parsed.get("position")),
        "angle": extract_angle(parsed),
        "skeletal_point": skeletal_point,
        "uwb": {
            "type": parsed.get("type"),
            "tagId": tag_id,
            "tagName": tag_name,
            "cycle": int(parsed.get("cycle") or parsed.get("frame") or 0),
            "timestampUs": int(parsed.get("timestampUs") or 0),
            "distances": distances,
            "position": parsed.get("position"),
            "imu": motion_data,
        },
        "source": source,
        "raw": raw_line,
    }


def parse_motion_payload(payload: str) -> Dict[str, Optional[float]]:
    parts = str(payload).split(",")

    def as_number(index: int) -> Optional[float]:
        if index >= len(parts):
            return None
        try:
            return float(parts[index])
        except (TypeError, ValueError):
            return None

    def as_flag(index: int) -> Optional[int]:
        if index >= len(parts):
            return None
        try:
            return int(parts[index], 16)
        except (TypeError, ValueError):
            return None

    return {
        "dtMs": as_number(0),
        "sampleCount": as_number(1),
        "flags": as_flag(2),
        "ax": as_number(3),
        "ay": as_number(4),
        "az": as_number(5),
        "gx": as_number(6),
        "gy": as_number(7),
        "gz": as_number(8),
        "dvx": as_number(9),
        "dvy": as_number(10),
        "dvz": as_number(11),
        "dax": as_number(12),
        "day": as_number(13),
        "daz": as_number(14),
        "motion": as_number(15),
    }


def distances_to_object(distances: Optional[List[Optional[float]]]) -> Optional[Dict[str, Optional[float]]]:
    if not isinstance(distances, list):
        return None
    values = distances[:4]
    while len(values) < 4:
        values.append(None)
    return {"A0": values[0], "A1": values[1], "A2": values[2], "A3": values[3]}


def build_painting_payload(
    distances: Optional[List[Optional[float]]],
    position: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ranges = []
    if isinstance(distances, list):
        ranges = [
            {"type": "range", "anchor": f"A{index}", "distance": distance}
            for index, distance in enumerate(distances[:4])
        ]
    points = []
    if isinstance(position, dict):
        points.append({"type": "point", "role": "tag", **position})
    return [*DEFAULT_ANCHORS, *ranges, *points]


def extract_angle(parsed: Dict[str, Any]) -> Optional[float]:
    if parsed.get("angle") is not None:
        return _to_float(parsed.get("angle"))
    motion_data = parsed.get("motionData")
    if isinstance(motion_data, dict):
        if motion_data.get("gz") is not None:
            return _to_float(motion_data.get("gz"))
        if motion_data.get("yaw") is not None:
            return _to_float(motion_data.get("yaw"))
    return None


def _normalize_distance(value: Any) -> Optional[float]:
    if value is None or str(value).strip().upper() == "NA":
        return None
    distance = _to_float(value)
    if distance is None:
        return None
    return distance / 1000 if distance > 100 else distance


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _timestamp_us_to_ms(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return round(number / 1000) if number > 0 else None
