"""Required helpers copied from the existing Web algorithm; source hashes in docs/jump-source-manifest.json.
Unused standalone CLI/model loaders are intentionally omitted.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from aruco_measure import REQUIRED_ARUCO_IDS, ArucoPlane, draw_aruco_debug, transform_points
from full_pipeline_aruco import expand_box, save_segmentation_outputs

@dataclass
class FramePacket:
    index: int
    timestamp: float
    image: np.ndarray


@dataclass
class EncodedFrame:
    index: int
    timestamp: float
    jpeg_bytes: bytes


@dataclass
class PersonDetection:
    bbox: List[int]
    confidence: float
    bottom_center_image: np.ndarray
    bottom_center_world: Optional[np.ndarray] = None

    @property
    def y2(self) -> int:
        return int(self.bbox[3])


@dataclass
class JumpFrameSample:
    packet: FramePacket
    detection: PersonDetection
    encoded_frame: EncodedFrame


class CalibrationStabilityTracker:
    def __init__(self, stable_seconds: float, jitter_px: float):
        self.stable_seconds = stable_seconds
        self.jitter_px = jitter_px
        self.started_at = None
        self.previous_anchors = None
        self.latest_detections = {}

    def reset(self):
        self.started_at = None
        self.previous_anchors = None
        self.latest_detections = {}

    def update(self, detections: Dict[int, object], timestamp: float) -> bool:
        self.latest_detections = detections
        if any(marker_id not in detections for marker_id in REQUIRED_ARUCO_IDS):
            self.reset()
            self.latest_detections = detections
            return False

        anchors = np.array(
            [detections[marker_id].anchor_point for marker_id in REQUIRED_ARUCO_IDS],
            dtype=np.float32,
        )
        if self.previous_anchors is None:
            self.started_at = timestamp
            self.previous_anchors = anchors
            return self.stable_seconds <= 0

        if not anchors_are_stable(self.previous_anchors, anchors, self.jitter_px):
            self.started_at = timestamp

        self.previous_anchors = anchors
        return (
            self.started_at is not None
            and timestamp - self.started_at >= self.stable_seconds
        )

    def elapsed(self, timestamp: float) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, timestamp - self.started_at)


def anchors_are_stable(previous: np.ndarray, current: np.ndarray, jitter_px: float):
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.shape != current.shape:
        return False
    distances = np.linalg.norm(previous - current, axis=1)
    return bool(np.max(distances) <= jitter_px)


def point_in_ready_zone(
    point: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    x_value, y_value = float(point[0]), float(point[1])
    return x_min <= x_value <= x_max and y_min <= y_value <= y_max


def bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def bbox_center(box: Sequence[int]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def match_locked_person(
    detections: Sequence[PersonDetection],
    previous_bbox: Optional[Sequence[int]],
    max_center_distance_px: float,
) -> Optional[PersonDetection]:
    if not detections:
        return None
    if previous_bbox is None:
        return max(
            detections,
            key=lambda detection: (
                detection.bbox[2] - detection.bbox[0]
            )
            * (detection.bbox[3] - detection.bbox[1]),
        )

    previous_center = bbox_center(previous_bbox)
    candidates = []
    for detection in detections:
        center_distance = float(
            np.linalg.norm(bbox_center(detection.bbox) - previous_center)
        )
        iou = bbox_iou(previous_bbox, detection.bbox)
        if iou >= 0.05 or center_distance <= max_center_distance_px:
            score = 3.0 * iou - center_distance / max(max_center_distance_px, 1.0)
            candidates.append((score, detection))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def select_ready_person(
    detections: Sequence[PersonDetection],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[PersonDetection]:
    candidates = [
        detection
        for detection in detections
        if detection.bottom_center_world is not None
        and point_in_ready_zone(
            detection.bottom_center_world,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda detection: abs(float(detection.bottom_center_world[0]))
        + 0.2 * abs(float(detection.bottom_center_world[1])),
    )


def detect_takeoff(
    recent_y2_values: Sequence[float],
    baseline_y2: float,
    frame_height: int,
    min_pixels: float,
    image_ratio: float,
) -> bool:
    threshold = max(min_pixels, frame_height * image_ratio)
    return len(recent_y2_values) >= 2 and all(
        baseline_y2 - float(y2) >= threshold for y2 in list(recent_y2_values)[-2:]
    )


def detect_people(yolo_model, image: np.ndarray, plane: Optional[ArucoPlane]):
    results = yolo_model.predict(source=image, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0:
                continue

            # Keep bbox conversion identical to the validated offline detector.
            x1, y1, x2, y2 = [int(float(value)) for value in box.xyxy[0]]
            if x2 <= x1 or y2 <= y1:
                continue

            bottom_center = np.array([(x1 + x2) / 2.0, float(y2)], dtype=np.float32)
            world_point = None
            if plane is not None:
                world_point = transform_points(
                    bottom_center.reshape(1, 2),
                    plane.image_to_world,
                )[0]
            detections.append(
                PersonDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(box.conf[0]),
                    bottom_center_image=bottom_center,
                    bottom_center_world=world_point,
                )
            )
    return detections
