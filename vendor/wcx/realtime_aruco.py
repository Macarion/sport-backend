"""
Realtime fixed-camera ArUco standing long-jump measurement.

Default camera mode:
    python .\wcx\realtime_aruco.py

Replay an existing video through the realtime state machine:
    python .\wcx\realtime_aruco.py --video-file test5.mp4
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WCX_DIR = Path(__file__).resolve().parent
ARUCOTEST_DIR = WCX_DIR / "arucotest"
SAM2_PROJECT_ROOT = WCX_DIR / "sam2"
REALTIME_OUTPUT_ROOT = ARUCOTEST_DIR / "output" / "realtime"
DEFAULT_YOLO_MODEL = WCX_DIR / "jump2test" / "yolov8n.pt"
DEFAULT_SAM2_CHECKPOINT = (
    SAM2_PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_KEYPOINT_REPO_ROOT = Path(r"D:\Python_Project\all-keypoints-jump-broadcast-main")
DEFAULT_KEYPOINT_WEIGHTS_PATH = (
    DEFAULT_KEYPOINT_REPO_ROOT
    / "transformer"
    / "pretrained_weights"
    / "jump_broadcast_head_angle_2L.pth.tar"
)
DEFAULT_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)

if str(ARUCOTEST_DIR) not in sys.path:
    sys.path.insert(0, str(ARUCOTEST_DIR))

from aruco_measure import (  # noqa: E402
    REQUIRED_ARUCO_IDS,
    ArucoPlane,
    build_aruco_plane,
    detect_aruco_markers,
    draw_aruco_debug,
    draw_world_reference,
    measure_jump_from_files,
    transform_points,
)
from full_pipeline_aruco import (  # noqa: E402
    configure_console_encoding,
    ensure_sam2_import_path,
    expand_box,
    normalize_sam2_config,
    save_segmentation_outputs,
)
from pipeline_aruco import write_result_file  # noqa: E402


configure_console_encoding()


class RealtimeState(str, Enum):
    LOADING_MODELS = "LOADING_MODELS"
    CALIBRATING = "CALIBRATING"
    WAITING_READY = "WAITING_READY"
    ARMED = "ARMED"
    IN_JUMP = "IN_JUMP"
    PROCESSING = "PROCESSING"
    RESULT = "RESULT"
    FAILED = "FAILED"


@dataclass
class FramePacket:
    index: int
    timestamp: float
    image: np.ndarray


def rotate_frame(image: np.ndarray, rotate_mode: str) -> np.ndarray:
    if rotate_mode == "none":
        return image
    if rotate_mode == "clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotate_mode == "counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_mode == "180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"Unsupported rotate mode: {rotate_mode}")


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


@dataclass
class ModelBundle:
    yolo_model: object
    sam2_predictor: object
    keypoint_infer: object
    torch_module: object


@dataclass
class ProcessingResult:
    score_cm: float
    visualization_path: Path
    result_path: Path
    json_path: Path


@dataclass
class AttemptContext:
    output_dir: Path
    calibration_frame: Optional[np.ndarray] = None
    calibration_path: Optional[Path] = None
    plane: Optional[ArucoPlane] = None
    ready_world_point: Optional[np.ndarray] = None
    ready_rtoe_image_point: Optional[np.ndarray] = None
    ready_rtoe_world_point: Optional[np.ndarray] = None
    ready_rtoe_confidence: Optional[float] = None
    ready_rtoe_debug_path: Optional[Path] = None
    baseline_y2: Optional[float] = None
    locked_bbox: Optional[List[int]] = None
    clip_frames: List[EncodedFrame] = field(default_factory=list)
    jump_samples: List[JumpFrameSample] = field(default_factory=list)
    review_samples: List[JumpFrameSample] = field(default_factory=list)
    jump_start_frame_index: Optional[int] = None
    trigger_landing_sample: Optional[JumpFrameSample] = None
    landing_sample: Optional[JumpFrameSample] = None
    peak_sample: Optional[JumpFrameSample] = None
    review_strategy: Optional[str] = None
    review_fallback_used: bool = False


class LatestCameraSource:
    """Capture camera frames in a background thread and expose the newest frame."""

    is_replay = False

    def __init__(self, camera_index: int, width: int, height: int, fps: float):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest_packet = None
        self._last_error = None
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        frame_index = 0
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                with self._condition:
                    self._last_error = "摄像头读取失败。"
                    self._condition.notify_all()
                break

            packet = FramePacket(
                index=frame_index,
                timestamp=time.monotonic(),
                image=frame,
            )
            frame_index += 1
            with self._condition:
                self._latest_packet = packet
                self._condition.notify_all()

    def read(self, last_index: Optional[int] = None, timeout: float = 1.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stopped:
                packet = self._latest_packet
                if packet is not None and (
                    last_index is None or packet.index != last_index
                ):
                    return packet
                if self._last_error:
                    raise RuntimeError(self._last_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return packet
                self._condition.wait(timeout=remaining)
        return None

    def close(self):
        self._stopped = True
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        self.cap.release()


class VideoReplaySource:
    """Read a video sequentially while using video time for state-machine timers."""

    is_replay = True

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开回放视频: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._index = 0

    def read(self, last_index: Optional[int] = None, timeout: float = 0.0):
        del last_index, timeout
        ok, frame = self.cap.read()
        if not ok:
            return None

        packet = FramePacket(
            index=self._index,
            timestamp=self._index / max(self.fps, 1.0),
            image=frame,
        )
        self._index += 1
        return packet

    def close(self):
        self.cap.release()


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


class ReadyStabilityTracker:
    def __init__(self, ready_seconds: float, jitter_cm: float):
        self.ready_seconds = ready_seconds
        self.jitter_cm = jitter_cm
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, detection: Optional[PersonDetection], timestamp: float) -> bool:
        if detection is None or detection.bottom_center_world is None:
            self.reset()
            return False

        point = np.asarray(detection.bottom_center_world, dtype=np.float32)
        self.samples.append((timestamp, point, detection))
        while (
            len(self.samples) >= 2
            and timestamp - self.samples[1][0] >= self.ready_seconds
        ):
            self.samples.popleft()

        if not self.samples:
            return False

        points = np.array([sample[1] for sample in self.samples], dtype=np.float32)
        center = np.median(points, axis=0)
        max_distance = float(np.max(np.linalg.norm(points - center, axis=1)))
        if max_distance > self.jitter_cm:
            self.samples.clear()
            self.samples.append((timestamp, point, detection))
            return False

        return timestamp - self.samples[0][0] >= self.ready_seconds

    def elapsed(self, timestamp: float) -> float:
        if not self.samples:
            return 0.0
        return max(0.0, timestamp - self.samples[0][0])

    def median_y2(self) -> float:
        return float(np.median([sample[2].y2 for sample in self.samples]))

    def median_world_point(self) -> np.ndarray:
        return np.median(
            np.array([sample[1] for sample in self.samples], dtype=np.float32),
            axis=0,
        )

    def latest_bbox(self) -> List[int]:
        return list(self.samples[-1][2].bbox)


class RtoeReadyStabilityTracker:
    def __init__(self, ready_seconds: float, jitter_cm: float):
        self.ready_seconds = ready_seconds
        self.jitter_cm = jitter_cm
        self.samples = deque()
        self.last_reason = "waiting"

    def reset(self, reason: str = "reset"):
        self.samples.clear()
        self.last_reason = reason

    def update(
        self,
        rtoe_image: Optional[np.ndarray],
        rtoe_world: Optional[np.ndarray],
        confidence: Optional[float],
        detection: Optional[PersonDetection],
        timestamp: float,
    ) -> bool:
        if rtoe_image is None or rtoe_world is None or detection is None:
            self.reset("no valid rtoe")
            return False

        image_point = np.asarray(rtoe_image, dtype=np.float32)
        world_point = np.asarray(rtoe_world, dtype=np.float32)
        conf_value = float(confidence or 0.0)
        self.samples.append((timestamp, image_point, world_point, conf_value, detection))
        while (
            len(self.samples) >= 2
            and timestamp - self.samples[1][0] >= self.ready_seconds
        ):
            self.samples.popleft()

        if not self.samples:
            return False

        points = np.array([sample[2] for sample in self.samples], dtype=np.float32)
        center = np.median(points, axis=0)
        max_distance = float(np.max(np.linalg.norm(points - center, axis=1)))
        if max_distance > self.jitter_cm:
            self.samples.clear()
            self.samples.append((timestamp, image_point, world_point, conf_value, detection))
            self.last_reason = f"rtoe jitter {max_distance:.1f}cm"
            return False

        self.last_reason = f"rtoe stable {max_distance:.1f}cm"
        return timestamp - self.samples[0][0] >= self.ready_seconds

    def elapsed(self, timestamp: float) -> float:
        if not self.samples:
            return 0.0
        return max(0.0, timestamp - self.samples[0][0])

    def median_y2(self) -> float:
        return float(np.median([sample[4].y2 for sample in self.samples]))

    def median_image_point(self) -> np.ndarray:
        return np.median(
            np.array([sample[1] for sample in self.samples], dtype=np.float32),
            axis=0,
        )

    def median_world_point(self) -> np.ndarray:
        return np.median(
            np.array([sample[2] for sample in self.samples], dtype=np.float32),
            axis=0,
        )

    def median_confidence(self) -> float:
        return float(np.median([sample[3] for sample in self.samples]))

    def latest_bbox(self) -> List[int]:
        return list(self.samples[-1][4].bbox)


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


def select_landing_sample(
    samples: Sequence[JumpFrameSample],
    stable_frames: int,
    max_change_px: float,
    min_forward_cm: float,
    start_world_x: float,
) -> Tuple[Optional[JumpFrameSample], Optional[JumpFrameSample]]:
    if len(samples) < stable_frames:
        return None, None

    peak_index = min(range(len(samples)), key=lambda index: samples[index].detection.y2)
    peak_sample = samples[peak_index]

    for start_index in range(peak_index, len(samples) - stable_frames + 1):
        window = samples[start_index : start_index + stable_frames]
        frame_indices = [sample.packet.index for sample in window]
        if any(
            frame_indices[index + 1] != frame_indices[index] + 1
            for index in range(len(frame_indices) - 1)
        ):
            continue

        y2_values = [sample.detection.y2 for sample in window]
        if max(y2_values) - min(y2_values) > max_change_px:
            continue

        first_world = window[0].detection.bottom_center_world
        if first_world is None or float(first_world[0]) - start_world_x < min_forward_cm:
            continue
        return window[0], peak_sample

    return None, peak_sample


def select_landing_sample_static_style(
    samples: Sequence[JumpFrameSample],
    stable_frames: int,
    max_change_px: float,
    min_forward_cm: float,
    start_world_x: float,
) -> Tuple[Optional[JumpFrameSample], Optional[JumpFrameSample], Optional[str]]:
    """Review cached frames with the validated offline landing-window rules."""
    if len(samples) < stable_frames:
        return None, None, None

    peak_index = min(range(len(samples)), key=lambda index: samples[index].detection.y2)
    peak_sample = samples[peak_index]
    window = deque(maxlen=stable_frames)

    for sample in samples[peak_index:]:
        window.append(sample)
        if len(window) < stable_frames:
            continue

        window_list = list(window)
        y2_values = [item.detection.y2 for item in window_list]
        first_sample = window_list[0]
        first_world = first_sample.detection.bottom_center_world
        if first_world is None or float(first_world[0]) - start_world_x < min_forward_cm:
            continue

        first_y2 = y2_values[0]
        if first_y2 == max(y2_values) and all(
            value <= first_y2 for value in y2_values[1:]
        ):
            return first_sample, peak_sample, "first_frame_max"

        if max(y2_values) - min(y2_values) <= max_change_px:
            return first_sample, peak_sample, "stable_window"

    return None, peak_sample, None


def person_bbox_area(detection: PersonDetection) -> int:
    x1, y1, x2, y2 = detection.bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def select_dominant_review_person(
    detections: Sequence[PersonDetection],
) -> Optional[PersonDetection]:
    """Select the jumping subject during cache review; the subject dominates the frame."""
    if not detections:
        return None
    return max(detections, key=person_bbox_area)


def encode_frame(packet: FramePacket, jpeg_quality: int = 85) -> EncodedFrame:
    ok, buffer = cv2.imencode(
        ".jpg",
        packet.image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("无法压缩摄像头帧。")
    return EncodedFrame(
        index=packet.index,
        timestamp=packet.timestamp,
        jpeg_bytes=buffer.tobytes(),
    )


def decode_frame(encoded_frame: EncodedFrame) -> np.ndarray:
    array = np.frombuffer(encoded_frame.jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法解码缓存帧: {encoded_frame.index}")
    return image


def save_clip(frames: Sequence[EncodedFrame], output_path: Path, fps: float):
    if not frames:
        return None

    first_image = decode_frame(frames[0])
    height, width = first_image.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建短视频文件: {output_path}")

    try:
        writer.write(first_image)
        for encoded_frame in frames[1:]:
            image = decode_frame(encoded_frame)
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()
    return output_path


def resolve_path(path_value: str, extra_roots: Sequence[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    for root in extra_roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return cwd_candidate


def resolve_video_file(path_value: str) -> Path:
    return resolve_path(path_value, [ARUCOTEST_DIR / "input", ARUCOTEST_DIR, WCX_DIR])


def choose_font_path(font_value: Optional[str]) -> Optional[Path]:
    if font_value:
        path = resolve_path(font_value, [WCX_DIR])
        if not path.exists():
            raise FileNotFoundError(f"中文字体不存在: {path}")
        return path
    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def create_session_dir() -> Path:
    session_dir = REALTIME_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    base_dir = session_dir
    while session_dir.exists():
        session_dir = Path(f"{base_dir}_{suffix:02d}")
        suffix += 1
    (session_dir / "attempts").mkdir(parents=True, exist_ok=False)
    return session_dir


def create_attempt_dir(session_dir: Path, attempt_number: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = session_dir / "attempts" / f"{timestamp}_{attempt_number:03d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def load_models(
    yolo_model_path: Path,
    sam2_checkpoint: Path,
    sam2_config: str,
    device: str,
    keypoint_repo_root: Path,
    keypoint_weights_path: Path,
    keypoint_bbox_expand_ratio: float,
):
    from ultralytics import YOLO
    from all_keypoints_infer import AllKeypointsJumpInfer

    ensure_sam2_import_path()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    yolo_model = YOLO(str(yolo_model_path))
    keypoint_infer = AllKeypointsJumpInfer(
        repo_root=keypoint_repo_root,
        weights_path=keypoint_weights_path,
        device=device,
        bbox_expand_ratio=keypoint_bbox_expand_ratio,
    )
    sam2_model = build_sam2(
        config_file=normalize_sam2_config(sam2_config),
        ckpt_path=str(sam2_checkpoint),
        device=device,
    )
    return ModelBundle(
        yolo_model=yolo_model,
        sam2_predictor=SAM2ImagePredictor(sam2_model),
        keypoint_infer=keypoint_infer,
        torch_module=torch,
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


def segment_with_loaded_predictor(
    models: ModelBundle,
    landing_frame: np.ndarray,
    prompt_box: Sequence[int],
):
    predictor = models.sam2_predictor
    image_rgb = cv2.cvtColor(landing_frame, cv2.COLOR_BGR2RGB)
    with models.torch_module.inference_mode():
        predictor.set_image(image_rgb)
        masks, iou_predictions, _ = predictor.predict(
            box=np.asarray(prompt_box, dtype=np.float32),
            multimask_output=True,
        )

    best_index = int(np.argmax(iou_predictions))
    return {
        "mask": (masks[best_index] > 0).astype(np.uint8),
        "best_mask_index": best_index,
        "iou_predictions": [float(value) for value in iou_predictions],
    }


def draw_ready_zone(image: np.ndarray, plane: Optional[ArucoPlane], args):
    if plane is None:
        return
    world_points = np.array(
        [
            [args.ready_x_min, args.ready_y_min],
            [args.ready_x_max, args.ready_y_min],
            [args.ready_x_max, args.ready_y_max],
            [args.ready_x_min, args.ready_y_max],
        ],
        dtype=np.float32,
    )
    image_points = transform_points(world_points, plane.world_to_image).astype(np.int32)
    cv2.polylines(image, [image_points], True, (0, 255, 255), 3, cv2.LINE_AA)


def draw_people(image: np.ndarray, detections, active_detection=None):
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 255, 0) if detection is active_detection else (180, 180, 180)
        thickness = 3 if detection is active_detection else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        point = tuple(detection.bottom_center_image.astype(int))
        cv2.circle(image, point, 6, color, -1, cv2.LINE_AA)


def draw_rtoe_point(image: np.ndarray, point: Optional[np.ndarray], confidence: Optional[float]):
    if point is None:
        return
    x_value, y_value = [int(round(float(value))) for value in point]
    cv2.circle(image, (x_value, y_value), 8, (0, 0, 255), -1, cv2.LINE_AA)
    label = "rtoe" if confidence is None else f"rtoe {confidence:.2f}"
    cv2.putText(
        image,
        label,
        (x_value + 8, y_value - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def save_ready_rtoe_debug(
    image: np.ndarray,
    output_path: Path,
    detections,
    active_detection,
    rtoe_image: Optional[np.ndarray],
    rtoe_world: Optional[np.ndarray],
    confidence: Optional[float],
    plane: Optional[ArucoPlane],
    args,
):
    debug = image.copy()
    if plane is not None:
        draw_world_reference(debug, plane)
        draw_ready_zone(debug, plane, args)
    draw_people(debug, detections, active_detection)
    draw_rtoe_point(debug, rtoe_image, confidence)
    if rtoe_image is not None:
        x_value, y_value = [int(round(float(value))) for value in rtoe_image]
        world_text = "rtoe world: n/a"
        if rtoe_world is not None:
            world_text = f"rtoe world x={float(rtoe_world[0]):.2f}cm y={float(rtoe_world[1]):.2f}cm"
        conf_text = "conf=n/a" if confidence is None else f"conf={float(confidence):.3f}"
        cv2.putText(
            debug,
            world_text,
            (max(10, x_value - 160), min(debug.shape[0] - 35, y_value + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            conf_text,
            (max(10, x_value - 160), min(debug.shape[0] - 10, y_value + 58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def draw_marker_detections(image: np.ndarray, detections: Dict[int, object]):
    for marker_id, detection in detections.items():
        corners = detection.corners.astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
        anchor = tuple(detection.anchor_point.astype(int))
        cv2.circle(image, anchor, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"ID{marker_id}",
            (anchor[0] + 6, anchor[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


class ChineseTextRenderer:
    def __init__(self, font_path: Optional[Path]):
        self.font_path = font_path
        self._cache = {}

    def font(self, size: int):
        key = int(size)
        if key not in self._cache:
            if self.font_path:
                self._cache[key] = ImageFont.truetype(str(self.font_path), key)
            else:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]

    def draw_lines(
        self,
        image: np.ndarray,
        lines: Sequence[Tuple[str, Tuple[int, int, int], int]],
        origin: Tuple[int, int],
        line_gap: int = 12,
    ):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(pil_image)
        x_value, y_value = origin
        for text, bgr_color, font_size in lines:
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            drawer.text((x_value, y_value), text, fill=rgb_color, font=self.font(font_size))
            y_value += font_size + line_gap
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


class RealtimeJumpApp:
    def __init__(self, args):
        self.args = args
        self.session_dir = create_session_dir()
        self.state = RealtimeState.LOADING_MODELS
        self.models = None
        self.model_error = None
        self.model_thread = None
        self.source = None
        self.font_renderer = ChineseTextRenderer(choose_font_path(args.font))
        self.calib_tracker = CalibrationStabilityTracker(
            stable_seconds=args.calib_stable_seconds,
            jitter_px=args.calib_jitter_px,
        )
        self.ready_tracker = RtoeReadyStabilityTracker(
            ready_seconds=args.ready_seconds,
            jitter_cm=args.ready_jitter_cm,
        )
        self.attempt_number = 0
        self.attempt = None
        self.last_packet = None
        self.last_processed_index = None
        self.last_detections = []
        self.active_detection = None
        self.last_ready_rtoe_image = None
        self.last_ready_rtoe_world = None
        self.last_ready_rtoe_confidence = None
        self.last_ready_status = "waiting"
        self.last_marker_detections = {}
        self.state_started_at = 0.0
        self.takeoff_y2_values = deque(maxlen=2)
        self.lost_person_frames = 0
        self.pre_roll_frames = deque(maxlen=max(1, int(args.camera_fps)))
        self.processing_thread = None
        self.processing_result = None
        self.processing_error = None
        self.result_image = None
        self.failure_reason = None
        self.source_error = None
        self.window_created = False
        self.should_stop = False

    def start_source(self):
        if self.args.video_file:
            video_path = resolve_video_file(self.args.video_file)
            if not video_path.exists():
                raise FileNotFoundError(f"回放视频不存在: {video_path}")
            self.source = VideoReplaySource(video_path)
            print(f"[OK] 回放视频: {video_path}")
        else:
            self.source = LatestCameraSource(
                camera_index=self.args.camera_index,
                width=self.args.camera_width,
                height=self.args.camera_height,
                fps=self.args.camera_fps,
            )
            print(f"[OK] 摄像头: {self.args.camera_index}")

    def start_model_loading(self):
        yolo_path = resolve_path(self.args.yolo_model, [WCX_DIR, ARUCOTEST_DIR])
        sam2_checkpoint = resolve_path(
            self.args.sam2_checkpoint,
            [WCX_DIR, ARUCOTEST_DIR],
        )
        keypoint_repo_root = resolve_path(
            self.args.keypoint_repo_root,
            [WCX_DIR, ARUCOTEST_DIR],
        )
        keypoint_weights_path = resolve_path(
            self.args.keypoint_weights_path,
            [keypoint_repo_root, WCX_DIR, ARUCOTEST_DIR],
        )
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO 模型不存在: {yolo_path}")
        if not sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint 不存在: {sam2_checkpoint}")

        if not keypoint_repo_root.exists():
            raise FileNotFoundError(f"all-keypoints repo_root does not exist: {keypoint_repo_root}")
        if not keypoint_weights_path.exists():
            raise FileNotFoundError(f"all-keypoints weights do not exist: {keypoint_weights_path}")

        def loader():
            try:
                self.models = load_models(
                    yolo_model_path=yolo_path,
                    sam2_checkpoint=sam2_checkpoint,
                    sam2_config=self.args.sam2_config,
                    device=self.args.device,
                    keypoint_repo_root=keypoint_repo_root,
                    keypoint_weights_path=keypoint_weights_path,
                    keypoint_bbox_expand_ratio=self.args.keypoint_bbox_expand_ratio,
                )
            except Exception as exc:
                self.model_error = exc

        self.model_thread = threading.Thread(target=loader, daemon=True)
        self.model_thread.start()

    def reset_for_calibration(self, timestamp: float):
        self.attempt_number += 1
        self.attempt = AttemptContext(
            output_dir=create_attempt_dir(self.session_dir, self.attempt_number)
        )
        self.state = RealtimeState.CALIBRATING
        self.state_started_at = timestamp
        self.calib_tracker.reset()
        self.ready_tracker.reset()
        self.last_marker_detections = {}
        self.last_detections = []
        self.active_detection = None
        self.last_ready_rtoe_image = None
        self.last_ready_rtoe_world = None
        self.last_ready_rtoe_confidence = None
        self.last_ready_status = "waiting"
        self.takeoff_y2_values.clear()
        self.lost_person_frames = 0
        self.pre_roll_frames.clear()
        self.processing_result = None
        self.processing_error = None
        self.result_image = None
        self.failure_reason = None
        print(f"[STATE] {self.state.value}: {self.attempt.output_dir}")

    def read_source(self, last_index: Optional[int] = None, timeout: float = 0.0):
        if self.source_error is not None:
            return None
        try:
            packet = self.source.read(last_index=last_index, timeout=timeout)
            if packet is None:
                return None
            rotated_image = rotate_frame(packet.image, self.args.rotate_frame)
            if rotated_image is packet.image:
                return packet
            return FramePacket(
                index=packet.index,
                timestamp=packet.timestamp,
                image=rotated_image,
            )
        except Exception as exc:
            self.source_error = str(exc)
            self.fail(f"视频源读取失败: {exc}", self.last_packet)
            return None

    def transition(self, state: RealtimeState, timestamp: float):
        self.state = state
        self.state_started_at = timestamp
        print(f"[STATE] {state.value}")

    def handle_calibrating(self, packet: FramePacket):
        detections = detect_aruco_markers(packet.image)
        self.last_marker_detections = detections
        if not self.calib_tracker.update(detections, packet.timestamp):
            return

        calib_path = self.attempt.output_dir / "calib_frame.jpg"
        debug_path = self.attempt.output_dir / "calib_aruco_debug.jpg"
        cv2.imwrite(str(calib_path), packet.image)
        plane = build_aruco_plane(packet.image, debug_path=debug_path)
        self.attempt.calibration_frame = packet.image.copy()
        self.attempt.calibration_path = calib_path
        self.attempt.plane = plane
        self.ready_tracker.reset()
        self.transition(RealtimeState.WAITING_READY, packet.timestamp)

    def predict_ready_rtoe(self, image: np.ndarray, detection: PersonDetection):
        keypoints = self.models.keypoint_infer.predict_frame(
            image,
            bbox=detection.bbox,
            bbox_format="xyxy",
        )
        rtoe = self.models.keypoint_infer.get_keypoint(keypoints, "rtoe")
        if rtoe is None:
            return None, None, None

        confidence = float(rtoe.get("confidence", 0.0))
        image_point = np.array([float(rtoe["x"]), float(rtoe["y"])], dtype=np.float32)
        world_point = transform_points(
            image_point.reshape(1, 2),
            self.attempt.plane.image_to_world,
        )[0]
        return image_point, world_point, confidence

    def handle_waiting_ready(self, packet: FramePacket):
        self.last_detections = detect_people(
            self.models.yolo_model,
            packet.image,
            self.attempt.plane,
        )
        self.active_detection = select_ready_person(
            self.last_detections,
            x_min=self.args.ready_x_min,
            x_max=self.args.ready_x_max,
            y_min=self.args.ready_y_min,
            y_max=self.args.ready_y_max,
        )
        self.pre_roll_frames.append(encode_frame(packet, self.args.jpeg_quality))

        if self.active_detection is None:
            self.last_ready_rtoe_image = None
            self.last_ready_rtoe_world = None
            self.last_ready_rtoe_confidence = None
            self.last_ready_status = "no person in ready zone"
            self.ready_tracker.reset(self.last_ready_status)
            return

        try:
            rtoe_image, rtoe_world, rtoe_confidence = self.predict_ready_rtoe(
                packet.image,
                self.active_detection,
            )
        except Exception as exc:
            self.fail(f"keypoint inference failed during ready stage: {exc}", packet)
            return

        self.last_ready_rtoe_image = rtoe_image
        self.last_ready_rtoe_world = rtoe_world
        self.last_ready_rtoe_confidence = rtoe_confidence

        if rtoe_image is None or rtoe_world is None or rtoe_confidence is None:
            self.last_ready_status = "rtoe not detected"
            self.ready_tracker.reset(self.last_ready_status)
            return

        if rtoe_confidence < self.args.rtoe_confidence_threshold:
            self.last_ready_status = (
                f"rtoe low confidence {rtoe_confidence:.2f}"
            )
            self.ready_tracker.reset(self.last_ready_status)
            return

        if float(rtoe_world[0]) > self.args.start_line_tolerance_cm:
            self.last_ready_status = (
                f"rtoe over line x={float(rtoe_world[0]):.1f}cm"
            )
            self.ready_tracker.reset(self.last_ready_status)
            return

        if not self.ready_tracker.update(
            rtoe_image,
            rtoe_world,
            rtoe_confidence,
            self.active_detection,
            packet.timestamp,
        ):
            self.last_ready_status = self.ready_tracker.last_reason
            return

        self.attempt.baseline_y2 = self.ready_tracker.median_y2()
        self.attempt.ready_world_point = self.ready_tracker.median_world_point()
        self.attempt.ready_rtoe_image_point = self.ready_tracker.median_image_point()
        self.attempt.ready_rtoe_world_point = self.ready_tracker.median_world_point()
        self.attempt.ready_rtoe_confidence = self.ready_tracker.median_confidence()
        self.attempt.locked_bbox = self.ready_tracker.latest_bbox()
        self.attempt.clip_frames = list(self.pre_roll_frames)
        ready_debug_path = self.attempt.output_dir / "ready_rtoe_debug.jpg"
        save_ready_rtoe_debug(
            packet.image,
            ready_debug_path,
            self.last_detections,
            self.active_detection,
            self.last_ready_rtoe_image,
            self.last_ready_rtoe_world,
            self.last_ready_rtoe_confidence,
            self.attempt.plane,
            self.args,
        )
        self.attempt.ready_rtoe_debug_path = ready_debug_path
        self.last_ready_status = "rtoe ready"
        self.takeoff_y2_values.clear()
        self.transition(RealtimeState.ARMED, packet.timestamp)

    def handle_armed(self, packet: FramePacket):
        detections = detect_people(
            self.models.yolo_model,
            packet.image,
            self.attempt.plane,
        )
        active = match_locked_person(
            detections,
            self.attempt.locked_bbox,
            max_center_distance_px=self.args.max_track_distance_px,
        )
        self.last_detections = detections
        self.active_detection = active
        self.record_clip_packet(packet)

        if packet.timestamp - self.state_started_at > self.args.takeoff_timeout_seconds:
            self.fail("等待起跳超时。", packet)
            return

        if active is None:
            self.lost_person_frames += 1
            if self.lost_person_frames > self.args.max_lost_frames:
                self.fail("准备起跳阶段连续丢失测试人。", packet)
            return

        self.lost_person_frames = 0
        self.attempt.locked_bbox = list(active.bbox)
        self.takeoff_y2_values.append(active.y2)
        if detect_takeoff(
            self.takeoff_y2_values,
            baseline_y2=self.attempt.baseline_y2,
            frame_height=packet.image.shape[0],
            min_pixels=self.args.takeoff_min_px,
            image_ratio=self.args.takeoff_image_ratio,
        ):
            self.attempt.jump_samples = []
            self.attempt.jump_start_frame_index = packet.index
            self.append_jump_sample(packet, active)
            self.transition(RealtimeState.IN_JUMP, packet.timestamp)

    def handle_in_jump(self, packet: FramePacket):
        detections = detect_people(
            self.models.yolo_model,
            packet.image,
            self.attempt.plane,
        )
        active = match_locked_person(
            detections,
            self.attempt.locked_bbox,
            max_center_distance_px=self.args.max_track_distance_px,
        )
        self.last_detections = detections
        self.active_detection = active
        self.record_clip_packet(packet)

        if packet.timestamp - self.state_started_at > self.args.max_jump_seconds:
            self.fail("跳跃过程超时，未检测到稳定落地帧。", packet)
            return

        if active is None:
            self.lost_person_frames += 1
            if self.lost_person_frames > self.args.max_lost_frames:
                self.fail("跳跃过程中连续丢失测试人。", packet)
            return

        self.lost_person_frames = 0
        self.attempt.locked_bbox = list(active.bbox)
        self.append_jump_sample(packet, active)
        landing_sample, peak_sample = select_landing_sample(
            samples=self.attempt.jump_samples,
            stable_frames=self.args.landing_stable_frames,
            max_change_px=self.args.landing_max_change_px,
            min_forward_cm=self.args.min_forward_cm,
            start_world_x=float(self.attempt.ready_world_point[0]),
        )
        self.attempt.peak_sample = peak_sample
        if landing_sample is None:
            return

        self.attempt.trigger_landing_sample = landing_sample
        self.start_processing(packet.timestamp)

    def append_jump_sample(self, packet: FramePacket, detection: PersonDetection):
        encoded = self.record_clip_packet(packet)
        self.attempt.jump_samples.append(
            JumpFrameSample(
                packet=FramePacket(
                    index=packet.index,
                    timestamp=packet.timestamp,
                    image=packet.image.copy(),
                ),
                detection=detection,
                encoded_frame=encoded,
            )
        )

    def record_clip_packet(self, packet: FramePacket) -> EncodedFrame:
        if self.attempt.clip_frames and self.attempt.clip_frames[-1].index == packet.index:
            return self.attempt.clip_frames[-1]
        encoded = encode_frame(packet, self.args.jpeg_quality)
        self.attempt.clip_frames.append(encoded)
        return encoded

    def start_processing(self, timestamp: float):
        self.transition(RealtimeState.PROCESSING, timestamp)

        def worker():
            try:
                self.processing_result = self.process_attempt()
            except Exception as exc:
                self.processing_error = exc

        self.processing_thread = threading.Thread(target=worker, daemon=True)
        self.processing_thread.start()

    def review_cached_landing_sample(self):
        attempt = self.attempt
        landing_sample, peak_sample, strategy = select_landing_sample_static_style(
            samples=attempt.jump_samples,
            stable_frames=self.args.landing_stable_frames,
            max_change_px=self.args.landing_max_change_px,
            min_forward_cm=self.args.min_forward_cm,
            start_world_x=float(attempt.ready_world_point[0]),
        )
        if landing_sample is not None:
            attempt.review_samples = list(attempt.jump_samples)
            attempt.peak_sample = peak_sample
            attempt.review_strategy = f"tracked_raw_{strategy}"
            attempt.review_fallback_used = False
            return landing_sample

        review_samples = []
        for encoded_frame in attempt.clip_frames:
            if (
                attempt.jump_start_frame_index is not None
                and encoded_frame.index < attempt.jump_start_frame_index
            ):
                continue

            image = decode_frame(encoded_frame)
            detections = detect_people(
                self.models.yolo_model,
                image,
                attempt.plane,
            )
            detection = select_dominant_review_person(detections)
            if detection is None:
                continue

            review_samples.append(
                JumpFrameSample(
                    packet=FramePacket(
                        index=encoded_frame.index,
                        timestamp=encoded_frame.timestamp,
                        image=image,
                    ),
                    detection=detection,
                    encoded_frame=encoded_frame,
                )
            )

        attempt.review_samples = review_samples
        landing_sample, peak_sample, strategy = select_landing_sample_static_style(
            samples=review_samples,
            stable_frames=self.args.landing_stable_frames,
            max_change_px=self.args.landing_max_change_px,
            min_forward_cm=self.args.min_forward_cm,
            start_world_x=float(attempt.ready_world_point[0]),
        )
        attempt.peak_sample = peak_sample
        if landing_sample is not None:
            attempt.review_strategy = f"compressed_cache_{strategy}"
            attempt.review_fallback_used = False
            return landing_sample

        if attempt.trigger_landing_sample is None:
            raise ValueError("短缓存复核未找到落地帧，且没有在线候选帧。")

        attempt.review_strategy = "online_candidate_fallback"
        attempt.review_fallback_used = True
        return attempt.trigger_landing_sample

    def process_attempt(self) -> ProcessingResult:
        attempt = self.attempt
        attempt.landing_sample = self.review_cached_landing_sample()
        landing_frame = attempt.landing_sample.packet.image
        landing_bbox = attempt.landing_sample.detection.bbox
        expanded_bbox = expand_box(
            landing_bbox,
            landing_frame.shape,
            self.args.bbox_expand_ratio,
        )
        bbox_info = {
            "box": expanded_bbox,
            "original_box": landing_bbox,
            "confidence": attempt.landing_sample.detection.confidence,
            "detected_person_count": None,
            "source": "realtime_yolo",
        }

        landing_path = attempt.output_dir / "landing_frame.jpg"
        cv2.imwrite(str(landing_path), landing_frame)
        segmentation_info = segment_with_loaded_predictor(
            models=self.models,
            landing_frame=landing_frame,
            prompt_box=expanded_bbox,
        )
        output_paths = save_segmentation_outputs(
            landing_frame,
            segmentation_info["mask"],
            bbox_info,
            attempt.output_dir,
        )
        measurement = measure_jump_from_files(
            image_path=landing_path,
            mask_path=output_paths["mask_path"],
            output_dir=attempt.output_dir,
            calib_image_path=attempt.calibration_path,
        )
        result_path = attempt.output_dir / "result_aruco.txt"
        write_result_file(
            result_path,
            measurement,
            image_path=landing_path,
            mask_path=output_paths["mask_path"],
            calib_image_path=attempt.calibration_path,
        )
        if self.args.save_clip:
            save_clip(
                attempt.clip_frames,
                attempt.output_dir / "jump_clip.mp4",
                fps=self.source.fps,
            )

        json_path = attempt.output_dir / "realtime_result.json"
        payload = {
            "score_cm": measurement.score_cm,
            "state": "success",
            "calibration_frame": str(attempt.calibration_path),
            "landing_frame": str(landing_path),
            "landing_frame_index": attempt.landing_sample.packet.index,
            "trigger_landing_frame_index": (
                attempt.trigger_landing_sample.packet.index
                if attempt.trigger_landing_sample
                else None
            ),
            "review_strategy": attempt.review_strategy,
            "review_fallback_used": attempt.review_fallback_used,
            "peak_frame_index": (
                attempt.peak_sample.packet.index if attempt.peak_sample else None
            ),
            "baseline_y2": attempt.baseline_y2,
            "ready_world_point": [
                float(attempt.ready_world_point[0]),
                float(attempt.ready_world_point[1]),
            ],
            "ready_rtoe_image_point": (
                [
                    float(attempt.ready_rtoe_image_point[0]),
                    float(attempt.ready_rtoe_image_point[1]),
                ]
                if attempt.ready_rtoe_image_point is not None
                else None
            ),
            "ready_rtoe_world_point": (
                [
                    float(attempt.ready_rtoe_world_point[0]),
                    float(attempt.ready_rtoe_world_point[1]),
                ]
                if attempt.ready_rtoe_world_point is not None
                else None
            ),
            "ready_rtoe_confidence": attempt.ready_rtoe_confidence,
            "ready_rtoe_debug": (
                str(attempt.ready_rtoe_debug_path)
                if attempt.ready_rtoe_debug_path is not None
                else None
            ),
            "heel_image_point": [
                float(measurement.heel_result.image_point[0]),
                float(measurement.heel_result.image_point[1]),
            ],
            "heel_world_point": [
                float(measurement.heel_result.world_point[0]),
                float(measurement.heel_result.world_point[1]),
            ],
            "segmentation": {
                "best_mask_index": segmentation_info["best_mask_index"],
                "iou_predictions": segmentation_info["iou_predictions"],
            },
            "tracked_frames": [
                {
                    "frame_index": sample.packet.index,
                    "timestamp": sample.packet.timestamp,
                    "y2": sample.detection.y2,
                    "bottom_world_point": (
                        [
                            float(sample.detection.bottom_center_world[0]),
                            float(sample.detection.bottom_center_world[1]),
                        ]
                        if sample.detection.bottom_center_world is not None
                        else None
                    ),
                }
                for sample in attempt.jump_samples
            ],
            "reviewed_frames": [
                {
                    "frame_index": sample.packet.index,
                    "timestamp": sample.packet.timestamp,
                    "y2": sample.detection.y2,
                    "bottom_world_point": (
                        [
                            float(sample.detection.bottom_center_world[0]),
                            float(sample.detection.bottom_center_world[1]),
                        ]
                        if sample.detection.bottom_center_world is not None
                        else None
                    ),
                }
                for sample in attempt.review_samples
            ],
            "outputs": {
                "result": str(result_path),
                "visualization_aruco": str(measurement.visualization_path),
                "aruco_debug": str(measurement.aruco_debug_path),
                "person_mask": str(output_paths["mask_path"]),
            },
        }
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return ProcessingResult(
            score_cm=float(measurement.score_cm),
            visualization_path=measurement.visualization_path,
            result_path=result_path,
            json_path=json_path,
        )

    def poll_processing(self, timestamp: float):
        if self.processing_thread is None or self.processing_thread.is_alive():
            return
        if self.processing_error is not None:
            self.fail(f"成绩生成失败: {self.processing_error}", self.last_packet)
            return

        self.result_image = cv2.imread(str(self.processing_result.visualization_path))
        self.transition(RealtimeState.RESULT, timestamp)
        print(f"[OK] 跳远成绩: {self.processing_result.score_cm:.2f} cm")
        print(f"[OK] 结果目录: {self.attempt.output_dir}")

    def fail(self, reason: str, packet: Optional[FramePacket]):
        self.failure_reason = reason
        self.state = RealtimeState.FAILED
        print(f"[FAILED] {reason}")
        if self.attempt is None:
            return

        if packet is not None:
            cv2.imwrite(str(self.attempt.output_dir / "failure_last_frame.jpg"), packet.image)
        if self.args.save_clip and self.attempt.clip_frames:
            try:
                save_clip(
                    self.attempt.clip_frames,
                    self.attempt.output_dir / "jump_clip.mp4",
                    fps=self.source.fps,
                )
            except Exception as exc:
                print(f"[WARN] 保存失败短片时出错: {exc}")

        payload = {
            "state": "failed",
            "reason": reason,
            "attempt": self.attempt_number,
            "last_frame_index": packet.index if packet else None,
        }
        with open(self.attempt.output_dir / "failure.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def handle_packet(self, packet: FramePacket):
        if self.state == RealtimeState.CALIBRATING:
            self.handle_calibrating(packet)
        elif self.state == RealtimeState.WAITING_READY:
            self.handle_waiting_ready(packet)
        elif self.state == RealtimeState.ARMED:
            self.handle_armed(packet)
        elif self.state == RealtimeState.IN_JUMP:
            self.handle_in_jump(packet)
        elif self.state == RealtimeState.PROCESSING:
            self.poll_processing(packet.timestamp)

    def status_lines(self, timestamp: float):
        if self.state == RealtimeState.LOADING_MODELS:
            return [("模型加载中……", (0, 255, 255), 34)]
        if self.state == RealtimeState.CALIBRATING:
            ids = sorted(self.last_marker_detections.keys())
            elapsed = self.calib_tracker.elapsed(timestamp)
            return [
                ("正在进行场景检测……", (0, 255, 255), 34),
                (f"已识别 ArUco ID: {ids}", (255, 255, 255), 22),
                (
                    f"稳定确认: {elapsed:.1f}/{self.args.calib_stable_seconds:.1f} 秒",
                    (255, 255, 255),
                    22,
                ),
            ]
        if self.state == RealtimeState.WAITING_READY:
            elapsed = self.ready_tracker.elapsed(timestamp)
            rtoe_x = "n/a"
            if self.last_ready_rtoe_world is not None:
                rtoe_x = f"{float(self.last_ready_rtoe_world[0]):.1f}cm"
            rtoe_conf = "n/a"
            if self.last_ready_rtoe_confidence is not None:
                rtoe_conf = f"{float(self.last_ready_rtoe_confidence):.2f}"
            return [
                ("场景检测完成，请测试人准备", (0, 255, 0), 31),
                ("请站到起跳线附近并保持静止", (255, 255, 255), 24),
                (
                    f"站定确认: {elapsed:.1f}/{self.args.ready_seconds:.1f} 秒",
                    (0, 255, 255),
                    25,
                ),
                (f"rtoe x: {rtoe_x}  conf: {rtoe_conf}", (255, 255, 255), 22),
                (f"rtoe status: {self.last_ready_status}", (255, 255, 255), 20),
            ]
        if self.state == RealtimeState.ARMED:
            return [
                ("开始测试", (0, 255, 0), 42),
                ("请完成立定跳远", (255, 255, 255), 25),
            ]
        if self.state == RealtimeState.IN_JUMP:
            return [
                ("正在检测落地……", (0, 255, 255), 34),
                (f"已缓存帧数: {len(self.attempt.jump_samples)}", (255, 255, 255), 22),
            ]
        if self.state == RealtimeState.PROCESSING:
            return [("结果生成中……", (0, 255, 255), 38)]
        if self.state == RealtimeState.FAILED:
            return [
                ("本轮测试失败", (0, 0, 255), 38),
                (self.failure_reason or "未知原因", (255, 255, 255), 23),
                ("按空格重新标定并重试", (0, 255, 255), 24),
            ]
        return []

    def fit_image_to_region(self, image: np.ndarray, max_width: int, max_height: int):
        scale = min(
            1.0,
            max_width / max(image.shape[1], 1),
            max_height / max(image.shape[0], 1),
        )
        if scale < 1.0:
            return cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        return image

    def draw_preview_banner(self, image: np.ndarray, timestamp: float):
        lines = self.status_lines(timestamp)
        if not lines:
            return image
        banner_height = min(74, image.shape[0])
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)
        text, color, font_size = lines[0]
        return self.font_renderer.draw_lines(
            image,
            [(text, color, min(font_size, 34))],
            (16, 12),
            line_gap=0,
        )

    def render_live_canvas(self, packet: FramePacket):
        preview = packet.image.copy()
        if self.state == RealtimeState.CALIBRATING:
            draw_marker_detections(preview, self.last_marker_detections)
        if self.attempt and self.attempt.plane is not None:
            draw_world_reference(preview, self.attempt.plane)
            draw_ready_zone(preview, self.attempt.plane, self.args)
        draw_people(preview, self.last_detections, self.active_detection)
        if self.state == RealtimeState.WAITING_READY:
            draw_rtoe_point(
                preview,
                self.last_ready_rtoe_image,
                self.last_ready_rtoe_confidence,
            )
        preview = self.draw_preview_banner(preview, packet.timestamp)

        panel_width = min(self.args.panel_width, self.args.window_width - 320)
        preview_region_width = max(320, self.args.window_width - panel_width)
        preview = self.fit_image_to_region(
            preview,
            max_width=preview_region_width,
            max_height=self.args.window_height,
        )
        canvas = np.zeros(
            (self.args.window_height, self.args.window_width, 3),
            dtype=np.uint8,
        )
        canvas[:, :] = (24, 24, 24)
        canvas[: preview.shape[0], : preview.shape[1]] = preview
        panel_x = preview_region_width + 20
        lines = [(self.state.value, (180, 180, 180), 22)]
        lines.extend(self.status_lines(packet.timestamp))
        lines.extend(
            [
                ("", (255, 255, 255), 12),
                ("快捷键", (255, 255, 255), 24),
                ("空格: 下一轮 / 重试", (180, 180, 180), 20),
                ("R: 重新标定", (180, 180, 180), 20),
                ("Q / Esc: 退出", (180, 180, 180), 20),
            ]
        )
        return self.font_renderer.draw_lines(canvas, lines, (panel_x, 24))

    def render_result_canvas(self):
        image = self.result_image
        if image is None:
            image = self.last_packet.image.copy()
        panel_width = min(self.args.panel_width, self.args.window_width - 320)
        preview_region_width = max(320, self.args.window_width - panel_width)
        image = self.fit_image_to_region(
            image,
            max_width=preview_region_width,
            max_height=self.args.window_height,
        )
        canvas = np.zeros(
            (self.args.window_height, self.args.window_width, 3),
            dtype=np.uint8,
        )
        canvas[:, :] = (24, 24, 24)
        canvas[: image.shape[0], : image.shape[1]] = image
        panel_x = preview_region_width + 20
        score_text = f"{self.processing_result.score_cm:.2f} cm"
        lines = [
            ("测试完成", (0, 255, 0), 36),
            ("跳远成绩", (255, 255, 255), 28),
            (score_text, (0, 255, 255), 52),
            ("", (255, 255, 255), 14),
            ("按空格开始下一轮", (255, 255, 255), 24),
            ("下一轮将重新检测场景", (180, 180, 180), 20),
            ("Q / Esc: 退出", (180, 180, 180), 20),
        ]
        return self.font_renderer.draw_lines(canvas, lines, (panel_x, 34), line_gap=14)

    def create_window(self):
        if self.args.headless or self.window_created:
            return
        cv2.namedWindow(self.args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.args.window_name,
            self.args.window_width,
            self.args.window_height,
        )
        self.window_created = True

    def window_is_open(self):
        if self.args.headless or not self.window_created:
            return True
        try:
            return cv2.getWindowProperty(
                self.args.window_name,
                cv2.WND_PROP_VISIBLE,
            ) >= 1
        except cv2.error:
            return False

    def handle_key(self, key: int, timestamp: float):
        if key in (ord("q"), ord("Q"), 27):
            self.should_stop = True
            return
        if key in (ord("r"), ord("R")) and self.state not in (
            RealtimeState.LOADING_MODELS,
            RealtimeState.PROCESSING,
        ):
            self.reset_for_calibration(timestamp)
            return
        if key == 32 and self.state in (RealtimeState.RESULT, RealtimeState.FAILED):
            self.reset_for_calibration(timestamp)

    def run(self):
        self.start_source()
        self.start_model_loading()
        first_packet = self.read_source(timeout=3.0)
        if first_packet is None:
            raise RuntimeError("未读取到初始画面。")
        self.last_packet = first_packet
        self.create_window()

        try:
            while not self.should_stop:
                if not self.window_is_open():
                    self.should_stop = True
                    break
                if self.state == RealtimeState.LOADING_MODELS:
                    if self.model_error is not None:
                        raise RuntimeError(f"模型加载失败: {self.model_error}")
                    if self.models is not None:
                        self.reset_for_calibration(self.last_packet.timestamp)
                    elif not self.source.is_replay:
                        packet = self.read_source(
                            last_index=self.last_packet.index,
                            timeout=0.1,
                        )
                        if packet is not None:
                            self.last_packet = packet
                elif self.state == RealtimeState.PROCESSING:
                    if not self.source.is_replay:
                        packet = self.read_source(
                            last_index=self.last_packet.index,
                            timeout=0.1,
                        )
                        if packet is not None:
                            self.last_packet = packet
                    self.poll_processing(self.last_packet.timestamp)
                elif self.state in (RealtimeState.RESULT, RealtimeState.FAILED):
                    if self.args.auto_exit_on_result and self.state == RealtimeState.RESULT:
                        break
                    if self.args.headless and self.source.is_replay:
                        break
                    if not self.source.is_replay and self.source_error is None:
                        packet = self.read_source(
                            last_index=self.last_packet.index,
                            timeout=0.1,
                        )
                        if packet is not None:
                            self.last_packet = packet
                else:
                    packet = self.read_source(
                        last_index=self.last_packet.index,
                        timeout=0.2,
                    )
                    if packet is None:
                        if self.source.is_replay:
                            self.fail("回放视频已结束。", self.last_packet)
                        continue
                    self.last_packet = packet
                    if packet.index != self.last_processed_index:
                        self.last_processed_index = packet.index
                        self.handle_packet(packet)

                if not self.args.headless:
                    if not self.window_is_open():
                        self.should_stop = True
                        break
                    if self.state == RealtimeState.RESULT:
                        canvas = self.render_result_canvas()
                    else:
                        canvas = self.render_live_canvas(self.last_packet)
                    cv2.imshow(self.args.window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if not self.window_is_open():
                        self.should_stop = True
                        break
                    self.handle_key(key, self.last_packet.timestamp)
                else:
                    time.sleep(0.001)
        finally:
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join()
            self.source.close()
            cv2.destroyAllWindows()

        if self.state == RealtimeState.RESULT:
            return 0
        if self.state == RealtimeState.FAILED:
            return 2
        return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="ArUco 立定跳远实时测距",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头序号")
    parser.add_argument("--camera-width", type=int, default=1920, help="摄像头宽度")
    parser.add_argument("--camera-height", type=int, default=1080, help="摄像头高度")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="摄像头 FPS")
    parser.add_argument("--video-file", type=str, default=None, help="用视频回放代替摄像头")
    parser.add_argument("--device", type=str, default="cuda", help="SAM2 运行设备")
    parser.add_argument(
        "--rotate-frame",
        choices=("none", "clockwise", "counterclockwise", "180"),
        default="none",
        help="Rotate input frames before detection; useful for portrait phone cameras.",
    )
    parser.add_argument("--yolo-model", type=str, default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--keypoint-repo-root", type=str, default=str(DEFAULT_KEYPOINT_REPO_ROOT))
    parser.add_argument("--keypoint-weights-path", type=str, default=str(DEFAULT_KEYPOINT_WEIGHTS_PATH))
    parser.add_argument("--keypoint-bbox-expand-ratio", type=float, default=0.20)
    parser.add_argument("--rtoe-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--start-line-tolerance-cm", type=float, default=5.0)
    parser.add_argument("--sam2-checkpoint", type=str, default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2-config", type=str, default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--font", type=str, default=None, help="中文字体路径")
    parser.add_argument("--calib-stable-seconds", type=float, default=0.5)
    parser.add_argument("--calib-jitter-px", type=float, default=8.0)
    parser.add_argument("--ready-seconds", type=float, default=2.0)
    parser.add_argument("--ready-jitter-cm", type=float, default=20.0)
    parser.add_argument("--ready-x-min", type=float, default=-100.0)
    parser.add_argument("--ready-x-max", type=float, default=50.0)
    parser.add_argument("--ready-y-min", type=float, default=-100.0)
    parser.add_argument("--ready-y-max", type=float, default=100.0)
    parser.add_argument("--takeoff-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-jump-seconds", type=float, default=8.0)
    parser.add_argument("--takeoff-min-px", type=float, default=30.0)
    parser.add_argument("--takeoff-image-ratio", type=float, default=0.03)
    parser.add_argument("--landing-stable-frames", type=int, default=5)
    parser.add_argument("--landing-max-change-px", type=float, default=5.0)
    parser.add_argument("--min-forward-cm", type=float, default=30.0)
    parser.add_argument("--max-lost-frames", type=int, default=5)
    parser.add_argument("--max-track-distance-px", type=float, default=500.0)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.08)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--window-name", type=str, default="ArUco Realtime Jump")
    parser.add_argument("--window-width", type=int, default=1440, help="界面窗口宽度")
    parser.add_argument("--window-height", type=int, default=900, help="界面窗口高度")
    parser.add_argument("--panel-width", type=int, default=400, help="右侧状态栏宽度")
    parser.add_argument("--save-clip", action="store_true", default=True)
    parser.add_argument("--no-save-clip", action="store_false", dest="save_clip")
    parser.add_argument("--headless", action="store_true", help="无窗口运行，便于回放测试")
    parser.add_argument(
        "--auto-exit-on-result",
        action="store_true",
        help="生成首个成绩后自动退出，便于回放测试",
    )
    return parser


def main():
    args = build_parser().parse_args()
    try:
        return RealtimeJumpApp(args).run()
    except Exception as exc:
        print(f"错误: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
