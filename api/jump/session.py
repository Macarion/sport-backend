from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .model_runtime import get_algorithms, get_model_bundle


@dataclass
class FastReadyConfig:
    calib_stable_seconds: float = 0.5
    calib_jitter_px: float = 8.0
    auto_ready_frames: int = 3
    ready_x_min: float = -100.0
    ready_x_max: float = 50.0
    ready_y_min: float = -100.0
    ready_y_max: float = 100.0
    takeoff_timeout_seconds: float = 10.0
    max_jump_seconds: float = 8.0
    takeoff_min_px: float = 30.0
    takeoff_image_ratio: float = 0.03
    landing_stable_frames: int = 3
    landing_max_change_px: float = 15.0
    min_forward_cm: float = 30.0
    max_lost_frames: int = 20
    max_track_distance_px: float = 900.0
    bbox_expand_ratio: float = 0.08
    raw_buffer_seconds: float = 8.0
    # 原始帧缓冲的内存上限（MB）。按数量(秒×fps)与内存双重封顶，谁先到先淘汰最旧帧，
    # 防止高分辨率下整帧缓存把内存吃爆（240 帧 × HD ≈ 1.5GB）。
    raw_buffer_max_mb: float = 512.0
    expected_frame_fps: float = 30.0
    review_before_takeoff_seconds: float = 0.5
    post_takeoff_record_seconds: float = 2.0
    review_frame_step: int = 1
    landing_min_frames_after_peak: int = 1
    landing_min_recovery_px: float = 40.0
    stream_analysis_fps: float = 8.0
    # 等待/检测起跳阶段加密分析帧率：让“连续两帧越阈值”的起跳判定有足够密的样本，
    # 可靠捕捉起跳瞬间（单飞机制会自然把它限制在 GPU 实际能力以内，是上限不是强制值）。
    armed_analysis_fps: float = 16.0

    @classmethod
    def from_settings(cls, settings_obj=None) -> "FastReadyConfig":
        """从 Django settings 读取覆盖值：每个字段对应 settings.JUMP_<字段大写>。

        未在 settings 中配置的字段沿用此处默认值，因此默认行为完全不变。
        """
        config = cls()
        if settings_obj is None:
            from django.conf import settings as settings_obj
        for field in fields(cls):
            key = f"JUMP_{field.name.upper()}"
            if not hasattr(settings_obj, key):
                continue
            value = getattr(settings_obj, key)
            default_val = getattr(config, field.name)
            try:
                if isinstance(default_val, bool):
                    value = bool(value)
                elif isinstance(default_val, int):
                    value = int(value)
                elif isinstance(default_val, float):
                    value = float(value)
            except (TypeError, ValueError):
                continue
            setattr(config, field.name, value)
        return config


@dataclass
class RawWebFrame:
    frame_id: int
    timestamp_ms: int
    timestamp_sec: float
    # 缓存 JPEG 编码字节而非原始 ndarray：4K 帧原始 ~24MB/帧会迅速撞上内存上限，
    # 把离线复核窗口所需的帧挤掉；JPEG（约 1-2MB/帧）可在上限内保留完整窗口。
    encoded: bytes
    nbytes: int

    def decode(self):
        import cv2

        buffer = np.frombuffer(self.encoded, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


class FastReadyWebSession:
    """HTTP-fed version of the fast_ready realtime state machine."""

    STATE_ORDER = {
        "CALIBRATING": 0,
        "WAITING_READY": 1,
        "ARMED": 2,
        "IN_JUMP": 3,
        "PROCESSING": 4,
        "RESULT": 5,
    }
    TERMINAL_STATES = {"RESULT", "FAILED", "STOPPED"}

    def __init__(
        self,
        uid: str,
        output_root: Path,
        temp_root: Optional[Path] = None,
        config: Optional[FastReadyConfig] = None,
    ):
        self.uid = str(uid)
        self.session_id = uuid.uuid4().hex
        self.config = config or FastReadyConfig()
        self.output_dir = output_root / self.uid / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_root = temp_root or output_root.parent / "temp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.diag_log_path = self.temp_root / (
            f"jump_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{self._safe_filename_part(self.uid)}_{self.session_id}.jsonl"
        )

        self.lock = threading.Lock()
        # 单独保护原始帧环形缓冲：入缓冲只用这把锁，绝不与跑 YOLO 的 self.lock 竞争，
        # 这样 WebRTC 事件循环上的 ingest 永远不会被分析线程的长时间持锁卡住。
        self.raw_lock = threading.Lock()
        self.state = "CALIBRATING"
        # created_at 保留墙钟（仅用于会话寿命/清理，不参与状态计时）。
        self.created_at = time.time()
        # state_started_at 统一用“帧时钟”，在收到第一帧时按该帧时间戳懒初始化，
        # 不再混入墙钟，确保所有 elapsed 判定都和 timestamp_ms 同一基准。
        self.state_started_at: Optional[float] = None
        self.last_frame_id: Optional[int] = None
        self.last_timestamp_ms: Optional[int] = None

        self.calib_tracker = None
        self.calibration_path: Optional[Path] = None
        self.calibration_debug_path: Optional[Path] = None
        self.calibration_attempt_path: Optional[Path] = None
        self.calibration_attempt_debug_path: Optional[Path] = None
        self.plane = None

        self.fast_ready_samples = deque(maxlen=max(1, self.config.auto_ready_frames))
        self.baseline_y2: Optional[float] = None
        self.ready_world_point: Optional[np.ndarray] = None
        self.locked_bbox: Optional[List[int]] = None
        self.takeoff_y2_values = deque(maxlen=2)
        self.lost_person_frames = 0
        self.jump_start_frame_index: Optional[int] = None
        self.takeoff_timestamp_sec: Optional[float] = None
        raw_buffer_frames = max(
            1,
            int(
                max(1.0, float(self.config.raw_buffer_seconds))
                * max(1.0, float(self.config.expected_frame_fps))
            ),
        )
        self.raw_frames = deque(maxlen=raw_buffer_frames)
        self.raw_frames_appended = 0
        self.raw_frames_dropped = 0
        self.raw_frames_bytes = 0
        self.raw_buffer_max_bytes = int(
            max(1.0, float(self.config.raw_buffer_max_mb)) * 1024 * 1024
        )
        # 实时(WebRTC)入缓冲路径：事件循环只把原始帧塞进这个队列，由后台 encoder 线程做
        # JPEG 编码 + 入缓冲，避免在事件循环上做编码而拖慢 aiortc 收发导致传输层丢帧。
        # 队列有界：encoder 跟得上时几乎为空；万一持续过载则丢最新帧（计入 raw_encode_dropped），
        # 用有界内存换取“绝不阻塞事件循环”。HTTP 路径仍走同步编码，不经此队列。
        self._raw_encode_queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=32)
        self.raw_encode_dropped = 0
        self._raw_encoder = threading.Thread(
            target=self._raw_encoder_loop,
            name=f"jumpenc-{self.session_id[:8]}",
            daemon=True,
        )
        self._raw_encoder.start()
        self.jump_samples: List[Any] = []
        self.review_samples: List[Any] = []
        self.trigger_landing_sample = None
        self.landing_sample = None
        self.peak_sample = None
        self.review_strategy: Optional[str] = None
        self.review_fallback_used = False
        self.review_summary: Dict[str, Any] = {}
        self.stream_received_frames = 0
        self.stream_analyzed_frames = 0
        self.stream_started_at: Optional[float] = None
        self.stream_last_frame_at: Optional[float] = None
        self.stream_last_analysis_at: Optional[float] = None
        self.stream_last_response_at: Optional[float] = None
        self._stream_frame_id = 0

        self.processing_thread: Optional[threading.Thread] = None
        self.processing_error: Optional[str] = None
        self.score_cm: Optional[float] = None
        self.result_payload: Optional[Dict[str, Any]] = None

        self.message = "请保持 ArUco 标记可见"
        self.error: Optional[str] = None
        self.pause_frame_id: Optional[int] = None
        self.last_painting: List[Dict[str, Any]] = []
        self.debug_info: Dict[str, Any] = {}
        self.frame_diagnostics: Dict[str, Any] = {}
        self._diag_index = 0
        self._diag_lock = threading.Lock()
        self._diag_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._diag_writer = threading.Thread(
            target=self._diag_writer_loop,
            name=f"jumpdiag-{self.session_id[:8]}",
            daemon=True,
        )
        self._diag_writer.start()
        self._write_diag_event(
            {
                "event": "session_start",
                "event_zh": "会话开始",
                "reason_zh": "创建或重置测试会话，等待前端上传视频帧",
                "state": self.state,
                "message": self.message,
                "log_path": str(self.diag_log_path),
            }
        )

    def process_frame(self, frame, frame_id: int, timestamp_ms: int, stop: bool = False) -> Dict[str, Any]:
        started_at = time.perf_counter()
        frame_id = int(frame_id)
        timestamp_ms = int(timestamp_ms)
        with self.lock:
            previous_timestamp_ms = self.last_timestamp_ms
            self.frame_diagnostics = self._new_frame_diagnostics()
            try:
                response = self._process_frame_locked(frame, frame_id, timestamp_ms, stop)
            except Exception as exc:
                self.frame_diagnostics["exception"] = str(exc)
                process_ms = (time.perf_counter() - started_at) * 1000.0
                self._write_frame_diag_log(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    previous_timestamp_ms=previous_timestamp_ms,
                    process_ms=process_ms,
                    response=self._response(frame_id=frame_id, message=f"处理视频帧异常: {exc}"),
                    event_zh="处理视频帧异常",
                    reason_zh=str(exc),
                )
                raise

            process_ms = (time.perf_counter() - started_at) * 1000.0
            self._write_frame_diag_log(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                previous_timestamp_ms=previous_timestamp_ms,
                process_ms=process_ms,
                response=response,
            )
            return response

    def process_stream_frame(
        self,
        frame,
        timestamp_ms: Optional[int] = None,
        frame_id: Optional[int] = None,
        analysis_fps: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible helper: ingest the frame and analyze inline if due.

        实时 WebRTC 消费方应改用 ``ingest_stream_frame`` + ``stream_analysis_due`` +
        ``analyze_stream_frame``，把耗时的分析放到线程池里执行，避免阻塞事件循环。
        """
        timestamp_ms = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
        if frame_id is None:
            frame_id = self.next_stream_frame_id()
        frame_id = int(frame_id)
        self.ingest_stream_frame(frame, frame_id=frame_id, timestamp_ms=timestamp_ms)
        if not self.stream_analysis_due(timestamp_ms, analysis_fps=analysis_fps):
            return None
        return self.analyze_stream_frame(frame, frame_id=frame_id, timestamp_ms=timestamp_ms)

    def next_stream_frame_id(self) -> int:
        with self.lock:
            self._stream_frame_id += 1
            return self._stream_frame_id

    def ingest_stream_frame(self, frame, frame_id: int, timestamp_ms: int) -> None:
        """Cheap, runs on the event loop: just buffer the raw frame to keep it dense.

        只使用 raw_lock，绝不获取 self.lock，因此不会被分析线程跑 YOLO 时的长持锁阻塞。
        stream 计数仅用于诊断，轻微竞争可接受。
        """
        timestamp_ms = int(timestamp_ms)
        timestamp_sec = timestamp_ms / 1000.0
        if self.stream_started_at is None:
            self.stream_started_at = timestamp_sec
        self.stream_received_frames += 1
        self.stream_last_frame_at = timestamp_sec
        if self.state in self.TERMINAL_STATES:
            return
        # 不在事件循环上编码：仅把帧引用塞进队列（to_ndarray 已是我方独占数组，无需 copy），
        # 由后台 encoder 线程编码+入缓冲。队列满（持续过载）时丢最新帧并计数，绝不阻塞循环。
        try:
            self._raw_encode_queue.put_nowait((frame, int(frame_id), timestamp_ms))
        except queue.Full:
            self.raw_encode_dropped += 1

    def _raw_encoder_loop(self) -> None:
        while True:
            item = self._raw_encode_queue.get()
            if item is None:
                break
            frame, frame_id, timestamp_ms = item
            try:
                self._append_raw_frame(frame, frame_id, timestamp_ms)
            except Exception as exc:
                print(f"[jump raw encoder ignored] {exc}")

    def stream_analysis_due(self, timestamp_ms: int, analysis_fps: Optional[float] = None) -> bool:
        timestamp_sec = int(timestamp_ms) / 1000.0
        with self.lock:
            if self.state in self.TERMINAL_STATES:
                return False
            if analysis_fps is not None:
                fps = float(analysis_fps)
            elif self.state in {"WAITING_READY", "ARMED"}:
                # 等待/检测起跳阶段加密采样，避免在线“连续两帧”规则错过起跳瞬间。
                fps = float(self.config.armed_analysis_fps)
            else:
                fps = float(self.config.stream_analysis_fps)
            analysis_interval = 1.0 / max(0.1, fps)
            return (
                self.stream_last_analysis_at is None
                or timestamp_sec - float(self.stream_last_analysis_at) >= analysis_interval
                or self.state in {"PROCESSING", "RESULT", "FAILED"}
            )

    def analyze_stream_frame(
        self,
        frame,
        frame_id: int,
        timestamp_ms: int,
    ) -> Optional[Dict[str, Any]]:
        """Heavy path (YOLO 等): MUST be called off the event loop (executor/worker)."""
        frame_id = int(frame_id)
        timestamp_ms = int(timestamp_ms)
        timestamp_sec = timestamp_ms / 1000.0
        started_at = time.perf_counter()
        with self.lock:
            if self.state in self.TERMINAL_STATES:
                return self._response(frame_id=self.last_frame_id)

            previous_timestamp_ms = self.last_timestamp_ms
            self.frame_diagnostics = self._new_frame_diagnostics()
            self.frame_diagnostics["stream_received_frames"] = self.stream_received_frames
            self.frame_diagnostics["stream_analyzed_frames"] = self.stream_analyzed_frames + 1
            self.frame_diagnostics["stream_stats"] = self._stream_stats(timestamp_sec)
            try:
                response = self._process_frame_locked(
                    frame,
                    frame_id,
                    timestamp_ms,
                    stop=False,
                    append_raw=False,
                )
            except Exception as exc:
                self.frame_diagnostics["exception"] = str(exc)
                process_ms = (time.perf_counter() - started_at) * 1000.0
                self._write_frame_diag_log(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    previous_timestamp_ms=previous_timestamp_ms,
                    process_ms=process_ms,
                    response=self._response(frame_id=frame_id, message=f"处理 WebRTC 帧异常: {exc}"),
                    event_zh="处理 WebRTC 帧异常",
                    reason_zh=str(exc),
                )
                raise

            self.stream_analyzed_frames += 1
            self.stream_last_analysis_at = timestamp_sec
            self.stream_last_response_at = timestamp_sec
            process_ms = (time.perf_counter() - started_at) * 1000.0
            self._write_frame_diag_log(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                previous_timestamp_ms=previous_timestamp_ms,
                process_ms=process_ms,
                response=response,
                event_zh="处理 WebRTC 分析帧",
            )
            return response

    def _process_frame_locked(
        self,
        frame,
        frame_id: int,
        timestamp_ms: int,
        stop: bool,
        append_raw: bool = True,
    ) -> Dict[str, Any]:
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            self.frame_diagnostics["reason_zh"] = "重复或乱序帧，状态机未重复处理"
            return self._response(frame_id=frame_id, message="重复或乱序帧，已忽略")

        if self.state in self.TERMINAL_STATES:
            self.frame_diagnostics["reason_zh"] = "测试已处于终态，忽略后续视频帧"
            return self._response(frame_id=self.last_frame_id, message=self.message)

        self.last_frame_id = int(frame_id)
        self.last_timestamp_ms = int(timestamp_ms)
        if self.state_started_at is None:
            # 用第一帧的帧时钟锚定初始 CALIBRATING 起始时间，与后续 _transition 同基准。
            self.state_started_at = int(timestamp_ms) / 1000.0
        if append_raw:
            self._append_raw_frame(frame, frame_id, timestamp_ms)

        if stop and self.state not in self.TERMINAL_STATES:
            self._set_failed("视频流已结束，测试未完成")
            self.frame_diagnostics["reason_zh"] = "收到 stop=true，测试未完成，置为失败"
            return self._response(frame_id=frame_id)

        if self.state == "CALIBRATING":
            return self._handle_calibrating(frame, frame_id, timestamp_ms)
        if self.state == "WAITING_READY":
            return self._handle_waiting_ready(frame, frame_id, timestamp_ms)
        if self.state == "ARMED":
            return self._handle_armed(frame, frame_id, timestamp_ms)
        if self.state == "IN_JUMP":
            return self._handle_in_jump(frame, frame_id, timestamp_ms)
        if self.state == "PROCESSING":
            self.frame_diagnostics["reason_zh"] = "成绩计算中，忽略后续视频帧"
            return self._response(frame_id=frame_id, message="成绩计算中")
        return self._response(frame_id=frame_id)

    def status_response(self) -> Dict[str, Any]:
        with self.lock:
            response = self._response(frame_id=self.last_frame_id)
            self._write_diag_event(
                {
                    "event": "status",
                    "event_zh": "状态查询",
                    "reason_zh": response.get("message"),
                    "frame_id": self.last_frame_id,
                    "state": response.get("state"),
                    "message": response.get("message"),
                    "error": response.get("error"),
                    "score_cm": response.get("score_cm"),
                    "state_elapsed_s": self._status_state_elapsed_s(),
                }
            )
            return response

    def to_response(self, message: Optional[str] = None) -> Dict[str, Any]:
        return self._response(frame_id=self.last_frame_id, message=message)

    def fail(self, message: str, frame_id: Optional[Any] = None) -> Dict[str, Any]:
        with self.lock:
            self._set_failed(message)
            response = self._response(frame_id=frame_id)
            self._write_diag_event(
                {
                    "event": "session_failed",
                    "event_zh": "会话失败",
                    "reason_zh": message,
                    "frame_id": frame_id,
                    "state": self.state,
                    "message": self.message,
                    "error": self.error,
                }
            )
            return response

    def stop(self, message: str = "测试已主动停止") -> Dict[str, Any]:
        with self.lock:
            if self.state not in self.TERMINAL_STATES:
                self.state = "STOPPED"
                self.message = message
                self.error = None
                self.pause_frame_id = self.last_frame_id
                # 终态时间也用帧时钟，避免与 last_timestamp_ms 混算出垃圾 elapsed。
                if self.last_timestamp_ms is not None:
                    self.state_started_at = int(self.last_timestamp_ms) / 1000.0
            response = self._response(frame_id=self.last_frame_id, message=self.message)
            self._write_diag_event(
                {
                    "event": "session_stopped",
                    "event_zh": "会话主动停止",
                    "reason_zh": self.message,
                    "frame_id": self.last_frame_id,
                    "state": self.state,
                    "message": self.message,
                    "error": self.error,
                }
            )
            self._close_background_workers()
            return response

    def _close_background_workers(self) -> None:
        try:
            self._raw_encode_queue.put(None, timeout=0.1)
        except Exception:
            pass
        try:
            self._diag_queue.put_nowait(None)
        except Exception:
            pass

    def _handle_calibrating(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
        import cv2

        algos = get_algorithms()
        realtime = algos.realtime_core
        if self.calib_tracker is None:
            self.calib_tracker = realtime.CalibrationStabilityTracker(
                stable_seconds=self.config.calib_stable_seconds,
                jitter_px=self.config.calib_jitter_px,
            )

        detections = algos.detect_aruco_markers(frame)
        missing_ids = [
            marker_id
            for marker_id in algos.realtime_core.REQUIRED_ARUCO_IDS
            if marker_id not in detections
        ]
        self.last_painting = self._aruco_painting(detections)
        if frame_id == 1 or frame_id % 10 == 0:
            self._save_calibration_attempt(frame, detections, missing_ids)

        timestamp_sec = timestamp_ms / 1000.0
        if not self.calib_tracker.update(detections, timestamp_sec):
            detected_ids = sorted(int(marker_id) for marker_id in detections.keys())
            if missing_ids:
                self.message = (
                    f"场景检测中：已识别 ID {detected_ids}，"
                    f"缺失 {missing_ids}。请检查定帧画面/分辨率/清晰度"
                )
            else:
                elapsed = self.calib_tracker.elapsed(timestamp_sec)
                self.message = (
                    f"ArUco 四码已识别，等待稳定 "
                    f"{elapsed:.1f}/{self.config.calib_stable_seconds:.1f}s"
                )
            return self._response(frame_id=frame_id)

        self.calibration_path = self.output_dir / "calib_frame.jpg"
        self.calibration_debug_path = self.output_dir / "calib_aruco_debug.jpg"
        cv2.imwrite(str(self.calibration_path), frame)
        self.plane = algos.build_aruco_plane(frame, debug_path=self.calibration_debug_path)
        self._transition("WAITING_READY", timestamp_sec)
        self.message = "场景检测完成，请测试人进入起跳准备区"
        self.last_painting = self._aruco_painting(detections)
        return self._response(frame_id=frame_id)

    def _save_calibration_attempt(self, frame, detections, missing_ids) -> None:
        import cv2

        algos = get_algorithms()
        self.calibration_attempt_path = self.output_dir / "calib_latest_attempt.jpg"
        self.calibration_attempt_debug_path = self.output_dir / "calib_latest_attempt_aruco_debug.jpg"
        cv2.imwrite(str(self.calibration_attempt_path), frame)
        algos.realtime_core.draw_aruco_debug(
            frame,
            detections,
            self.calibration_attempt_debug_path,
            missing_ids=missing_ids,
        )

    def _handle_waiting_ready(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
        algos = get_algorithms()
        realtime = algos.realtime_core
        models = get_model_bundle()
        detections = realtime.detect_people(models.yolo_model, frame, self.plane)
        selected = self._select_fast_ready_person(detections)
        self._set_detection_diagnostics(
            detections,
            selected,
            match_method="ready_selection" if selected is not None else "none",
            match_fallback_used=False,
        )

        self.last_painting = self._people_painting(detections, selected)
        if selected is None or selected.bottom_center_world is None:
            self.fast_ready_samples.clear()
            self.message = "等待测试人进入画面"
            self.frame_diagnostics["reason_zh"] = "起跳准备阶段未检测到可用测试人"
            return self._response(frame_id=frame_id)

        if self.fast_ready_samples:
            previous = self.fast_ready_samples[-1].bbox
            matched = realtime.match_locked_person(
                [selected],
                previous,
                max_center_distance_px=self.config.max_track_distance_px,
            )
            if matched is None:
                self.fast_ready_samples.clear()

        self.fast_ready_samples.append(selected)
        needed = max(1, int(self.config.auto_ready_frames))
        if len(self.fast_ready_samples) < needed:
            self.message = f"检测到测试人，准备中 {len(self.fast_ready_samples)}/{needed}"
            self.frame_diagnostics["reason_zh"] = "测试人已检测到，等待连续准备帧满足要求"
            return self._response(frame_id=frame_id)

        points = np.array(
            [sample.bottom_center_world for sample in self.fast_ready_samples],
            dtype=np.float32,
        )
        self.baseline_y2 = float(np.median([sample.y2 for sample in self.fast_ready_samples]))
        self.ready_world_point = np.median(points, axis=0)
        self.locked_bbox = list(self.fast_ready_samples[-1].bbox)
        self.takeoff_y2_values.clear()
        self._transition("ARMED", timestamp_ms / 1000.0)
        self.message = "可以开始起跳"
        self.frame_diagnostics["reason_zh"] = "准备帧满足要求，进入可以起跳状态"
        return self._response(frame_id=frame_id)

    def _handle_armed(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
        algos = get_algorithms()
        realtime = algos.realtime_core
        models = get_model_bundle()
        timestamp_sec = timestamp_ms / 1000.0

        detections = realtime.detect_people(models.yolo_model, frame, self.plane)
        active = self._match_active_person(
            detections,
            self.locked_bbox,
        )
        self._set_detection_diagnostics(detections, active)
        self.last_painting = self._people_painting(detections, active)

        if timestamp_sec - self.state_started_at > self.config.takeoff_timeout_seconds:
            self._set_failed("等待起跳超时")
            self.frame_diagnostics["reason_zh"] = "等待起跳超时"
            return self._response(frame_id=frame_id)

        if active is None:
            self.lost_person_frames += 1
            # 人物丢失时清空起跳历史，确保“连续两帧越阈值”是连续“检测到”的两帧，
            # 而不是跨越一次丢失把丢失前后的样本错误地拼成连续。
            self.takeoff_y2_values.clear()
            self.frame_diagnostics["reason_zh"] = "起跳阶段人物丢失"
            if self.lost_person_frames > self.config.max_lost_frames:
                self._set_failed("准备起跳阶段连续丢失测试人")
            else:
                self.message = (
                    f"等待起跳：暂时丢失测试人 "
                    f"{self.lost_person_frames}/{self.config.max_lost_frames}"
                )
            return self._response(frame_id=frame_id)

        self.lost_person_frames = 0
        self.locked_bbox = list(active.bbox)
        self.takeoff_y2_values.append(active.y2)
        takeoff_threshold = max(
            float(self.config.takeoff_min_px),
            float(frame.shape[0]) * float(self.config.takeoff_image_ratio),
        )
        self.debug_info = {
            "phase": "takeoff_detection",
            "baseline_y2": float(self.baseline_y2),
            "active_y2": float(active.y2),
            "takeoff_delta_px": float(self.baseline_y2 - active.y2),
            "takeoff_threshold_px": takeoff_threshold,
            "recent_y2": [float(value) for value in self.takeoff_y2_values],
        }
        self.frame_diagnostics.update(self.debug_info)
        if realtime.detect_takeoff(
            self.takeoff_y2_values,
            baseline_y2=float(self.baseline_y2),
            frame_height=frame.shape[0],
            min_pixels=self.config.takeoff_min_px,
            image_ratio=self.config.takeoff_image_ratio,
        ):
            self.jump_samples = []
            self.review_samples = []
            self.review_summary = {}
            self.jump_start_frame_index = frame_id
            self.takeoff_timestamp_sec = timestamp_sec
            self._append_jump_sample(frame, frame_id, timestamp_ms, active)
            self._transition("IN_JUMP", timestamp_sec)
            self.message = "起跳后录制中 0.0/{:.1f}s".format(
                float(self.config.post_takeoff_record_seconds)
            )
            self.frame_diagnostics["jump_samples_count"] = len(self.jump_samples)
            self.frame_diagnostics["reason_zh"] = "起跳条件满足，进入跳跃中"
            return self._response(frame_id=frame_id)

        self.message = "可以开始起跳"
        self.frame_diagnostics["reason_zh"] = "起跳条件未满足：最近两帧未同时超过阈值"
        return self._response(frame_id=frame_id)

    def _handle_in_jump(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
        timestamp_sec = timestamp_ms / 1000.0
        elapsed = max(0.0, timestamp_sec - float(self.state_started_at))
        target = max(0.1, float(self.config.post_takeoff_record_seconds))
        self.frame_diagnostics["post_takeoff_elapsed_s"] = elapsed
        self.frame_diagnostics["post_takeoff_record_seconds"] = target
        self.frame_diagnostics["raw_buffer"] = self._raw_buffer_stats()
        self.frame_diagnostics["stream_stats"] = self._stream_stats(timestamp_sec)

        if elapsed < target:
            self.message = f"起跳后录制中 {elapsed:.1f}/{target:.1f}s"
            self.frame_diagnostics["reason_zh"] = "起跳后固定窗口录制中，暂不在线判定落地"
            return self._response(frame_id=frame_id)

        self._start_processing(timestamp_sec)
        self.frame_diagnostics["landing_reason_zh"] = "起跳后固定录制窗口完成，进入离线落地帧复核"
        self.frame_diagnostics["reason_zh"] = "起跳后固定录制窗口完成，进入离线落地帧复核"
        return self._response(frame_id=frame_id, message="结果生成中")

    def _append_raw_frame(self, frame, frame_id: int, timestamp_ms: int) -> None:
        import cv2

        # 在锁外做 JPEG 编码（替代整帧拷贝）：既避免持 raw_lock 期间做重活，又把每帧内存
        # 从原始 ndarray 的几十 MB 压到 1-2MB，使 512MB 上限能容纳完整的离线复核窗口。
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not ok:
            return
        encoded = buffer.tobytes()
        packet = RawWebFrame(
            frame_id=int(frame_id),
            timestamp_ms=int(timestamp_ms),
            timestamp_sec=int(timestamp_ms) / 1000.0,
            encoded=encoded,
            nbytes=len(encoded),
        )
        frame_bytes = packet.nbytes
        with self.raw_lock:
            # 内存上限：先淘汰最旧帧直到放得下新帧（始终保留至少当前这一帧）。
            while self.raw_frames and (
                self.raw_frames_bytes + frame_bytes > self.raw_buffer_max_bytes
            ):
                self._evict_oldest_raw_frame_locked()
            # 数量上限：deque 满时 append 会自动丢最旧，这里手动先丢以便正确记账。
            if self.raw_frames.maxlen and len(self.raw_frames) >= self.raw_frames.maxlen:
                self._evict_oldest_raw_frame_locked()
            self.raw_frames.append(packet)
            self.raw_frames_bytes += frame_bytes
            self.raw_frames_appended += 1

    def _evict_oldest_raw_frame_locked(self) -> None:
        # 调用方必须已持有 self.raw_lock。
        evicted = self.raw_frames.popleft()
        self.raw_frames_bytes -= int(evicted.nbytes)
        if self.raw_frames_bytes < 0:
            self.raw_frames_bytes = 0
        self.raw_frames_dropped += 1

    def _raw_frames_between(self, start_time: float, end_time: float, step: int = 1) -> List[RawWebFrame]:
        step = max(1, int(step))
        with self.raw_lock:
            frames = [
                packet
                for packet in self.raw_frames
                if float(start_time) <= float(packet.timestamp_sec) <= float(end_time)
            ]
        return frames[::step]

    def _raw_buffer_stats(self) -> Dict[str, Any]:
        max_mb = round(self.raw_buffer_max_bytes / (1024 * 1024), 1)
        with self.raw_lock:
            if not self.raw_frames:
                return {
                    "size": 0,
                    "maxlen": self.raw_frames.maxlen,
                    "appended": self.raw_frames_appended,
                    "dropped": self.raw_frames_dropped,
                    "encode_dropped": self.raw_encode_dropped,
                    "used_mb": 0.0,
                    "max_mb": max_mb,
                }
            oldest = self.raw_frames[0]
            latest = self.raw_frames[-1]
            size = len(self.raw_frames)
            maxlen = self.raw_frames.maxlen
            appended = self.raw_frames_appended
            dropped = self.raw_frames_dropped
            used_bytes = self.raw_frames_bytes
        span = max(0.001, float(latest.timestamp_sec) - float(oldest.timestamp_sec))
        fps = (size - 1) / span if size >= 2 else None
        return {
            "size": size,
            "maxlen": maxlen,
            "appended": appended,
            "dropped": dropped,
            "oldest_frame_id": int(oldest.frame_id),
            "latest_frame_id": int(latest.frame_id),
            "fps_estimate": fps,
            "used_mb": round(used_bytes / (1024 * 1024), 1),
            "max_mb": max_mb,
            "encode_dropped": self.raw_encode_dropped,
        }

    def _stream_stats(self, now_sec: Optional[float] = None) -> Dict[str, Any]:
        now_sec = float(now_sec if now_sec is not None else time.time())
        span = None
        received_fps = None
        analysis_fps = None
        if self.stream_started_at is not None:
            span = max(0.001, now_sec - float(self.stream_started_at))
            received_fps = self.stream_received_frames / span
            analysis_fps = self.stream_analyzed_frames / span
        return {
            "received_frames": self.stream_received_frames,
            "analyzed_frames": self.stream_analyzed_frames,
            "received_fps": received_fps,
            "analysis_fps": analysis_fps,
            "buffer_size": len(self.raw_frames),
            "buffer_maxlen": self.raw_frames.maxlen,
        }

    def _append_jump_sample(self, frame, frame_id: int, timestamp_ms: int, detection) -> None:
        realtime = get_algorithms().realtime_core
        packet = realtime.FramePacket(
            index=int(frame_id),
            timestamp=timestamp_ms / 1000.0,
            image=frame.copy(),
        )
        self.jump_samples.append(
            realtime.JumpFrameSample(
                packet=packet,
                detection=detection,
                encoded_frame=None,
            )
        )

    def _start_processing(self, timestamp_sec: float) -> None:
        if self.processing_thread is not None:
            return
        self._transition("PROCESSING", timestamp_sec)
        self.pause_frame_id = (
            self.trigger_landing_sample.packet.index if self.trigger_landing_sample else self.last_frame_id
        )
        self.message = "结果生成中"
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()

    def _processing_worker(self) -> None:
        try:
            result = self._run_measurement()
            with self.lock:
                if self.state == "FAILED":
                    return
                self.result_payload = result
                self.score_cm = float(result["score_cm"])
                self.error = None
                self._transition("RESULT", time.time())
                self.message = "成绩计算完成"
                self.last_painting = result.get("painting", self.last_painting)
                self._write_diag_event(
                    {
                        "event": "processing_result",
                        "event_zh": "后台成绩计算完成",
                        "reason_zh": "成绩计算完成，已进入 RESULT",
                        "frame_id": self.last_frame_id,
                        "state": self.state,
                        "message": self.message,
                        "score_cm": self.score_cm,
                        "landing_frame_id": result.get("landing_frame_id"),
                        "outputs": result.get("outputs"),
                    }
                )
        except Exception as exc:
            with self.lock:
                self.processing_error = str(exc)
                self._set_failed(f"成绩计算失败: {exc}")
                self._write_diag_event(
                    {
                        "event": "processing_failed",
                        "event_zh": "后台成绩计算失败",
                        "reason_zh": str(exc),
                        "frame_id": self.last_frame_id,
                        "state": self.state,
                        "message": self.message,
                        "error": self.error,
                    }
                )

    def _run_measurement(self) -> Dict[str, Any]:
        import cv2

        algos = get_algorithms()
        realtime = algos.realtime_core
        models = get_model_bundle()

        landing_sample = self._select_final_landing_sample_from_raw_buffer()
        landing_frame = landing_sample.packet.image.copy()
        landing_bbox = list(landing_sample.detection.bbox)
        with self.lock:
            self.landing_sample = landing_sample

        expanded_bbox = realtime.expand_box(
            landing_bbox,
            landing_frame.shape,
            self.config.bbox_expand_ratio,
        )
        bbox_info = {
            "box": expanded_bbox,
            "original_box": landing_bbox,
            "confidence": float(landing_sample.detection.confidence),
            "detected_person_count": None,
            "source": "wcxweb_realtime_yolo",
        }

        landing_path = self.output_dir / "landing_frame.jpg"
        cv2.imwrite(str(landing_path), landing_frame)
        segmentation_info = realtime.segment_with_loaded_predictor(
            models=models,
            landing_frame=landing_frame,
            prompt_box=expanded_bbox,
        )
        output_paths = realtime.save_segmentation_outputs(
            landing_frame,
            segmentation_info["mask"],
            bbox_info,
            self.output_dir,
        )
        measurement = algos.measure_jump_from_files(
            image_path=landing_path,
            mask_path=output_paths["mask_path"],
            output_dir=self.output_dir,
            calib_image_path=self.calibration_path,
        )
        result_path = self.output_dir / "result_aruco.txt"
        algos.write_result_file(
            result_path,
            measurement,
            image_path=landing_path,
            mask_path=output_paths["mask_path"],
            calib_image_path=self.calibration_path,
        )

        # 把“真正用于测距的落地帧”编码成缩略图随结果回传，让前端出结果时把它画成底图，
        # 叠加 painting（脚跟/人框/成绩），而不是停在 2 秒录制末帧上只冒一个红点。
        landing_preview = None
        try:
            import base64

            preview_img = landing_frame
            height, width = landing_frame.shape[:2]
            # 限制在 720 宽 / q72：base64 后约几十 KB，稳妥落在 WebRTC DataChannel 单条消息上限内；
            # 前端画布是原生分辨率，预览图会拉伸铺满，painting 坐标仍按原生像素对齐。
            max_width = 720
            if width > max_width:
                scale = max_width / float(width)
                preview_img = cv2.resize(
                    landing_frame,
                    (max_width, max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            ok_preview, preview_buf = cv2.imencode(
                ".jpg", preview_img, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
            )
            if ok_preview:
                landing_preview = "data:image/jpeg;base64," + base64.b64encode(
                    preview_buf.tobytes()
                ).decode("ascii")
        except Exception as exc:
            print(f"[jump landing preview skipped] {exc}")

        payload = {
            "uid": self.uid,
            "session_id": self.session_id,
            "state": "RESULT",
            "score_cm": float(measurement.score_cm),
            "landing_frame_id": int(landing_sample.packet.index),
            "landing_frame_preview": landing_preview,
            "landing_frame_size": [
                int(landing_frame.shape[1]),
                int(landing_frame.shape[0]),
            ],
            "heel_image_point": [
                float(measurement.heel_result.image_point[0]),
                float(measurement.heel_result.image_point[1]),
            ],
            "heel_world_point": [
                float(measurement.heel_result.world_point[0]),
                float(measurement.heel_result.world_point[1]),
            ],
            "review_strategy": self.review_strategy,
            "review_fallback_used": self.review_fallback_used,
            "post_takeoff_record_seconds": float(self.config.post_takeoff_record_seconds),
            "landing_detection_mode": "offline_window",
            "buffered_review": self.review_summary,
            "outputs": {
                "result": str(result_path),
                "visualization_aruco": str(measurement.visualization_path),
                "aruco_debug": str(measurement.aruco_debug_path),
                "person_mask": str(output_paths["mask_path"]),
                "landing_frame": str(landing_path),
            },
            "painting": [
                {"kind": "bbox", "target": "athlete", "xyxy": expanded_bbox, "color": "#00ff00"},
                {
                    "kind": "heel",
                    "point": [
                        float(measurement.heel_result.image_point[0]),
                        float(measurement.heel_result.image_point[1]),
                    ],
                    "color": "#ff3344",
                },
                {
                    "kind": "text",
                    "text": f"{float(measurement.score_cm):.2f} cm",
                    "position": [24, 42],
                    "color": "#ff3344",
                },
            ],
        }
        json_path = self.output_dir / "realtime_result_web.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["outputs"]["json"] = str(json_path)
        return payload

    def _select_final_landing_sample_from_raw_buffer(self):
        realtime = get_algorithms().realtime_core
        models = get_model_bundle()

        with self.lock:
            if self.takeoff_timestamp_sec is None:
                raise ValueError("未记录起跳时间，无法执行固定窗口复核")
            if self.ready_world_point is None:
                raise ValueError("未记录起点参考点，无法执行固定窗口复核")
            if self.locked_bbox is None:
                raise ValueError("未锁定测试人框，无法执行固定窗口复核")
            if self.plane is None:
                raise ValueError("未完成 ArUco 标定，无法执行固定窗口复核")

            review_start = float(self.takeoff_timestamp_sec) - float(
                self.config.review_before_takeoff_seconds
            )
            review_end = float(self.takeoff_timestamp_sec) + float(
                self.config.post_takeoff_record_seconds
            )
            raw_packets = self._raw_frames_between(
                review_start,
                review_end,
                step=self.config.review_frame_step,
            )
            previous_bbox = list(self.locked_bbox)
            start_world_x = float(self.ready_world_point[0])
            plane = self.plane

        if not raw_packets:
            raise ValueError("固定窗口缓存为空，无法执行落地帧复核")

        review_samples = []
        reviewed_frame_count = 0
        detected_frame_count = 0
        missed_frame_count = 0
        yolo_started_at = time.perf_counter()

        for raw_packet in raw_packets:
            if float(raw_packet.timestamp_sec) < float(self.takeoff_timestamp_sec):
                continue
            reviewed_frame_count += 1
            frame_packet = realtime.FramePacket(
                index=int(raw_packet.frame_id),
                timestamp=float(raw_packet.timestamp_sec),
                image=raw_packet.decode(),
            )
            detections = realtime.detect_people(
                models.yolo_model,
                frame_packet.image,
                plane,
            )
            detection = realtime.match_locked_person(
                detections,
                previous_bbox,
                max_center_distance_px=self.config.max_track_distance_px,
            )
            if detection is None:
                missed_frame_count += 1
                continue

            detected_frame_count += 1
            previous_bbox = list(detection.bbox)
            review_samples.append(
                realtime.JumpFrameSample(
                    packet=frame_packet,
                    detection=detection,
                    encoded_frame=None,
                )
            )

        yolo_seconds = time.perf_counter() - yolo_started_at
        landing_sample, peak_sample, strategy = self._select_offline_window_landing_sample(
            samples=review_samples,
            stable_frames=self.config.landing_stable_frames,
            max_change_px=self.config.landing_max_change_px,
            min_forward_cm=self.config.min_forward_cm,
            start_world_x=start_world_x,
            min_frames_after_peak=self.config.landing_min_frames_after_peak,
            min_recovery_px=self.config.landing_min_recovery_px,
        )
        summary = {
            "trigger_reason": "post_takeoff_window_complete",
            "start_time": review_start,
            "end_time": review_end,
            "raw_frame_count": len(raw_packets),
            "reviewed_frame_count": reviewed_frame_count,
            "detected_frame_count": detected_frame_count,
            "missed_frame_count": missed_frame_count,
            "landing_found": landing_sample is not None,
            "landing_frame_id": int(landing_sample.packet.index) if landing_sample is not None else None,
            "peak_frame_id": int(peak_sample.packet.index) if peak_sample is not None else None,
            "review_start_frame_id": int(raw_packets[0].frame_id),
            "review_end_frame_id": int(raw_packets[-1].frame_id),
            "strategy": strategy,
            "yolo_seconds": yolo_seconds,
            "review_frame_step": int(self.config.review_frame_step),
            "post_takeoff_record_seconds": float(self.config.post_takeoff_record_seconds),
            "landing_detection_mode": "offline_window",
            "raw_buffer": self._raw_buffer_stats(),
            "stream_stats": self._stream_stats(),
        }

        with self.lock:
            self.review_samples = review_samples
            self.review_summary = summary
            self.peak_sample = peak_sample

        if landing_sample is not None:
            with self.lock:
                self.review_strategy = f"offline_window_{strategy}"
                self.review_fallback_used = False
            return landing_sample

        raise ValueError("固定窗口离线复核未找到可用于测距的落地帧")

    def _select_offline_window_landing_sample(
        self,
        samples,
        stable_frames: int,
        max_change_px: float,
        min_forward_cm: float,
        start_world_x: float,
        min_frames_after_peak: int,
        min_recovery_px: float,
    ):
        stable_frames = max(1, int(stable_frames))
        if len(samples) < stable_frames:
            return None, None, None

        peak_index = min(range(len(samples)), key=lambda index: samples[index].detection.y2)
        peak_sample = samples[peak_index]
        peak_y2 = float(peak_sample.detection.y2)
        first_candidate_index = peak_index + max(1, int(min_frames_after_peak))
        if first_candidate_index >= len(samples):
            return None, peak_sample, None

        window = deque(maxlen=stable_frames)
        for sample in samples[first_candidate_index:]:
            window.append(sample)
            if len(window) < stable_frames:
                continue

            window_list = list(window)
            first_sample = window_list[0]
            first_world = first_sample.detection.bottom_center_world
            if first_world is None:
                continue
            if float(first_world[0]) - float(start_world_x) < float(min_forward_cm):
                continue

            y2_values = [item.detection.y2 for item in window_list]
            first_y2 = float(y2_values[0])
            if first_y2 < peak_y2 + max(0.0, float(min_recovery_px)):
                continue

            if first_y2 == max(y2_values) and all(
                value <= first_y2 for value in y2_values[1:]
            ):
                return first_sample, peak_sample, "offline_first_frame_max"

            if max(y2_values) - min(y2_values) <= float(max_change_px):
                return first_sample, peak_sample, "offline_stable_window"

        return None, peak_sample, None

    def _select_fast_ready_person(self, detections):
        realtime = get_algorithms().realtime_core
        selected = realtime.select_ready_person(
            detections,
            x_min=self.config.ready_x_min,
            x_max=self.config.ready_x_max,
            y_min=self.config.ready_y_min,
            y_max=self.config.ready_y_max,
        )
        if selected is not None:
            return selected
        if not detections:
            return None
        return max(detections, key=lambda detection: self._bbox_area(detection.bbox))

    def _match_active_person(self, detections, previous_bbox):
        realtime = get_algorithms().realtime_core
        matched = realtime.match_locked_person(
            detections,
            previous_bbox,
            max_center_distance_px=self.config.max_track_distance_px,
        )
        if matched is not None:
            self._set_match_diagnostics("locked_person", False)
            return matched
        if not detections:
            self._set_match_diagnostics("none", False)
            return None

        ready_candidate = realtime.select_ready_person(
            detections,
            x_min=-1000.0,
            x_max=1000.0,
            y_min=-1000.0,
            y_max=1000.0,
        )
        if ready_candidate is not None:
            self._set_match_diagnostics("ready_zone_fallback", True)
            return ready_candidate
        self._set_match_diagnostics("largest_bbox_fallback", True)
        return max(detections, key=lambda detection: self._bbox_area(detection.bbox))

    def _new_frame_diagnostics(self) -> Dict[str, Any]:
        return {
            "detected_person_count": None,
            "active_bbox": None,
            "locked_bbox": self._bbox_or_none(self.locked_bbox),
            "match_method": None,
            "match_fallback_used": False,
            "jump_samples_count": len(self.jump_samples),
            "landing_reason_zh": None,
        }

    def _set_detection_diagnostics(
        self,
        detections,
        active=None,
        match_method: Optional[str] = None,
        match_fallback_used: Optional[bool] = None,
    ) -> None:
        self.frame_diagnostics["detected_person_count"] = len(detections or [])
        self.frame_diagnostics["active_bbox"] = (
            self._bbox_or_none(active.bbox) if active is not None else None
        )
        self.frame_diagnostics["locked_bbox"] = self._bbox_or_none(self.locked_bbox)
        if match_method is not None:
            self.frame_diagnostics["match_method"] = match_method
        if match_fallback_used is not None:
            self.frame_diagnostics["match_fallback_used"] = bool(match_fallback_used)

    def _set_match_diagnostics(self, method: str, fallback_used: bool) -> None:
        self.frame_diagnostics["match_method"] = method
        self.frame_diagnostics["match_fallback_used"] = bool(fallback_used)

    def _diagnose_landing_not_triggered(self) -> Dict[str, Any]:
        samples = list(self.jump_samples)
        stable_frames = max(1, int(self.config.landing_stable_frames))
        diagnostics: Dict[str, Any] = {
            "jump_samples_count": len(samples),
            "landing_stable_frames": stable_frames,
            "landing_max_change_px": float(self.config.landing_max_change_px),
            "landing_min_forward_cm": float(self.config.min_forward_cm),
        }
        if len(samples) < stable_frames:
            diagnostics["landing_reason_zh"] = (
                f"样本不足：jump_samples 少于 stable_frames "
                f"({len(samples)}/{stable_frames})"
            )
            return diagnostics

        start_world_x = float(self.ready_world_point[0]) if self.ready_world_point is not None else None
        peak_index = min(range(len(samples)), key=lambda index: samples[index].detection.y2)
        diagnostics["landing_peak_frame_id"] = int(samples[peak_index].packet.index)
        failure_counts = {
            "window_not_contiguous": 0,
            "y2_jitter_too_large": 0,
            "forward_not_enough": 0,
            "world_point_missing": 0,
        }
        last_window: Dict[str, Any] = {}

        for start_index in range(peak_index, len(samples) - stable_frames + 1):
            window = samples[start_index : start_index + stable_frames]
            frame_indices = [int(sample.packet.index) for sample in window]
            y2_values = [float(sample.detection.y2) for sample in window]
            y2_range = max(y2_values) - min(y2_values)
            first_world = window[0].detection.bottom_center_world
            forward_cm = (
                float(first_world[0]) - start_world_x
                if first_world is not None and start_world_x is not None
                else None
            )
            last_window = {
                "frame_indices": frame_indices,
                "y2_values": y2_values,
                "y2_range_px": y2_range,
                "forward_cm": forward_cm,
            }
            if any(
                frame_indices[index + 1] != frame_indices[index] + 1
                for index in range(len(frame_indices) - 1)
            ):
                failure_counts["window_not_contiguous"] += 1
                continue
            if y2_range > float(self.config.landing_max_change_px):
                failure_counts["y2_jitter_too_large"] += 1
                continue
            if first_world is None or start_world_x is None:
                failure_counts["world_point_missing"] += 1
                continue
            if forward_cm is None or forward_cm < float(self.config.min_forward_cm):
                failure_counts["forward_not_enough"] += 1
                continue

            diagnostics["landing_reason_zh"] = "落地窗口已满足，但主流程尚未触发；请检查算法调用结果"
            diagnostics["landing_failure_counts"] = failure_counts
            diagnostics["landing_last_window"] = last_window
            return diagnostics

        diagnostics["landing_failure_counts"] = failure_counts
        diagnostics["landing_last_window"] = last_window
        if failure_counts["window_not_contiguous"] > 0:
            diagnostics["landing_reason_zh"] = "窗口不连续：帧号不连续"
        elif failure_counts["y2_jitter_too_large"] > 0:
            diagnostics["landing_reason_zh"] = "y2 抖动过大：超过 landing_max_change_px"
        elif failure_counts["forward_not_enough"] > 0:
            diagnostics["landing_reason_zh"] = "前移不足：未达到 min_forward_cm"
        elif failure_counts["world_point_missing"] > 0:
            diagnostics["landing_reason_zh"] = "前移不足：缺少世界坐标，无法判断 min_forward_cm"
        else:
            diagnostics["landing_reason_zh"] = "落地未触发：未找到可检查的稳定窗口"
        return diagnostics

    def _write_frame_diag_log(
        self,
        frame_id: int,
        timestamp_ms: int,
        previous_timestamp_ms: Optional[int],
        process_ms: float,
        response: Dict[str, Any],
        event_zh: Optional[str] = None,
        reason_zh: Optional[str] = None,
    ) -> None:
        dt_ms = (
            int(timestamp_ms) - int(previous_timestamp_ms)
            if previous_timestamp_ms is not None
            else None
        )
        effective_fps = 1000.0 / dt_ms if dt_ms and dt_ms > 0 else None
        timestamp_sec = int(timestamp_ms) / 1000.0
        state_elapsed_s = (
            max(0.0, timestamp_sec - float(self.state_started_at))
            if self.state_started_at is not None
            else None
        )
        event = {
            "event": "process_frame",
            "event_zh": event_zh or "处理视频帧",
            "reason_zh": reason_zh or self.frame_diagnostics.get("reason_zh") or response.get("message"),
            "frame_id": frame_id,
            "timestamp_ms": int(timestamp_ms),
            "process_ms": round(float(process_ms), 2),
            "dt_ms": dt_ms,
            "effective_fps": round(effective_fps, 2) if effective_fps is not None else None,
            "state_elapsed_s": round(state_elapsed_s, 3),
            "state": response.get("state", self.state),
            "message": response.get("message"),
            "error": response.get("error"),
            "ok": response.get("ok"),
            "control": response.get("control"),
            "score_cm": response.get("score_cm"),
        }
        event.update(self.frame_diagnostics)
        event["locked_bbox"] = self._bbox_or_none(self.locked_bbox)
        self._write_diag_event(event)

    def _write_diag_event(self, event: Dict[str, Any]) -> None:
        # 在调用线程里做廉价且能立即快照当前内容的序列化，把昂贵的磁盘 I/O 交给后台线程，
        # 避免在 16fps 分析等热路径上反复 open/write/flush 文件。
        try:
            with self._diag_lock:
                self._diag_index += 1
                diag_index = self._diag_index
            payload = {
                "diag_index": diag_index,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "uid": self.uid,
                "session_id": self.session_id,
            }
            payload.update(event)
            line = json.dumps(payload, ensure_ascii=False, default=self._json_default)
            self._diag_queue.put_nowait(line)
        except Exception as exc:
            print(f"[jump diag log ignored] {exc}")

    def _diag_writer_loop(self) -> None:
        handle = None
        try:
            while True:
                line = self._diag_queue.get()
                if line is None:
                    break
                if handle is None:
                    handle = self.diag_log_path.open("a", encoding="utf-8")
                handle.write(line + "\n")
                handle.flush()
        except Exception as exc:
            print(f"[jump diag writer stopped] {exc}")
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    def _status_state_elapsed_s(self) -> Optional[float]:
        # 统一用帧时钟：当前状态已持续多久 = 最近一帧的帧时间 - 状态起始帧时间。
        # 对 MP4 虚拟钟、摄像头单调钟都成立，不再依赖墙钟。
        if self.state_started_at is None or self.last_timestamp_ms is None:
            return None
        return round(
            max(0.0, int(self.last_timestamp_ms) / 1000.0 - float(self.state_started_at)),
            3,
        )

    @staticmethod
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        safe = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in str(value)
        )
        return safe[:64] or "uid"

    @staticmethod
    def _bbox_or_none(box) -> Optional[List[int]]:
        if box is None:
            return None
        return [int(value) for value in box]

    def _transition(self, state: str, timestamp_sec: float) -> None:
        if self.state in self.TERMINAL_STATES and state != self.state:
            return
        current_order = self.STATE_ORDER.get(self.state)
        next_order = self.STATE_ORDER.get(state)
        if current_order is not None and next_order is not None and next_order < current_order:
            self.message = f"忽略非法状态回退：{self.state} -> {state}"
            return
        self.state = state
        self.state_started_at = timestamp_sec

    def _set_failed(self, message: str) -> None:
        self.state = "FAILED"
        self.message = message
        self.error = message

    def _response(self, frame_id: Optional[Any], message: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "ok": self.error is None,
            "uid": self.uid,
            "session_id": self.session_id,
            "frame_id": frame_id,
            "state": self.state,
            "message": message or self.message,
            "control": {
                "pause": self.state in {"PROCESSING", "RESULT", "FAILED", "STOPPED"},
                "pause_frame_id": self.pause_frame_id,
            },
            "painting": self.last_painting,
            "score_cm": self.score_cm,
            "error": self.error,
            "stream_stats": self._stream_stats(),
            "raw_buffer": self._raw_buffer_stats(),
        }
        if self.state == "CALIBRATING" and self.calibration_attempt_path is not None:
            payload["debug"] = {
                "calib_latest_attempt": str(self.calibration_attempt_path),
                "calib_latest_attempt_aruco_debug": str(self.calibration_attempt_debug_path),
            }
        if self.result_payload:
            payload["result"] = self.result_payload
            payload["painting"] = self.result_payload.get("painting", payload["painting"])
        if self.debug_info:
            payload.setdefault("debug", {}).update(self.debug_info)
        return payload

    @staticmethod
    def _bbox_area(box: List[int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _people_painting(detections, active) -> List[Dict[str, Any]]:
        painting = []
        active_bbox = list(active.bbox) if active is not None else None
        for detection in detections:
            bbox = [int(value) for value in detection.bbox]
            painting.append(
                {
                    "kind": "bbox",
                    "target": "athlete" if bbox == active_bbox else "person",
                    "xyxy": bbox,
                    "confidence": float(detection.confidence),
                    "color": "#00ff00" if bbox == active_bbox else "#ffcc00",
                }
            )
        return painting

    @staticmethod
    def _aruco_painting(detections) -> List[Dict[str, Any]]:
        painting = []
        for marker_id, detection in sorted(detections.items()):
            painting.append(
                {
                    "kind": "aruco",
                    "id": int(marker_id),
                    "points": detection.corners.astype(float).tolist(),
                    "anchor": detection.anchor_point.astype(float).tolist(),
                    "color": "#38bdf8",
                }
            )
        return painting
