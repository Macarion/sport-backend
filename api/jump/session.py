from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
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
    landing_stable_frames: int = 5
    landing_max_change_px: float = 5.0
    min_forward_cm: float = 30.0
    max_lost_frames: int = 15
    max_track_distance_px: float = 900.0
    bbox_expand_ratio: float = 0.08


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
        self.state = "CALIBRATING"
        self.state_started_at = time.time()
        self.created_at = self.state_started_at
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
        self.jump_samples: List[Any] = []
        self.trigger_landing_sample = None
        self.landing_sample = None
        self.peak_sample = None
        self.review_strategy: Optional[str] = None
        self.review_fallback_used = False

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

    def process_frame(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
        started_at = time.perf_counter()
        frame_id = int(frame_id)
        timestamp_ms = int(timestamp_ms)
        with self.lock:
            previous_timestamp_ms = self.last_timestamp_ms
            self.frame_diagnostics = self._new_frame_diagnostics()
            try:
                response = self._process_frame_locked(frame, frame_id, timestamp_ms)
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

    def _process_frame_locked(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            self.frame_diagnostics["reason_zh"] = "重复或乱序帧，状态机未重复处理"
            return self._response(frame_id=frame_id, message="重复或乱序帧，已忽略")

        if self.state in self.TERMINAL_STATES:
            self.frame_diagnostics["reason_zh"] = "测试已处于终态，忽略后续视频帧"
            return self._response(frame_id=self.last_frame_id, message=self.message)

        self.last_frame_id = int(frame_id)
        self.last_timestamp_ms = int(timestamp_ms)

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
                self.state_started_at = time.time()
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
            return response

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
            self.jump_start_frame_index = frame_id
            self._append_jump_sample(frame, frame_id, timestamp_ms, active)
            self._transition("IN_JUMP", timestamp_sec)
            self.message = "跳跃中"
            self.frame_diagnostics["jump_samples_count"] = len(self.jump_samples)
            self.frame_diagnostics["reason_zh"] = "起跳条件满足，进入跳跃中"
            return self._response(frame_id=frame_id)

        self.message = "可以开始起跳"
        self.frame_diagnostics["reason_zh"] = "起跳条件未满足：最近两帧未同时超过阈值"
        return self._response(frame_id=frame_id)

    def _handle_in_jump(self, frame, frame_id: int, timestamp_ms: int) -> Dict[str, Any]:
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

        if timestamp_sec - self.state_started_at > self.config.max_jump_seconds:
            self._set_failed("跳跃过程超时，未检测到稳定落地帧")
            self.frame_diagnostics.update(self._diagnose_landing_not_triggered())
            self.frame_diagnostics["reason_zh"] = "跳跃过程超时，未检测到稳定落地帧"
            return self._response(frame_id=frame_id)

        if active is None:
            self.lost_person_frames += 1
            self.frame_diagnostics["landing_reason_zh"] = "跳跃阶段人物丢失"
            self.frame_diagnostics["reason_zh"] = "跳跃阶段人物丢失"
            if self.lost_person_frames > self.config.max_lost_frames:
                self._set_failed("跳跃过程中连续丢失测试人")
            else:
                self.message = (
                    f"跳跃中：暂时丢失测试人 "
                    f"{self.lost_person_frames}/{self.config.max_lost_frames}"
                )
            return self._response(frame_id=frame_id)

        self.lost_person_frames = 0
        self.locked_bbox = list(active.bbox)
        self._append_jump_sample(frame, frame_id, timestamp_ms, active)
        self.frame_diagnostics["jump_samples_count"] = len(self.jump_samples)
        landing_sample, peak_sample = realtime.select_landing_sample(
            samples=self.jump_samples,
            stable_frames=self.config.landing_stable_frames,
            max_change_px=self.config.landing_max_change_px,
            min_forward_cm=self.config.min_forward_cm,
            start_world_x=float(self.ready_world_point[0]),
        )
        self.peak_sample = peak_sample
        if landing_sample is None:
            self.message = "跳跃中"
            self.frame_diagnostics.update(self._diagnose_landing_not_triggered())
            self.frame_diagnostics.setdefault("reason_zh", self.frame_diagnostics.get("landing_reason_zh"))
            return self._response(frame_id=frame_id)

        self.trigger_landing_sample = landing_sample
        self._start_processing(timestamp_sec)
        self.frame_diagnostics["landing_reason_zh"] = "已检测到落地，进入成绩计算"
        self.frame_diagnostics["reason_zh"] = "已检测到落地，进入成绩计算"
        return self._response(frame_id=frame_id, message="已检测到落地，正在计算成绩")

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
        self.message = "已检测到落地，正在计算成绩"
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()

    def _processing_worker(self) -> None:
        try:
            result = self._run_measurement()
            with self.lock:
                if self.state != "PROCESSING":
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
                if self.state != "PROCESSING":
                    return
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

        with self.lock:
            landing_sample = self._select_final_landing_sample_unlocked()
            landing_frame = landing_sample.packet.image.copy()
            landing_bbox = list(landing_sample.detection.bbox)
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

        payload = {
            "uid": self.uid,
            "session_id": self.session_id,
            "state": "RESULT",
            "score_cm": float(measurement.score_cm),
            "landing_frame_id": int(landing_sample.packet.index),
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

    def _select_final_landing_sample_unlocked(self):
        realtime = get_algorithms().realtime_core
        landing_sample, peak_sample, strategy = realtime.select_landing_sample_static_style(
            samples=self.jump_samples,
            stable_frames=self.config.landing_stable_frames,
            max_change_px=self.config.landing_max_change_px,
            min_forward_cm=self.config.min_forward_cm,
            start_world_x=float(self.ready_world_point[0]),
        )
        if landing_sample is not None:
            self.peak_sample = peak_sample
            self.review_strategy = f"tracked_raw_{strategy}"
            self.review_fallback_used = False
            return landing_sample

        if self.trigger_landing_sample is None:
            raise ValueError("未找到可用于测距的落地帧")
        self.review_strategy = "online_candidate_fallback"
        self.review_fallback_used = True
        return self.trigger_landing_sample

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
        state_elapsed_s = max(0.0, timestamp_sec - float(self.state_started_at))
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
        try:
            self._diag_index += 1
            payload = {
                "diag_index": self._diag_index,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "uid": self.uid,
                "session_id": self.session_id,
            }
            payload.update(event)
            with self.diag_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, default=self._json_default)
                    + "\n"
                )
        except Exception as exc:
            print(f"[jump diag log ignored] {exc}")

    def _status_state_elapsed_s(self) -> Optional[float]:
        if self.state_started_at < 1_000_000_000:
            return None
        return round(max(0.0, time.time() - float(self.state_started_at)), 3)

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
            "timestamp": int(time.time() * 1000),
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
