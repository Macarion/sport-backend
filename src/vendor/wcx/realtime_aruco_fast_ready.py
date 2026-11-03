"""Emergency realtime ArUco pipeline with fast YOLO-only ready stage.

This script intentionally skips the rtoe/keypoint ready check. After ArUco
calibration, it locks the detected athlete with YOLO and immediately arms the
jump test after a few consecutive detections.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import realtime_aruco as realtime_core
from realtime_aruco import (
    ARUCOTEST_DIR,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_YOLO_MODEL,
    RealtimeJumpApp,
    RealtimeState,
    ModelBundle,
    WCX_DIR,
    build_parser as build_base_parser,
    configure_console_encoding,
    bbox_center,
    bbox_iou,
    detect_people,
    detect_takeoff,
    ensure_sam2_import_path,
    match_locked_person,
    normalize_sam2_config,
    resolve_path,
    select_landing_sample,
    select_ready_person,
)


READY_MODE = "fast_yolo_ready"


def configure_output_root(output_root):
    if not output_root:
        return
    output_path = resolve_path(output_root, [WCX_DIR, ARUCOTEST_DIR])
    output_path.mkdir(parents=True, exist_ok=True)
    realtime_core.REALTIME_OUTPUT_ROOT = output_path


def load_models_without_keypoint(
    yolo_model_path: Path,
    sam2_checkpoint: Path,
    sam2_config: str,
    device: str,
):
    from ultralytics import YOLO

    ensure_sam2_import_path()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    yolo_model = YOLO(str(yolo_model_path))
    sam2_model = build_sam2(
        config_file=normalize_sam2_config(sam2_config),
        ckpt_path=str(sam2_checkpoint),
        device=device,
    )
    return ModelBundle(
        yolo_model=yolo_model,
        sam2_predictor=SAM2ImagePredictor(sam2_model),
        keypoint_infer=None,
        torch_module=torch,
    )


def bbox_area(detection) -> int:
    x1, y1, x2, y2 = detection.bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


class FastReadyRealtimeApp(RealtimeJumpApp):
    ready_mode = READY_MODE

    def __init__(self, args):
        super().__init__(args)
        self.fast_ready_samples = deque(maxlen=max(1, int(args.auto_ready_frames)))
        self.armed_baseline_y2_values = deque(
            maxlen=max(1, int(args.armed_baseline_window))
        )
        self.fast_ready_status = "waiting for YOLO person"
        self.debug_log_path = self.session_dir / "fast_ready_debug.log"
        self._last_debug_packet_index = None
        self._last_debug_packet_timestamp = None
        self._last_debug_wall_time = None
        self._current_debug_metrics = {}
        self._attempt_debug_log_path = None
        if self.args.debug_log:
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_log(
                "fast ready debug log started "
                f"session_dir={self.session_dir} "
                f"camera_index={self.args.camera_index} "
                f"camera_size={self.args.camera_width}x{self.args.camera_height} "
                f"camera_fps={self.args.camera_fps} "
                f"max_track_distance_px={self.args.max_track_distance_px} "
                f"max_lost_frames={self.args.max_lost_frames} "
                f"takeoff_min_px={self.args.takeoff_min_px} "
                f"takeoff_image_ratio={self.args.takeoff_image_ratio} "
                f"dynamic_armed_baseline={self.args.dynamic_armed_baseline} "
                f"armed_baseline_window={self.args.armed_baseline_window} "
                f"single_frame_takeoff_ratio={self.args.single_frame_takeoff_ratio}",
                force=True,
            )
            print(f"[LOG] fast debug log: {self.debug_log_path}")

    def debug_log(self, message: str, packet=None, force: bool = False):
        if not getattr(self.args, "debug_log", True):
            return

        wall_text = time.strftime("%Y-%m-%d %H:%M:%S")
        state = getattr(self, "state", "UNKNOWN")
        state_value = state.value if hasattr(state, "value") else str(state)
        metrics = getattr(self, "_current_debug_metrics", {}) or {}
        packet_index = packet.index if packet is not None else metrics.get("packet_index")
        packet_time = packet.timestamp if packet is not None else metrics.get("packet_time")
        prefix = [
            wall_text,
            f"state={state_value}",
        ]
        if packet_index is not None:
            prefix.append(f"frame={packet_index}")
        if packet_time is not None:
            prefix.append(f"t={float(packet_time):.3f}")
        for key in ("skipped", "dt_packet_ms", "dt_wall_ms", "process_fps", "handle_ms"):
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                prefix.append(f"{key}={value:.2f}")
            else:
                prefix.append(f"{key}={value}")
        line = " | ".join(prefix) + " | " + message

        paths = [self.debug_log_path]
        if self._attempt_debug_log_path is not None:
            paths.append(self._attempt_debug_log_path)
        for path in dict.fromkeys(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

        if force or getattr(self.args, "debug_log_console", False):
            print(f"[FASTLOG] {line}")

    def build_packet_metrics(self, packet):
        now = time.perf_counter()
        skipped = None
        if self._last_debug_packet_index is not None:
            skipped = max(0, int(packet.index) - int(self._last_debug_packet_index) - 1)

        dt_packet_ms = None
        if self._last_debug_packet_timestamp is not None:
            dt_packet_ms = max(
                0.0,
                (float(packet.timestamp) - float(self._last_debug_packet_timestamp)) * 1000.0,
            )

        dt_wall_ms = None
        process_fps = None
        if self._last_debug_wall_time is not None:
            dt_wall_ms = max(0.0, (now - self._last_debug_wall_time) * 1000.0)
            if dt_wall_ms > 0:
                process_fps = 1000.0 / dt_wall_ms

        return {
            "packet_index": packet.index,
            "packet_time": packet.timestamp,
            "skipped": skipped,
            "dt_packet_ms": dt_packet_ms,
            "dt_wall_ms": dt_wall_ms,
            "process_fps": process_fps,
        }

    def handle_packet(self, packet):
        self._current_debug_metrics = self.build_packet_metrics(packet)
        started = time.perf_counter()
        previous_state = self.state
        try:
            super().handle_packet(packet)
        finally:
            self._current_debug_metrics["handle_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            if previous_state != self.state:
                self.debug_log(
                    f"state changed {previous_state.value}->{self.state.value}",
                    packet,
                    force=True,
                )
            self._last_debug_packet_index = packet.index
            self._last_debug_packet_timestamp = packet.timestamp
            self._last_debug_wall_time = time.perf_counter()

    def transition(self, state: RealtimeState, timestamp: float):
        self.debug_log(
            f"transition request {self.state.value}->{state.value} at_t={timestamp:.3f}",
            force=True,
        )
        super().transition(state, timestamp)

    def fail(self, reason: str, packet):
        self.debug_log(f"FAIL reason={reason}", packet, force=True)
        super().fail(reason, packet)

    def should_log_sample(self, packet, always: bool = False) -> bool:
        if always:
            return True
        every = max(1, int(getattr(self.args, "debug_log_every", 5)))
        return int(packet.index) % every == 0

    def format_detection(self, detection) -> str:
        if detection is None:
            return "None"
        center = bbox_center(detection.bbox)
        world = detection.bottom_center_world
        world_text = "world=None"
        if world is not None:
            world_text = f"world=({float(world[0]):.1f},{float(world[1]):.1f})"
        return (
            f"bbox={list(detection.bbox)} conf={float(detection.confidence):.3f} "
            f"center=({float(center[0]):.1f},{float(center[1]):.1f}) "
            f"y2={int(detection.y2)} area={bbox_area(detection)} {world_text}"
        )

    def match_debug_summary(self, previous_bbox, detections) -> str:
        if previous_bbox is None:
            return "prev_bbox=None"
        if not detections:
            return f"prev_bbox={list(previous_bbox)} detections=0"
        previous_center = bbox_center(previous_bbox)
        rows = []
        for index, detection in enumerate(detections):
            center_distance = float(
                np.linalg.norm(bbox_center(detection.bbox) - previous_center)
            )
            iou = bbox_iou(previous_bbox, detection.bbox)
            accepted = (
                iou >= 0.05
                or center_distance <= float(self.args.max_track_distance_px)
            )
            rows.append(
                (
                    center_distance,
                    f"d{index}:dist={center_distance:.1f},iou={iou:.3f},"
                    f"accepted={accepted},bbox={list(detection.bbox)},"
                    f"conf={float(detection.confidence):.3f}",
                )
            )
        rows.sort(key=lambda item: item[0])
        return f"prev_bbox={list(previous_bbox)} " + " ; ".join(
            row for _, row in rows[:5]
        )

    def update_armed_baseline(self, active_y2: float, threshold: float):
        if not self.args.dynamic_armed_baseline:
            return float(self.attempt.baseline_y2), False, "disabled"

        old_baseline = float(self.attempt.baseline_y2)
        active_y2 = float(active_y2)

        # y2 grows downward in image space. Ground-contact frames should be close
        # to, or larger than, the current baseline. Airborne frames have much
        # smaller y2 and must not pull the baseline upward/downward by mistake.
        update_margin = threshold * float(self.args.armed_baseline_update_margin_ratio)
        if active_y2 >= old_baseline - update_margin:
            self.armed_baseline_y2_values.append(active_y2)

        if not self.armed_baseline_y2_values:
            return old_baseline, False, "empty"

        candidate = float(max(self.armed_baseline_y2_values))
        new_baseline = max(old_baseline, candidate)
        changed = new_baseline > old_baseline
        if changed:
            self.attempt.baseline_y2 = new_baseline
        return new_baseline, changed, (
            f"candidate={candidate:.1f},samples="
            f"{[int(value) for value in self.armed_baseline_y2_values]}"
        )

    def start_model_loading(self):
        yolo_path = resolve_path(self.args.yolo_model, [WCX_DIR, ARUCOTEST_DIR])
        sam2_checkpoint = resolve_path(
            self.args.sam2_checkpoint,
            [WCX_DIR, ARUCOTEST_DIR],
        )
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model does not exist: {yolo_path}")
        if not sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {sam2_checkpoint}")

        def loader():
            try:
                self.models = load_models_without_keypoint(
                    yolo_model_path=yolo_path,
                    sam2_checkpoint=sam2_checkpoint,
                    sam2_config=self.args.sam2_config,
                    device=self.args.device,
                )
            except Exception as exc:
                self.model_error = exc

        self.model_thread = threading.Thread(target=loader, daemon=True)
        self.model_thread.start()
        self.debug_log(
            f"model loading started yolo={yolo_path} sam2={sam2_checkpoint} "
            f"device={self.args.device}",
            force=True,
        )

    def reset_for_calibration(self, timestamp: float):
        super().reset_for_calibration(timestamp)
        self.fast_ready_samples.clear()
        self.armed_baseline_y2_values.clear()
        self.fast_ready_status = "waiting for YOLO person"
        self._attempt_debug_log_path = self.attempt.output_dir / "fast_ready_debug.log"
        self.debug_log(
            f"new attempt attempt={self.attempt_number} output_dir={self.attempt.output_dir}",
            force=True,
        )

    def select_fast_ready_person(self, detections):
        selected = select_ready_person(
            detections,
            x_min=self.args.ready_x_min,
            x_max=self.args.ready_x_max,
            y_min=self.args.ready_y_min,
            y_max=self.args.ready_y_max,
        )
        if selected is not None:
            return selected
        if not detections:
            return None
        return max(detections, key=bbox_area)

    def handle_waiting_ready(self, packet):
        yolo_started = time.perf_counter()
        self.last_detections = detect_people(
            self.models.yolo_model,
            packet.image,
            self.attempt.plane,
        )
        yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        selected = self.select_fast_ready_person(self.last_detections)
        self.active_detection = selected
        self.pre_roll_frames.append(self.record_clip_packet(packet))

        if selected is None or selected.bottom_center_world is None:
            self.fast_ready_samples.clear()
            self.fast_ready_status = "no YOLO person"
            if self.should_log_sample(packet):
                self.debug_log(
                    f"WAITING_READY yolo_ms={yolo_ms:.1f} "
                    f"detections={len(self.last_detections)} selected=None",
                    packet,
                )
            return

        if self.fast_ready_samples:
            previous = self.fast_ready_samples[-1].bbox
            matched = match_locked_person(
                [selected],
                previous,
                max_center_distance_px=self.args.max_track_distance_px,
            )
            if matched is None:
                self.debug_log(
                    "WAITING_READY fast sample reset by match failure "
                    + self.match_debug_summary(previous, [selected]),
                    packet,
                )
                self.fast_ready_samples.clear()

        self.fast_ready_samples.append(selected)
        needed = max(1, int(self.args.auto_ready_frames))
        self.fast_ready_status = f"YOLO person {len(self.fast_ready_samples)}/{needed}"
        if self.should_log_sample(packet):
            self.debug_log(
                f"WAITING_READY yolo_ms={yolo_ms:.1f} "
                f"detections={len(self.last_detections)} "
                f"samples={len(self.fast_ready_samples)}/{needed} "
                f"selected={self.format_detection(selected)}",
                packet,
            )
        if len(self.fast_ready_samples) < needed:
            return

        points = np.array(
            [sample.bottom_center_world for sample in self.fast_ready_samples],
            dtype=np.float32,
        )
        self.attempt.baseline_y2 = float(
            np.median([sample.y2 for sample in self.fast_ready_samples])
        )
        self.attempt.ready_world_point = np.median(points, axis=0)
        self.attempt.locked_bbox = list(self.fast_ready_samples[-1].bbox)
        self.attempt.clip_frames = list(self.pre_roll_frames)
        self.armed_baseline_y2_values.clear()
        for sample in self.fast_ready_samples:
            self.armed_baseline_y2_values.append(float(sample.y2))
        self.takeoff_y2_values.clear()
        self.fast_ready_status = "armed by YOLO bbox"
        self.debug_log(
            f"READY_DONE baseline_y2={self.attempt.baseline_y2:.1f} "
            f"ready_world=({float(self.attempt.ready_world_point[0]):.1f},"
            f"{float(self.attempt.ready_world_point[1]):.1f}) "
            f"locked={self.attempt.locked_bbox}",
            packet,
            force=True,
        )
        self.transition(RealtimeState.ARMED, packet.timestamp)

    def handle_armed(self, packet):
        yolo_started = time.perf_counter()
        detections = detect_people(
            self.models.yolo_model,
            packet.image,
            self.attempt.plane,
        )
        yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        previous_bbox = list(self.attempt.locked_bbox)
        active = match_locked_person(
            detections,
            previous_bbox,
            max_center_distance_px=self.args.max_track_distance_px,
        )
        self.last_detections = detections
        self.active_detection = active
        self.record_clip_packet(packet)

        if packet.timestamp - self.state_started_at > self.args.takeoff_timeout_seconds:
            self.fail("等待起跳超时。", packet)
            return

        threshold = max(
            float(self.args.takeoff_min_px),
            float(packet.image.shape[0]) * float(self.args.takeoff_image_ratio),
        )
        if active is None:
            self.lost_person_frames += 1
            self.debug_log(
                f"ARMED_LOST yolo_ms={yolo_ms:.1f} "
                f"detections={len(detections)} lost={self.lost_person_frames}/"
                f"{self.args.max_lost_frames} threshold={threshold:.1f} "
                + self.match_debug_summary(previous_bbox, detections),
                packet,
                force=True,
            )
            if self.lost_person_frames > self.args.max_lost_frames:
                self.fail("准备起跳阶段连续丢失测试人。", packet)
            return

        self.lost_person_frames = 0
        self.attempt.locked_bbox = list(active.bbox)
        self.takeoff_y2_values.append(active.y2)
        baseline_before = float(self.attempt.baseline_y2)
        baseline_after, baseline_changed, baseline_reason = self.update_armed_baseline(
            active.y2,
            threshold,
        )
        rise = float(self.attempt.baseline_y2) - float(active.y2)
        recent = [int(value) for value in self.takeoff_y2_values]
        two_frame_takeoff = detect_takeoff(
            self.takeoff_y2_values,
            baseline_y2=self.attempt.baseline_y2,
            frame_height=packet.image.shape[0],
            min_pixels=self.args.takeoff_min_px,
            image_ratio=self.args.takeoff_image_ratio,
        )
        strong_threshold = threshold * float(self.args.single_frame_takeoff_ratio)
        strong_single_takeoff = (
            float(self.args.single_frame_takeoff_ratio) > 0
            and rise >= strong_threshold
        )
        takeoff = two_frame_takeoff or strong_single_takeoff
        if strong_single_takeoff:
            takeoff_reason = "single_frame_strong"
        elif two_frame_takeoff:
            takeoff_reason = "two_frame"
        else:
            takeoff_reason = "none"
        if baseline_changed:
            self.debug_log(
                f"ARMED_BASELINE_UPDATE old={baseline_before:.1f} "
                f"new={baseline_after:.1f} active_y2={int(active.y2)} "
                f"threshold={threshold:.1f} {baseline_reason}",
                packet,
                force=True,
            )
        if takeoff or self.should_log_sample(packet):
            self.debug_log(
                f"ARMED_TRACK yolo_ms={yolo_ms:.1f} detections={len(detections)} "
                f"baseline_y2={float(self.attempt.baseline_y2):.1f} "
                f"active_y2={int(active.y2)} rise={rise:.1f} "
                f"threshold={threshold:.1f} strong_threshold={strong_threshold:.1f} "
                f"recent_y2={recent} takeoff={takeoff} reason={takeoff_reason} "
                f"active={self.format_detection(active)} "
                + self.match_debug_summary(previous_bbox, detections),
                packet,
                force=takeoff,
            )
        if takeoff:
            self.attempt.jump_samples = []
            self.attempt.jump_start_frame_index = packet.index
            self.append_jump_sample(packet, active)
            self.debug_log(
                f"TAKEOFF_DETECTED jump_start_frame={packet.index} "
                f"baseline_y2={float(self.attempt.baseline_y2):.1f} "
                f"active_y2={int(active.y2)} rise={rise:.1f} "
                f"reason={takeoff_reason}",
                packet,
                force=True,
            )
            self.transition(RealtimeState.IN_JUMP, packet.timestamp)

    def handle_in_jump(self, packet):
        yolo_started = time.perf_counter()
        detections = detect_people(
            self.models.yolo_model,
            packet.image,
            self.attempt.plane,
        )
        yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        previous_bbox = list(self.attempt.locked_bbox)
        active = match_locked_person(
            detections,
            previous_bbox,
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
            self.debug_log(
                f"IN_JUMP_LOST yolo_ms={yolo_ms:.1f} "
                f"detections={len(detections)} lost={self.lost_person_frames}/"
                f"{self.args.max_lost_frames} samples={len(self.attempt.jump_samples)} "
                + self.match_debug_summary(previous_bbox, detections),
                packet,
                force=True,
            )
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
        peak_text = "None"
        if peak_sample is not None:
            peak_text = (
                f"frame={peak_sample.packet.index},y2={peak_sample.detection.y2}"
            )
        if landing_sample is not None or self.should_log_sample(packet):
            self.debug_log(
                f"IN_JUMP_TRACK yolo_ms={yolo_ms:.1f} detections={len(detections)} "
                f"samples={len(self.attempt.jump_samples)} active={self.format_detection(active)} "
                f"peak={peak_text} landing_found={landing_sample is not None} "
                + self.match_debug_summary(previous_bbox, detections),
                packet,
                force=landing_sample is not None,
            )
        if landing_sample is None:
            return

        self.attempt.trigger_landing_sample = landing_sample
        self.debug_log(
            f"LANDING_TRIGGER frame={landing_sample.packet.index} "
            f"y2={landing_sample.detection.y2} "
            f"world={landing_sample.detection.bottom_center_world}",
            packet,
            force=True,
        )
        self.start_processing(packet.timestamp)

    def status_lines(self, timestamp: float):
        if self.state != RealtimeState.WAITING_READY:
            return super().status_lines(timestamp)
        return [
            ("紧急快速准备模式", (0, 255, 255), 32),
            ("检测到测试人后自动开始，不做越线判定", (255, 255, 255), 23),
            (self.fast_ready_status, (0, 255, 255), 24),
        ]

    def process_attempt(self):
        result = super().process_attempt()
        self.add_ready_mode_to_json(result.json_path)
        return result

    def add_ready_mode_to_json(self, json_path: Path):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["ready_mode"] = self.ready_mode
        data["auto_ready_frames"] = int(self.args.auto_ready_frames)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser():
    parser = build_base_parser()
    parser.add_argument("--auto-ready-frames", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="实时结果输出根目录；每次运行会在下面自动创建时间戳会话目录",
    )
    parser.add_argument("--debug-log", action="store_true", default=True)
    parser.add_argument("--no-debug-log", action="store_false", dest="debug_log")
    parser.add_argument(
        "--debug-log-console",
        action="store_true",
        help="同时把详细日志打印到控制台；现场排查时会比较刷屏",
    )
    parser.add_argument(
        "--debug-log-every",
        type=int,
        default=5,
        help="WAITING_READY/正常追踪阶段每隔多少帧写一条采样日志",
    )
    parser.add_argument("--dynamic-armed-baseline", action="store_true", default=True)
    parser.add_argument(
        "--no-dynamic-armed-baseline",
        action="store_false",
        dest="dynamic_armed_baseline",
    )
    parser.add_argument(
        "--armed-baseline-window",
        type=int,
        default=5,
        help="ARMED 阶段用于修正脚底 baseline_y2 的最近帧窗口",
    )
    parser.add_argument(
        "--armed-baseline-update-margin-ratio",
        type=float,
        default=0.5,
        help="允许 baseline 更新的 y2 容差，按起跳阈值比例计算",
    )
    parser.add_argument(
        "--single-frame-takeoff-ratio",
        type=float,
        default=2.5,
        help="单帧 rise 超过 threshold 的倍数后直接判定起跳；<=0 表示关闭",
    )
    parser.set_defaults(window_name="ArUco Fast Ready Jump")
    return parser


def main():
    configure_console_encoding()
    args = build_parser().parse_args()
    configure_output_root(args.output_root)
    try:
        return FastReadyRealtimeApp(args).run()
    except Exception as exc:
        print(f"错误: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
