from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "yolov8n.pt"


def resolve_existing_path(path_str, base_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    base_path = base_dir / path
    if base_path.exists():
        return base_path

    return base_path


class JumpDetection:
    def __init__(self, video_path, model_path=None):
        self.video_path = resolve_existing_path(video_path, BASE_DIR)
        self.model_path = resolve_existing_path(
            model_path or str(DEFAULT_MODEL_PATH), BASE_DIR
        )

        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO 模型不存在: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.frames = self.load_all_frames(str(self.video_path))

        if not self.frames:
            raise ValueError(f"未从视频中读取到任何帧: {self.video_path}")

    def load_all_frames(self, video_path):
        """
        读取视频中的所有帧，并返回帧列表。
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def detect_person_yolo(self, frame):
        """判断当前帧是否有人，用二分查找进入/离开画面。"""
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                if box.cls[0] == 0:
                    return True
        return False

    def find_entry_exit_frames(self):
        """二分法查找人物进入帧和离开帧。"""
        total_frames = len(self.frames)
        step = 10

        left, right = 0, total_frames - 1
        entry_frame = None
        while left <= right:
            mid = (left + right) // 2
            if self.detect_person_yolo(self.frames[mid]):
                entry_frame = mid
                right = mid - step
            else:
                left = mid + step

        left, right = entry_frame if entry_frame else 0, total_frames - 1
        exit_frame = None
        consecutive_no_detection = 0
        max_no_detection_threshold = 10

        while left <= right:
            mid = (left + right) // 2
            if self.detect_person_yolo(self.frames[mid]):
                exit_frame = mid
                left = mid + step
                consecutive_no_detection = 0
            else:
                consecutive_no_detection += 1
                if consecutive_no_detection > max_no_detection_threshold:
                    break
                right = mid - step

        if exit_frame is None:
            exit_frame = total_frames - 1

        return entry_frame, exit_frame

    def detect_peak_frame_with_skip(
        self,
        start_frame,
        end_frame,
        skip_frames=3,
        look_ahead_box_size=5,
        size_ratio_threshold=2.0,
    ):
        """
        在 [start_frame, end_frame] 用跳帧和向后过滤小框的方法找到 Peak Frame。
        """
        raw_data = []
        frame_idx = start_frame
        while frame_idx <= end_frame:
            frame_bgr = self.frames[frame_idx]
            results = self.model(frame_bgr)
            boxes = []
            for result in results:
                for box in result.boxes:
                    if box.cls[0] == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2))

            if boxes:
                x1, y1, x2, y2 = boxes[0]
                height = y2 - y1
                foot_y = y2
                print(f"[PEAK STAGE] Frame {frame_idx}: foot_y={foot_y}, h={height}")
                raw_data.append((frame_idx, height, foot_y))
            else:
                print(f"[PEAK STAGE] Frame {frame_idx}: No valid person.")
            frame_idx += skip_frames

        if len(raw_data) < 2:
            print("[PEAK STAGE] Not enough data.")
            return None

        filtered_data = []
        for index, (frame_i, height_i, foot_i) in enumerate(raw_data):
            found_bigger = False
            for offset in range(1, look_ahead_box_size + 1):
                next_index = index + offset
                if next_index < len(raw_data):
                    _, next_height, _ = raw_data[next_index]
                    if next_height >= height_i * size_ratio_threshold:
                        found_bigger = True
                        break
            if found_bigger:
                print(f"[PEAK STAGE] Frame {frame_i} skip, next bigger box found.")
            else:
                filtered_data.append((frame_i, foot_i))

        if len(filtered_data) < 2:
            print("[PEAK STAGE] All frames skipped or too few left.")
            return None

        peak_frame, peak_y = min(filtered_data, key=lambda item: item[1])
        print(f"[PEAK STAGE] Detected Peak Frame: {peak_frame}, y2={peak_y}")
        return peak_frame

    def detect_landing_frame_with_window(
        self, peak_frame, end_frame, stable_frames=5, max_change_threshold=5
    ):
        """落地帧检测逻辑。"""
        foot_y_window = deque(maxlen=stable_frames)
        frame_idx = peak_frame
        landing_frame = None

        while frame_idx <= end_frame:
            frame_bgr = self.frames[frame_idx]
            results = self.model(frame_bgr)
            boxes = []
            for result in results:
                for box in result.boxes:
                    if box.cls[0] == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2))

            if not boxes:
                print(f"Frame {frame_idx}: No valid person detected.")
                frame_idx += 1
                continue

            _, _, _, y2 = boxes[0]
            foot_y = y2
            print(f"Frame {frame_idx}: foot_y={foot_y}")
            foot_y_window.append((frame_idx, foot_y))

            if len(foot_y_window) >= stable_frames:
                window_list = list(foot_y_window)
                max_y = max(value for _, value in window_list)
                min_y = min(value for _, value in window_list)
                first_frame_idx, first_foot_y = window_list[0]
                print(f"Window: {window_list}, max_y={max_y}, min_y={min_y}")

                if first_foot_y == max_y and all(
                    value <= first_foot_y for _, value in window_list[1:]
                ):
                    landing_frame = first_frame_idx
                    print(f"Landing Frame Detected (First Frame Max): {landing_frame}")
                    break

                if max_y - min_y <= max_change_threshold:
                    landing_frame = first_frame_idx
                    print(f"Landing Frame Detected (Stable Window): {landing_frame}")
                    break

            frame_idx += 1

        if landing_frame is None:
            print("Landing frame not detected within range.")
        return landing_frame
