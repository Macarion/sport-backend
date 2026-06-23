"""
backend/api/sitreach_core.py

坐位体前屈自动检测系统

保留功能：
1. 准备阶段 PREPARING
2. 姿态检测
3. 弯膝作弊检测
4. ArUco 标定
5. 中指检测
6. 实时成绩计算
7. 最大成绩记录
8. 15 秒测量倒计时
9. 原始视频保存
10. 带标注实时画面保存
11. 过程数据缓存
12. 结果保存
13. 前端实时获取图像和数据
"""

import os
import cv2
import time
import threading
import numpy as np
import mediapipe as mp

from datetime import datetime
from typing import Dict, Optional, Any

from .detectors.measure import PerspectiveCalibrator, HandDetector
from .detectors.pose_cheat_detector import PoseCheatDetector
import pyttsx3

import logging
logging.getLogger("comtypes").setLevel(logging.ERROR)

# =============================
# 系统参数
# =============================

ARUCO_SIZE_CM = 5.6
ARUCO_GAP_CM = 16.0

PREPARE_STABLE_FRAMES = 30
MEASURE_TIME_LIMIT = 40

VIDEO_DIR = "video"
RESULT_FILE = "result.txt"


# =============================
# 全局会话管理
# =============================

_SESSIONS: Dict[str, "SitReachSession"] = {}
_LOCK = threading.Lock()

class Voice:
    def __init__(self):
        self.lock = threading.Lock()
        self.speaking = False

    def speak(self, text):
        if self.speaking:
            return

        def _speak():
            self.speaking = True
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[VOICE ERROR] {e}")
            finally:
                self.speaking = False

        threading.Thread(target=_speak, daemon=True).start()

# =============================
# 骨架绘制模块
# =============================

class PoseDrawer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.drawer = mp.solutions.drawing_utils

    def draw(self, frame, results):
        if results and results.pose_landmarks:
            self.drawer.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.drawer.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2
                ),
                self.drawer.DrawingSpec(
                    color=(0, 0, 255),
                    thickness=2
                )
            )
            return True
        return False


# =============================
# 姿态有效性判断
# =============================

def is_valid_pose(results):
    if not results or not results.pose_landmarks:
        return False

    lm = results.pose_landmarks.landmark

    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27

    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28

    def visible(idx):
        return lm[idx].visibility > 0.5

    left_ok = (
        visible(LEFT_HIP)
        and visible(LEFT_KNEE)
        and visible(LEFT_ANKLE)
    )

    right_ok = (
        visible(RIGHT_HIP)
        and visible(RIGHT_KNEE)
        and visible(RIGHT_ANKLE)
    )

    return left_ok or right_ok


# =============================
# 单个用户的坐位体前屈测试会话
# =============================

class SitReachSession:

    def __init__(self, uid: str):
        self.uid = str(uid)

        # 算法模块
        self.calibrator = PerspectiveCalibrator(
            ARUCO_SIZE_CM,
            ARUCO_GAP_CM,
            marker_ids=(0, 1)
        )
        self.hand_detector = HandDetector()
        self.cheat_detector = PoseCheatDetector()
        self.pose_drawer = PoseDrawer()

        # 阶段状态
        self.stage = "PREPARING"
        self.prepare_frames = 0
        self.measure_start_time = None
        self.started_at = time.time()
        self.finished_at = None
        self.stopped = False

        # 成绩状态
        self.score = 0.0
        self.max_score = -999.0
        self.final_score = None

        # 检测状态
        self.cheat = False
        self.valid_pose = False
        self.is_calibrated = False
        self.has_finger = False
        self.finger_px = None
        self.cheat_vis = {}

        # 前端图像缓存
        self.latest_raw_frame = None
        self.latest_draw_frame = None

        # 过程数据
        self.frame_ids = []
        self.timestamps = []
        self.scores = []
        self.max_scores = []
        self.cheats = []
        self.valid_poses = []
        self.calibrated_list = []
        self.finger_list = []
        self.stages = []
        self.remaining_times = []

        # 视频保存
        os.makedirs(VIDEO_DIR, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # self.raw_video_path = os.path.join(
        #     VIDEO_DIR,
        #     f"{timestamp}_{self.uid}_sitreach_raw.mp4"
        # )
        # self.draw_video_path = os.path.join(
        #     VIDEO_DIR,
        #     f"{timestamp}_{self.uid}_sitreach_draw.mp4"
        # )
        self.raw_video_path = None
        self.draw_video_path = None

        self.raw_writer = None
        self.draw_writer = None
        self.video_fps = 30

        self.voice = Voice()
        self.has_spoken_start = False
        self.has_spoken_measure = False
        self.has_spoken_finish = False
        self.last_cheat_voice_time = 0

        self.voice.speak("请坐好并双腿伸直，脚掌贴紧挡板，双手向前伸")

    # =============================
    # 初始化视频写入器
    # =============================

    def _init_video_writer(self, frame):
        if self.raw_writer is not None and self.draw_writer is not None:
            return

        h, w = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.raw_writer = cv2.VideoWriter(
            self.raw_video_path,
            fourcc,
            self.video_fps,
            (w, h)
        )

        self.draw_writer = cv2.VideoWriter(
            self.draw_video_path,
            fourcc,
            self.video_fps,
            (w, h)
        )

    # =============================
    # 处理一帧
    # =============================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> Dict[str, Any]:

        if frame is None:
            return self._make_data(frame_id)

        # self._init_video_writer(frame)

        raw_frame = frame.copy()
        draw_frame = frame.copy()

        self.latest_raw_frame = raw_frame.copy()

        score_cm = None
        remain_time = MEASURE_TIME_LIMIT

        # 保存原始视频
        if self.raw_writer is not None:
            self.raw_writer.write(raw_frame)

        # =============================
        # 姿态检测
        # =============================
        results = self.cheat_detector.process(draw_frame)
        self.pose_drawer.draw(draw_frame, results)

        self.cheat, self.cheat_vis = self.cheat_detector.check_cheat(results)
        self.valid_pose = is_valid_pose(results)

        # =============================
        # 准备阶段
        # =============================
        if self.stage == "PREPARING":

            if not self.valid_pose:
                self.prepare_frames = 0
                cv2.putText(
                    draw_frame,
                    "Lower body not detected",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            elif self.cheat:
                self.prepare_frames = 0
                cv2.putText(
                    draw_frame,
                    "Knees not straight",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            else:
                self.prepare_frames += 1
                cv2.putText(
                    draw_frame,
                    f"Preparing {self.prepare_frames}/{PREPARE_STABLE_FRAMES}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

            if self.prepare_frames >= PREPARE_STABLE_FRAMES:
                self.stage = "MEASURING"
                self.measure_start_time = time.time()

                if not self.has_spoken_measure:
                    self.voice.speak(f"准备完成，开始测量，计时{MEASURE_TIME_LIMIT}秒")
                    self.has_spoken_measure = True

        # =============================
        # 测量阶段
        # =============================
        if self.stage == "MEASURING":

            if self.cheat:
                if time.time() - self.last_cheat_voice_time > 5:
                    self.voice.speak("请保持双腿伸直")
                    self.last_cheat_voice_time = time.time()
                cv2.putText(
                    draw_frame,
                    "Knees not straight",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # ArUco 标定
            self.is_calibrated = self.calibrator.calibrate(draw_frame)

            # 手指检测
            finger_px = self.hand_detector.find_middle_finger(draw_frame)
            self.finger_px = finger_px
            self.has_finger = finger_px is not None

            if self.is_calibrated:
                self.calibrator.draw_augmented_reality(draw_frame)

                if finger_px:
                    world = self.calibrator.pixel_to_world(finger_px)

                    if world:
                        score_cm = float(world[1])
                        self.score = round(score_cm, 1)

                        if score_cm > self.max_score:
                            self.max_score = round(score_cm, 1)

                        fx, fy = int(finger_px[0]), int(finger_px[1])
                        cv2.circle(
                            draw_frame,
                            (fx, fy),
                            10,
                            (0, 255, 255),
                            -1
                        )

                        self.calibrator.draw_augmented_reality(
                            draw_frame,
                            finger_world_y=score_cm
                        )

            # 倒计时
            elapsed_time = time.time() - self.measure_start_time
            remain_time = max(0, MEASURE_TIME_LIMIT - int(elapsed_time))

            if score_cm is not None:
                cv2.putText(
                    draw_frame,
                    f"Score: {self.score:.1f} cm",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )

            if self.max_score > -999:
                cv2.putText(
                    draw_frame,
                    f"Max: {self.max_score:.1f} cm",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

            cv2.putText(
                draw_frame,
                f"Time Left: {remain_time}s",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                3
            )

            if elapsed_time >= MEASURE_TIME_LIMIT:
                self.finish()

        # =============================
        # 完成阶段
        # =============================
        if self.stage == "FINISHED":
            if self.final_score is not None:
                cv2.putText(
                    draw_frame,
                    f"Final Score: {self.final_score:.1f} cm",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )
            else:
                cv2.putText(
                    draw_frame,
                    "No valid score",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        # 阶段显示
        cv2.putText(
            draw_frame,
            f"Stage: {self.stage}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        self.latest_draw_frame = draw_frame.copy()

        if self.draw_writer is not None:
            self.draw_writer.write(draw_frame)

        # 保存过程化数据
        self._append_runtime_data(frame_id, remain_time)

        return self._make_data(frame_id)

    # =============================
    # 保存过程数据
    # =============================

    def _append_runtime_data(self, frame_id: int, remain_time: int):
        now = datetime.now().isoformat(timespec="milliseconds")

        real_max = self.max_score if self.max_score > -999 else 0.0

        self.frame_ids.append(frame_id)
        self.timestamps.append(now)
        self.scores.append(float(self.score))
        self.max_scores.append(float(real_max))
        self.cheats.append(bool(self.cheat))
        self.valid_poses.append(bool(self.valid_pose))
        self.calibrated_list.append(bool(self.is_calibrated))
        self.finger_list.append(bool(self.has_finger))
        self.stages.append(self.stage)
        self.remaining_times.append(int(remain_time))

    # =============================
    # 返回前端数据
    # =============================

    def _make_data(self, frame_id: int = 0) -> Dict[str, Any]:
        real_max = self.max_score if self.max_score > -999 else 0.0

        return {
            "success": True,
            "uid": self.uid,
            "frame": frame_id,

            "score": float(self.score),
            "max_score": float(real_max),
            "final_score": self.final_score,

            "cheat": bool(self.cheat),
            "valid_pose": bool(self.valid_pose),
            "is_calibrated": bool(self.is_calibrated),
            "has_finger": bool(self.has_finger),

            "stage": self.stage,
            "prepare_frames": int(self.prepare_frames),
            "prepare_required": PREPARE_STABLE_FRAMES,
            "time_limit": MEASURE_TIME_LIMIT,

            "timestamp": datetime.now().isoformat(timespec="milliseconds"),

            "frame_ids": self.frame_ids,
            "timestamps": self.timestamps,
            "scores": self.scores,
            "max_scores": self.max_scores,
            "cheats": self.cheats,
            "valid_poses": self.valid_poses,
            "calibrated": self.calibrated_list,
            "has_fingers": self.finger_list,
            "stages": self.stages,
            "remaining_times": self.remaining_times,

            "finger_px": (
                [float(self.finger_px[0]), float(self.finger_px[1])]
                if self.finger_px is not None else None
            ),

            "cheat_vis": self.cheat_vis,

            "raw_video_path": self.raw_video_path,
            "draw_video_path": self.draw_video_path,
        }

    # =============================
    # 获取最新 JPEG 图像
    # =============================

    def get_latest_jpeg(self, draw=True) -> Optional[bytes]:
        frame = self.latest_draw_frame if draw else self.latest_raw_frame

        if frame is None:
            return None

        ok, buffer = cv2.imencode(".jpg", frame)

        if not ok:
            return None

        return buffer.tobytes()

    # =============================
    # 结束测试
    # =============================

    def finish(self):
        if self.stage == "FINISHED":
            return

        self.stage = "FINISHED"
        self.stopped = True
        self.finished_at = time.time()

        if self.max_score == -999:
            self.final_score = None
        else:
            result = round(self.max_score, 1)
            self.final_score = int(result * 1000) / 1000

        self._save_result()

        if not self.has_spoken_finish:
            if self.final_score is None:
                self.voice.speak("未能检测到有效成绩，请重新测试")
            else:
                self.voice.speak(f"测试完成，您的成绩为 {self.final_score} 厘米")
            self.has_spoken_finish = True

    # =============================
    # 保存最终成绩
    # =============================

    def _save_result(self):
        if self.final_score is None:
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"uid={self.uid}  invalid\n"
            )
        else:
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"uid={self.uid}  {self.final_score} cm\n"
            )

        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(line)

    # =============================
    # 释放资源
    # =============================

    def close(self):
        if self.stage != "FINISHED":
            self.finish()

        try:
            if self.raw_writer is not None:
                self.raw_writer.release()
        except Exception:
            pass

        try:
            if self.draw_writer is not None:
                self.draw_writer.release()
        except Exception:
            pass

        try:
            self.cheat_detector.close()
        except Exception:
            pass


# =============================
# 对外接口：启动
# =============================

def sitreach_start(uid):
    uid = str(uid)

    with _LOCK:
        if uid in _SESSIONS:
            _SESSIONS[uid].close()

        _SESSIONS[uid] = SitReachSession(uid)

    return {
        "success": True,
        "message": "坐位体前屈测试已启动",
        "uid": uid,
        "stage": "PREPARING",
        "prepare_required": PREPARE_STABLE_FRAMES,
        "time_limit": MEASURE_TIME_LIMIT,
    }


# =============================
# 对外接口：处理视频帧
# =============================

def sitreach_process_frame(
    uid,
    frame: np.ndarray,
    frame_id: int = 0,
):
    uid = str(uid)

    with _LOCK:
        if uid not in _SESSIONS:
            _SESSIONS[uid] = SitReachSession(uid)

        session = _SESSIONS[uid]

    return session.process_frame(
        frame=frame,
        frame_id=frame_id,
    )


# =============================
# 对外接口：获取实时数据
# =============================

def sitreach_fetch_inc_data(uid):
    uid = str(uid)

    with _LOCK:
        session = _SESSIONS.get(uid)

    if session is None:
        return {
            "success": False,
            "message": "当前用户没有正在进行的坐位体前屈测试",
            "uid": uid,
            "score": 0.0,
            "max_score": 0.0,
            "final_score": None,
            "cheat": False,
            "valid_pose": False,
            "is_calibrated": False,
            "has_finger": False,
            "stage": "NOT_STARTED",
            "scores": [],
            "max_scores": [],
            "cheats": [],
            "timestamps": [],
        }

    return session._make_data()


# =============================
# 对外接口：停止
# =============================

def sitreach_stop(uid):
    uid = str(uid)

    with _LOCK:
        session = _SESSIONS.pop(uid, None)

    if session is not None:
        session.close()

    return {
        "success": True,
        "message": "坐位体前屈测试已结束",
        "uid": uid,
    }


# =============================
# 对外接口：获取最新图像
# =============================

def sitreach_get_latest_frame(uid, draw=True):
    uid = str(uid)

    with _LOCK:
        session = _SESSIONS.get(uid)

    if session is None:
        return None

    return session.get_latest_jpeg(draw=draw)

# =============================
# 本地摄像头测试线程（可删）
# =============================

_CAMERA_THREADS = {}
_CAMERA_STOP_FLAGS = {}


def sitreach_start_local_camera(uid, camera_id=0):
    """
    后端本地打开摄像头测试。
    """
    uid = str(uid)

    sitreach_start(uid)

    stop_flag = threading.Event()
    _CAMERA_STOP_FLAGS[uid] = stop_flag

    def _camera_loop():
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_id = 0

        if not cap.isOpened():
            print("[sitreach] 无法打开摄像头")
            return

        print("[sitreach] 本地摄像头测试已启动")

        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            sitreach_process_frame(
                uid=uid,
                frame=frame,
                frame_id=frame_id,
            )

            frame_id += 1

            # 控制帧率，避免后端CPU太高
            time.sleep(1 / 25)

        cap.release()
        sitreach_stop(uid)
        print("[sitreach] 本地摄像头测试已停止")

    t = threading.Thread(target=_camera_loop, daemon=True)
    _CAMERA_THREADS[uid] = t
    t.start()

    return {
        "success": True,
        "message": "坐位体前屈本地摄像头测试已启动",
        "uid": uid
    }


def sitreach_stop_local_camera(uid):
    uid = str(uid)

    flag = _CAMERA_STOP_FLAGS.get(uid)
    if flag:
        flag.set()

    sitreach_stop(uid)

    return {
        "success": True,
        "message": "坐位体前屈本地摄像头测试已停止",
        "uid": uid
    }