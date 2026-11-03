"""
detectors/pose_cheat_detector.py

坐位体前屈作弊检测模块：侧视角优化版

改进点：
1. MediaPipe Pose 使用 model_complexity=2
2. 不再使用“双腿取最差腿”策略
3. 优先选择可见度最高、关键点最稳定的一条腿
4. 加入膝关节角度检测
5. 加入腿部伸直比例检测
6. 加入时间平滑
7. 加入连续帧投票机制，降低单帧误判
"""

import mediapipe as mp
import math
from collections import deque
from typing import Tuple, Dict, Optional


mp_pose = mp.solutions.pose


# ==============================
# 参数配置
# ==============================
PARAMS = {
    # 膝盖角度阈值
    # 侧视角下 MediaPipe 抖动较大，不建议设太高
    "knee_angle_threshold_deg": 110.0,

    # landmark 平均最低可用置信度
    "visibility_threshold": 0.65,

    # 腿长比例阈值
    # 髋->踝直线距离 / (髋->膝 + 膝->踝)
    # 越接近 1 说明腿越直
    "leg_straight_ratio_threshold": 0.88,

    # 角度和平直比例平滑窗口
    "smooth_window": 8,

    # 连续帧投票窗口
    "cheat_vote_window": 8,

    # 投票窗口内至少多少帧异常才认为作弊
    "cheat_vote_threshold": 5,
}


# ==============================
# 几何工具
# ==============================
def distance_3d(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def angle_between_3d(a, b, c):
    """
    计算 ABC 夹角，B 点为顶点
    例如：
    hip - knee - ankle
    返回膝关节角度
    """
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]

    norm_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    norm_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = dot / (norm_ba * norm_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def leg_straight_ratio(hip, knee, ankle):
    """
    计算腿部伸直比例。

    full  = 髋到踝的直线距离
    split = 髋到膝 + 膝到踝

    full / split 越接近 1，说明腿越直。
    """
    full = distance_3d(hip, ankle)
    split = distance_3d(hip, knee) + distance_3d(knee, ankle)

    if split == 0:
        return 0

    return full / split


# ==============================
# 主检测类
# ==============================
class PoseCheatDetector:

    def __init__(self, params=None):
        self.params = PARAMS.copy()
        if params:
            self.params.update(params)

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # 平滑缓存
        self.angle_buffer = deque(maxlen=self.params["smooth_window"])
        self.ratio_buffer = deque(maxlen=self.params["smooth_window"])

        # 连续帧投票缓存
        self.cheat_buffer = deque(maxlen=self.params["cheat_vote_window"])

    # ==============================
    # 推理
    # ==============================
    def process(self, frame_bgr):
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.pose.process(frame_rgb)
        return results

    # ==============================
    # 获取关键点与可见度
    # ==============================
    def _get_landmark_with_vis(self, results, name) -> Optional[Dict]:
        if not results or not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark[
            getattr(mp_pose.PoseLandmark, name).value
        ]

        return {
            "point": (lm.x, lm.y, lm.z),
            "visibility": lm.visibility,
        }

    # ==============================
    # 单腿分析
    # ==============================
    def _analyze_leg(self, results, side):
        hip_data = self._get_landmark_with_vis(results, f"{side}_HIP")
        knee_data = self._get_landmark_with_vis(results, f"{side}_KNEE")
        ankle_data = self._get_landmark_with_vis(results, f"{side}_ANKLE")

        if not hip_data or not knee_data or not ankle_data:
            return None

        hip = hip_data["point"]
        knee = knee_data["point"]
        ankle = ankle_data["point"]

        vis_score = (
            hip_data["visibility"]
            + knee_data["visibility"]
            + ankle_data["visibility"]
        ) / 3.0

        # 如果这条腿整体可见度太差，直接不用
        if vis_score < self.params["visibility_threshold"]:
            return None

        angle = angle_between_3d(hip, knee, ankle)
        ratio = leg_straight_ratio(hip, knee, ankle)

        if angle is None:
            return None

        return {
            "side": side,
            "angle": angle,
            "ratio": ratio,
            "vis_score": vis_score,
            "hip_vis": hip_data["visibility"],
            "knee_vis": knee_data["visibility"],
            "ankle_vis": ankle_data["visibility"],
        }

    # ==============================
    # 主作弊检测
    # ==============================
    def check_cheat(self, results) -> Tuple[bool, Dict[str, str]]:
        vis = {
            "knee_angle": "n/a",
            "leg_ratio": "n/a",
            "used_leg": "n/a",
            "visibility": "n/a",
            "raw_cheat": "False",
            "vote": "0/0",
        }

        if not results or not results.pose_landmarks:
            self.cheat_buffer.append(False)
            return False, vis

        left_leg = self._analyze_leg(results, "LEFT")
        right_leg = self._analyze_leg(results, "RIGHT")

        candidates = []

        if left_leg:
            candidates.append(left_leg)

        if right_leg:
            candidates.append(right_leg)

        # 如果两条腿都不可用，不直接判作弊，避免误报
        if not candidates:
            self.cheat_buffer.append(False)
            return False, vis

        # 关键改动：
        # 不再选择“最差腿”
        # 而是选择可见度最高、关键点最可靠的一条腿
        best_leg = max(candidates, key=lambda x: x["vis_score"])

        angle = best_leg["angle"]
        ratio = best_leg["ratio"]

        # 平滑处理
        self.angle_buffer.append(angle)
        self.ratio_buffer.append(ratio)

        smooth_angle = sum(self.angle_buffer) / len(self.angle_buffer)
        smooth_ratio = sum(self.ratio_buffer) / len(self.ratio_buffer)

        # 原始单帧/平滑后判断
        angle_cheat = smooth_angle < self.params["knee_angle_threshold_deg"]
        ratio_cheat = smooth_ratio < self.params["leg_straight_ratio_threshold"]

        raw_cheat = angle_cheat or ratio_cheat

        # 连续帧投票
        self.cheat_buffer.append(raw_cheat)

        cheat_votes = sum(self.cheat_buffer)
        total_votes = len(self.cheat_buffer)

        cheat = cheat_votes >= self.params["cheat_vote_threshold"]

        vis["knee_angle"] = f"{smooth_angle:.1f}°"
        vis["leg_ratio"] = f"{smooth_ratio:.3f}"
        vis["used_leg"] = best_leg["side"]
        vis["visibility"] = f"{best_leg['vis_score']:.2f}"
        vis["raw_cheat"] = str(raw_cheat)
        vis["vote"] = f"{cheat_votes}/{total_votes}"

        return cheat, vis

    # ==============================
    # 关闭资源
    # ==============================
    def close(self):
        self.pose.close()


# ==============================
# 调试入口
# ==============================
if __name__ == "__main__":

    import cv2

    detector = PoseCheatDetector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = detector.process(frame)
        cheat, vis = detector.check_cheat(results)

        text1 = (
            f"Leg: {vis['used_leg']}  "
            f"Knee: {vis['knee_angle']}  "
            f"Ratio: {vis['leg_ratio']}"
        )

        text2 = (
            f"Vis: {vis['visibility']}  "
            f"Raw: {vis['raw_cheat']}  "
            f"Vote: {vis['vote']}"
        )

        color = (0, 0, 255) if cheat else (0, 255, 0)

        if cheat:
            text1 += "  CHEAT"

        cv2.putText(
            frame,
            text1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            frame,
            text2,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.imshow("Pose Cheat Detector - Side View Optimized", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()