from enum import IntEnum
import math
import cv2
import time
from multiprocessing import Process
from multiprocessing.connection import PipeConnection

import numpy as np
from PIL import ImageDraw, ImageFont
from typing import Optional


def _detect_marker_circle_bgr(frame: np.ndarray) -> Optional[tuple[int, int, int]]:
    """
    将 backend/api/mark_cv.py 的标志物识别逻辑封装为函数：
    - HSV 绿色阈值分割
    - 形态学开运算去噪
    - Canny 边缘
    - HoughCircles 圆检测
    返回 (x, y, r) 像素坐标与半径；未检测到返回 None。
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 100])  # 设定绿色的阈值下限
        upper_green = np.array([100, 255, 255])  # 设定绿色的阈值上限

        mask = cv2.inRange(hsv, lower_green, upper_green)  # 设定掩膜取值范围
        _MARK_KERNEL = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MARK_KERNEL)
        edges = cv2.Canny(opening, 80, 160)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            1,
            100,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=100,
        )
        if circles is None:
            return None

        # 选半径最大的圆作为标志物（更稳）
        # c = max(circles[0], key=lambda v: float(v[2]))
        c = circles[0][0]  # 或者简单循环里用最后一个
        return (int(c[0]), int(c[1]), int(c[2]))
    except Exception:
        return None


class PosePoint:

    def __init__(self, idx, x, y, z):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z


class PointDetectionWorker:

    def __init__(self):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 降低复杂度以提升速度（0最快，1中等，2最慢）
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
        )

    # 循环处理函数
    def process(self, frame_id, img, drawing=False):
        h = img.shape[0]
        w = img.shape[1]
        points: list[PosePoint] = []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe需要RGB输入
        results = self.pose.process(img_rgb)  # mediapipe人体姿态推理

        if results.pose_landmarks:
            # 检测到人体关键点

            # 遍历全部33个人体关键点，换算像素坐标存入position
            for i in range(33):
                # 归一化坐标转画面像素
                cx = int(results.pose_landmarks.landmark[i].x * w)
                cy = int(results.pose_landmarks.landmark[i].y * h)
                cz = results.pose_landmarks.landmark[i].z  # 关键点深度值
                points.append(PosePoint(i, cx, cy, cz))

            # ========== 圆形标志物检测 ==========
            marker_circle = _detect_marker_circle_bgr(
                img
            )  # 检测脚底圆形标定物，返回(圆心x,y,半径)
            if marker_circle is not None:
                mx, my, mr = marker_circle

                # 替换关键点坐标
                points[23].x = mx
                points[23].y = my

            if drawing:
                if marker_circle is not None:
                    # 绘制标志物外圈+圆心点+文字
                    cv2.circle(img, (mx, my), mr, (0, 0, 255), 3)
                    cv2.circle(img, (mx, my), 3, (255, 255, 0), -1)
                    cv2.putText(
                        img,
                        "Marker",
                        (mx + mr + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # 绘制全身骨骼连线
                self.mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

                # 遍历全部33个人体关键点，绘制不同颜色关键点
                radius = 2  # 关键点绘制圆点半径
                for p in points:
                    # 按关键点分组填充不同颜色圆点
                    if p.idx == 0:
                        img = cv2.circle(img, (p.x, p.y), radius, (0, 0, 255), -1)
                    elif p.idx in [11, 12]:
                        img = cv2.circle(img, (p.x, p.y), radius, (223, 155, 6), -1)
                    elif p.idx in [23, 24]:
                        img = cv2.circle(img, (p.x, p.y), radius, (1, 240, 255), -1)
                    elif p.idx in [13, 14]:
                        img = cv2.circle(img, (p.x, p.y), radius, (140, 47, 240), -1)
                    elif p.idx in [25, 26]:
                        img = cv2.circle(img, (p.x, p.y), radius, (0, 0, 255), -1)
                    elif p.idx in [15, 16, 27, 28]:
                        img = cv2.circle(img, (p.x, p.y), radius, (223, 155, 60), -1)
                    elif p.idx in [17, 19, 21]:
                        img = cv2.circle(img, (p.x, p.y), radius, (94, 218, 121), -1)
                    elif p.idx in [18, 20, 22]:
                        img = cv2.circle(img, (p.x, p.y), radius, (16, 144, 247), -1)
                    elif p.idx in [27, 29, 31]:
                        img = cv2.circle(img, (p.x, p.y), radius, (29, 123, 243), -1)
                    elif p.idx in [28, 30, 32]:
                        img = cv2.circle(img, (p.x, p.y), radius, (193, 182, 255), -1)
                    elif p.idx in [9, 10]:
                        img = cv2.circle(img, (p.x, p.y), radius, (205, 235, 255), -1)
                    elif p.idx in [1, 2, 3, 4, 5, 6, 7, 8]:
                        img = cv2.circle(img, (p.x, p.y), radius, (94, 218, 121), -1)
                    else:
                        img = cv2.circle(img, (p.x, p.y), radius, (0, 255, 0), -1)

        return {
            "frame_id": frame_id,
            "has_person": results.pose_landmarks,
            "points": points,
        }


# 评测状态
class State(IntEnum):
    WAITING = 0
    READY = 1
    FLAT = 2
    RISE = 3
    WAIT_FLAT = 4
    FINISHED = 5


class Evaluator:
    # 评测算法

    def __init__(self):

        self.state = State.WAITING
        self.start_time = None
        self.num = 0
        self.num_all = 0

        # 观测角度缓存 obsY1~obsY5：各类人体角度时序数据
        (
            self.obsX,
            self.obsY1,
            self.obsY2,
            self.obsY3,
            self.obsY4,
            self.obsY5,
            self.wait,
        ) = ([], [], [], [], [], [], [])

        self.shoulder_x, self.shoulder_y = 0, 0
        self.heal_x, self.heal_y = 0, 0

        # 单帧作弊标记
        self.is_arm_force_cheat = False  # 手臂大开借力作弊标记
        self.is_out_of_bounds = False  # 人员替换/替考作弊标记
        self.is_flat_insufficient = False  # 肩背未着地作弊标记
        self.is_rise_insufficient = False  # 上起角度不足作弊标记
        self.flat_threshold = 30
        self.rise_threshold = 40
        self.last_flat_angle = 90
        self.last_rise_angle = 0

        # 流程状态标记
        self.env_ifok = 0  # 环境检测状态：0检测中/1合格/-1~-4各类环境异常
        self.env_ok_count = 0  # 环境检测合格帧计数器
        self.ready_count = 0  # 准备动作帧计数器

    # ==========================
    # 循环处理函数
    # ==========================
    def process(self, img, data):
        frame_id = data["frame_id"]
        points: list[PosePoint] = data["points"]
        has_person = data["has_person"]

        if not has_person:
            return

        h = img.shape[0]
        w = img.shape[1]
        scaler = 1

        # =================================
        # 参考线标识
        # ================================

        # 预设参考地面线两点坐标（左上、右下）
        x01, y001 = 0, 440
        x02, y002 = 640, 240

        # k0:拟合地面基线斜率 b0:拟合基线截距
        k0, b0 = 0, 0
        # k1：预设参考地面直线斜率
        k1 = (y001 - y002) / (x01 - x02)
        x3 = 320  # 参考线中点X

        # 左肩(11)、左脚踝(29)像素坐标，用于拟合地面基准线
        x1 = int(points[11].x)
        y1 = int(points[11].y)
        x2 = int(points[29].x)
        y2 = int(points[29].y)

        # 未正式测试时，绘制绿色预设参考地面线+中点标记
        if self.state == State.WAITING:
            img = cv2.line(img, (x01, y001), (x02, y002), (0, 255, 0), 3)
            img = cv2.circle(img, (x3, int(x3 * k1 + y001)), 5, (255, 0, 0), -1)

            # 参考线与当前肩脚连线夹角差≥5°且未进入测试，缓存肩、脚踝坐标用于拟合地面线
            if abs(math.degrees(math.atan(k0)) - math.degrees(math.atan(k1))) >= 5:
                shoulder_x = x1
                shoulder_y = y1
                heal_x = x2
                heal_y = y2

        # 使用最新一组肩、脚踝点计算拟合地面基线 k0斜率 b截距，避免除零
        k0 = (
            (shoulder_y - heal_y) / (shoulder_x - heal_x)
            if shoulder_x - heal_x != 0
            else 0
        )
        b0 = shoulder_y - k0 * shoulder_x
        # 计算拟合基线在x01/x02两点对应的Y值
        y01, y02 = int(k0 * x01 + b0), int(k0 * x02 + b0)
        # 绘制红色拟合地面基准线
        img = cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)

        # =================================
        # 左上角时间显示
        # ================================

        if self.start_time is None:
            self.start_time = time.time()
        total_time = int(time.time() - self.start_time)
        minute, second = total_time // 60, total_time % 60
        time_str = f"{minute}\"{second}'"

        img = cv2ImgAddText(
            img,
            f"{max(0, self.num)}",
            25 * scaler,
            60 * scaler,
            (0, 0, 255),
            40,
            drawBox=False,
        )
        img = cv2ImgAddText(
            img,
            f"{time_str}",
            25 * scaler,
            105 * scaler,
            (255, 165, 0),
            40,
            drawBox=False,
        )

        # =================================
        # 计算关键部位角度
        # ================================

        # ========== 计算躯干与地面夹角 body_ground_angle ==========
        A1, B1 = [], []
        A1.append(float(points[11].x))  # 左肩X
        A1.append(float(points[23].x))  # 脚底X
        A1.append(float(640))
        B1.append(float(points[11].y))  # 左肩Y
        B1.append(float(points[23].y))  # 脚底Y
        B1.append(float(points[23].y))
        body_ground_angle = AngleCalculate(A1, B1)
        self.wucha = 8  # 固定角度补偿误差
        self.obsY3.append(body_ground_angle - abs((math.atan(k0)) / math.pi * 180))

        # ========== 计算左手臂肩肘夹角 shoulder_elbow_angle ==========
        A2, B2 = [], []
        A2.append(float(points[16].x))
        A2.append(float(points[14].x))
        A2.append(float(points[12].x))
        B2.append(float(points[16].y))
        B2.append(float(points[14].y))
        B2.append(float(points[12].y))
        shoulder_elbow_angle = AngleCalculate(A2, B2)

        # ========== 计算右手臂肩肘夹角 shoulder_elbow_angle2 ==========
        A6, B6 = [], []
        A6.append(float(points[15].x))
        A6.append(float(points[13].x))
        A6.append(float(points[11].x))
        B6.append(float(points[15].y))
        B6.append(float(points[13].y))
        B6.append(float(points[11].y))
        shoulder_elbow_angle2 = AngleCalculate(A6, B6)
        self.obsY2.append(shoulder_elbow_angle)
        self.obsY4.append(shoulder_elbow_angle2)

        # ========== 脚底地面基准夹角 base_angle ==========
        A4, B4 = [], []
        A4.append(float(points[23].x))
        A4.append(float(points[29].x))
        A4.append(float(640))
        B4.append(float(points[23].y))
        B4.append(float(points[29].y))
        B4.append(float(points[29].y))
        base_angle = AngleCalculate(A4, B4)
        self.obsY5.append(body_ground_angle - abs((math.atan(k1)) / math.pi * 180))

        # ========== 环境合格判定逻辑：肩、头、脚关键点在地面线下方 ==========
        if self.state == State.WAITING:
            if (
                points[29].y <= k0 * points[29].x + b0 + 15
                and points[23].y <= k0 * points[23].x + b0 + 15
                and points[11].y <= k0 * points[11].x + b0 + 15
            ):
                if points[3].y > 0 and points[3].x > 0:
                    # 双脚在画面内，累计合格帧
                    if (
                        points[31].y <= 480
                        and points[32].y <= 480
                        and points[31].x <= 640
                        and points[32].x <= 640
                    ):
                        self.env_ok_count += 1
                        # 连续10帧环境正常，标记环境合格env_ifok=1
                        if frame_id >= 20 and self.env_ok_count >= 10:
                            self.env_ifok = 1
                    else:
                        self.env_ifok = -1  # 脚出框
                        self.env_ok_count = 0
                else:
                    self.env_ifok = -2  # 头部关键点丢失
                    self.env_ok_count = 0
            else:
                self.env_ifok = -4  # 躯干高于地面基准线，位置偏移
                self.env_ok_count = 0

        # ========== 环境状态文字提示分支（env_ifok区分环境异常） ==========
        if self.state == State.WAITING:
            if self.env_ifok == 0:
                img = cv2ImgAddText(
                    img,
                    "测试环境检测中",
                    25 * scaler,
                    15 * scaler,
                    (0, 0, 255),
                    40,
                    drawBox=False,
                )
                img = cv2ImgAddText(
                    img,
                    "请准备",
                    400 * scaler,
                    50 * scaler,
                    (255, 165, 0),
                    40,
                    drawBox=False,
                )
            elif self.env_ifok == -1:
                img = cv2ImgAddText(
                    img,
                    "脚出画框，请重新调整",
                    25 * scaler,
                    15 * scaler,
                    (0, 0, 255),
                    40,
                    drawBox=False,
                )
            elif self.env_ifok == -2:
                img = cv2ImgAddText(
                    img,
                    "头部出画框，请重新调整",
                    25 * scaler,
                    15 * scaler,
                    (0, 0, 255),
                    40,
                    drawBox=False,
                )
            elif self.env_ifok == -3:
                img = cv2ImgAddText(
                    img,
                    "斜度大于15°，请重新调整",
                    25 * scaler,
                    15 * scaler,
                    (0, 0, 255),
                    40,
                    drawBox=False,
                )
            elif self.env_ifok == -4:
                img = cv2ImgAddText(
                    img,
                    "请调整躯干低于中心线",
                    25 * scaler,
                    15 * scaler,
                    (0, 0, 255),
                    40,
                    drawBox=False,
                )
        elif self.state == State.READY:
            img = cv2ImgAddText(
                img,
                "检测成功！",
                25 * scaler,
                15 * scaler,
                (0, 0, 255),
                40,
                drawBox=False,
            )
            img = cv2ImgAddText(
                img,
                "请开始测试",
                400 * scaler,
                50 * scaler,
                (255, 165, 0),
                40,
                drawBox=False,
            )

        # ========== 判定是否进入预备阶段 ==========
        # 躯干脚底夹角差值<=10，环境合格，累计预备帧
        if (
            self.state == State.WAITING
            and body_ground_angle - base_angle <= 10
            and self.env_ifok == 1
        ):
            self.ready_count += 1
            # 10帧预备姿态，正式开启测试
            if self.ready_count >= 10:
                self.state = State.READY

        # ========== 正式测试阶段主逻辑 ==========
        if self.state >= State.READY and self.state < State.FINISHED:

            is_new_rise = False

            # ========== 修正躯干夹角body_ground_angle2（分段补偿误差） ==========
            # TODO: 不清楚为什么这么做
            if body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 15:
                body_ground_angle2 = (
                    body_ground_angle
                    - abs((math.atan(k0)) / math.pi * 180)
                    + self.wucha
                )
            if 15 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 30:
                body_ground_angle2 = (
                    body_ground_angle
                    - abs((math.atan(k0)) / math.pi * 180)
                    + self.wucha
                    + 10
                )
            if 30 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 40:
                body_ground_angle2 = (
                    body_ground_angle
                    - abs((math.atan(k0)) / math.pi * 180)
                    + self.wucha
                    + 20
                )
            if 30 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180):
                body_ground_angle2 = 60
            if body_ground_angle2 >= 60:
                body_ground_angle2 = 60
            self.obsY1.append(body_ground_angle2)

            # ========== 各类作弊帧检测，赋值作弊标记 ==========
            # 作弊1：手臂夹角≥150，连续3帧判定手臂大开借力
            if shoulder_elbow_angle >= 150:
                self.arm_force_count += 1
            else:
                self.arm_force_count = 0

            if self.arm_force_count >= 3:
                self.is_arm_force_cheat = True

            # 作弊2：人体关键点大面积出画面，判定替考换人
            if (
                min(points[2].y, points[5].y) < 0
                or min(points[7].x, points[8].x) < 0
                or max(points[7].x, points[8].x) > 640
                or max(points[25].y, points[26].y) > 480
                or min(points[25].x, points[26].x) < 0
                or max(points[25].x, points[26].x) > 640
            ):
                self.is_out_of_bounds = True

            # 记录波峰波谷
            self.last_flat_angle = min(body_ground_angle2, self.last_flat_angle)
            self.last_rise_angle = max(body_ground_angle2, self.last_rise_angle)

            # 作弊3：波谷角度>30 → 肩背未贴地
            if (
                self.last_flat_angle > self.flat_threshold
                and body_ground_angle2 > self.last_flat_angle + 10
            ):
                self.is_flat_insufficient = True

            # 作弊4：波峰角度<40 → 起身高度不足
            if (
                self.last_rise_angle < self.rise_threshold
                and body_ground_angle2 < self.last_rise_angle - 10
            ):
                self.is_rise_insufficient = True

            # 平躺判定逻辑
            if (
                self.state == State.READY
                or self.state == State.RISE
                or self.state == State.WAIT_FLAT
            ):

                if body_ground_angle2 <= self.flat_threshold:
                    self.state = State.FLAT
                    self.is_flat_insufficient = False
                    self.last_rise_angle = 0

            # 起身判定逻辑
            if self.state == State.FLAT:

                if body_ground_angle2 >= self.rise_threshold:
                    self.state = State.RISE
                    self.is_rise_insufficient = False

                    if self.last_flat_angle < self.flat_threshold:
                        is_new_rise = True

                    self.last_flat_angle = 90

            # ========== 有效计数num累加逻辑 ==========
            if is_new_rise:
                self.num_all += 1

            if self.is_out_of_bounds:
                self.state = State.FINISHED
                is_new_rise = False

            elif self.is_arm_force_cheat:
                self.state = State.WAIT_FLAT
                is_new_rise = False

            if is_new_rise:
                self.num += 1

            # ========== 作弊状态显示 ==========
            if self.is_arm_force_cheat:
                img = cv2ImgAddText(
                    img, "手臂打开借力", 25 * scaler, 350 * scaler, (255, 0, 255), 40
                )
            if self.is_out_of_bounds:
                img = cv2ImgAddText(
                    img, "测试者出画面", 25 * scaler, 375 * scaler, (255, 0, 255), 40
                )
            if self.is_flat_insufficient:
                img = cv2ImgAddText(
                    img, "肩背未着地", 25 * scaler, 400 * scaler, (255, 0, 255), 40
                )
            if self.is_rise_insufficient:
                img = cv2ImgAddText(
                    img, "起身角度不足", 25 * scaler, 400 * scaler, (255, 0, 255), 40
                )

        # ========== 结束阶段主逻辑 ==========
        if self.state == State.FINISHED:
            img = cv2ImgAddText(
                img,
                "测试结束！",
                25 * scaler,
                50 * scaler,
                (0, 0, 255),
                40,
                drawBox=False,
            )

        return {
            "frame": img,
            "frame_id": frame_id,
            "state": self.state,
            "score": self.num,
            "num_all": self.num_all,
            "angle": body_ground_angle2,
            "is_new_rise": is_new_rise,
            "is_arm_force_cheat": self.is_arm_force_cheat,
            "is_out_of_bounds": self.is_out_of_bounds,
            "is_flat_insufficient": self.is_flat_insufficient,
            "is_rise_insufficient": self.is_rise_insufficient,
            "timestamp": int(round(time.time() * 1000)),
        }


def AngleCalculate(x, y):
    a = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2
    b = (x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2
    c = (x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2
    if a == 0 or c == 0:
        angle = 0
    else:
        angle = abs(math.degrees(math.acos((a + c - b) / math.sqrt(4 * a * c))))
    return angle


def cv2ImgAddText(
    img,
    text,
    left,
    top,
    textColor=(0, 255, 0),
    textSize=10,
    borderColor=(0, 0, 0),
    borderWidth=2,
    drawBox=True,
    boxColor=(255, 0, 0),
    boxPadding=4,
    boxThickness=2,
):
    draw = ImageDraw.Draw(img)  # 创建一个可以在给定图像上绘图的对象
    try:
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8", index=1
        )
    except:
        fontStyle = ImageFont.load_default()

    bbox = draw.textbbox((left, top), text, font=fontStyle)
    text_width = bbox[2] - bbox[0]

    # 获取更准确的高度
    ascent, descent = fontStyle.getmetrics()
    text_height = ascent + descent

    # 可选绘制矩形框
    if drawBox:
        box_left = left - boxPadding
        box_top = top - boxPadding
        box_right = left + text_width + boxPadding
        box_bottom = top + text_height + boxPadding
        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            fill=None,
            outline=boxColor,
            width=boxThickness,
        )

    # 绘制边框
    for x in range(-borderWidth, borderWidth + 1):
        for y in range(-borderWidth, borderWidth + 1):
            if x != 0 or y != 0:
                draw.text((left + x, top + y), text, borderColor, font=fontStyle)

    # 绘制加粗的文本（通过多次绘制模拟加粗效果）
    for i in range(1, 3):
        draw.text((left + i, top), text, textColor, font=fontStyle)
        draw.text((left - i, top), text, textColor, font=fontStyle)
        draw.text((left, top + i), text, textColor, font=fontStyle)
        draw.text((left, top - i), text, textColor, font=fontStyle)

    # 再绘制一次文本以覆盖边框的部分
    draw.text((left, top), text, textColor, font=fontStyle)

    return img
