"""
已弃用
"""

from enum import IntEnum
import math
import cv2
import time
import os
import threading
import contextlib
from multiprocessing import Process
from multiprocessing.connection import Pipe, PipeConnection


class ReadWriteLock:
    """
    读写锁实现：
    - 支持多个读线程同时获取读锁
    - 写锁独占，持有写锁时不能有其他读或写线程
    - 写锁优先，防止写线程饥饿
    """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    @contextlib.contextmanager
    def read_lock(self):
        """获取读锁，支持多个读线程同时访问"""
        with self._read_ready:
            self._readers += 1
        try:
            yield
        finally:
            with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()

    @contextlib.contextmanager
    def write_lock(self):
        """获取写锁，独占访问"""
        with self._read_ready:
            while self._readers > 0:
                self._read_ready.wait()
            # 标记进入写模式，防止新的读线程进入
            self._readers = -1
        try:
            yield
        finally:
            with self._read_ready:
                self._readers = 0
                self._read_ready.notify_all()


# 全局读写锁实例
DATA_LOCK = ReadWriteLock()
import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import dataframe_image as dfi
import pandas as pd
import matplotlib.image as mpimg
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path
from queue import Queue, Empty
from collections import deque
import signal, sys
from pathlib import Path
import requests
import argparse
from contextlib import ExitStack

from api.base_sport import BaseSport
from api.config import CAMERA_INDEX


def _write_upload_debug(payload: Dict[str, Any]) -> None:
    try:
        debug_file = Path(__file__).with_name("test.json")
        debug_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Write upload debug failed: {e}")

STOP_FLAG = Path(__file__).with_suffix(".stop.flag")
stopping = {"flag": False}
def _request_stop(*_):
    stopping["flag"] = True

# 收到 Ctrl+C / Ctrl+Break 时，等同按下 q
signal.signal(signal.SIGINT, _request_stop)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _request_stop)
    
DATA_FILE: Path = Path(__file__).resolve().parent / "runtime_data.json"
_JSON_LOCK = threading.Lock()
def _append_record(record: Dict[str, Any]) -> None:
    """用新记录覆盖原有 JSON 内容（线程安全，仅保留最新一条）。"""
    with _JSON_LOCK:
        # 仍以数组形式保存，便于后续统一解析
        DATA_FILE.write_text(
            json.dumps([record], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

_MARK_KERNEL = np.ones((5, 5), np.uint8)

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
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        lower_green = np.array([35, 50, 100])  # 设定绿色的阈值下限
        upper_green = np.array([100, 255, 255])  # 设定绿色的阈值上限

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.inRange(hsv, lower_green, upper_green)  # 设定掩膜取值范围
        # mask = cv2.bitwise_or(mask1, mask2)
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

filename = time.strftime('%Y-%m-%d %H %M %S', time.localtime())
datapath = fr'video/{filename}-situp'

global frame_id, K
K, B, Y1, Y2 = [], [], [], []
shoulder_x, shoulder_y, heal_x, heal_y = 0, 0, 0, 0

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots()
# ax2 = ax.twiny()
fig.set_size_inches(6, 6)  # 设置图像大小
plt.ion()  # 打开交互模式

def get_maxima(values, order=8):
    values = np.asarray(values)
    if len(values) == 0:
        return [], []

    max_indices = []
    max_values = []

    i = 0
    while i < len(values):
        window = values[max(0, i - order): min(len(values), i + order + 1)]
        max_val = max(window)

        if values[i] == max_val:
            # 允许平坦峰值
            start = i
            while i + 1 < len(values) and values[i + 1] == values[i]:
                i += 1  # 跳过平坦区域

            # 只存储平坦区域的第一个点或最后一个点（这里选择第一个）
            max_indices.append(start)
            max_values.append(values[start])

        i += 1  # 移动到下一个值

    return max_indices, max_values

def get_minima(values, order=8):
    values = np.asarray(values)
    if len(values) == 0:
        return [], []

    min_indices = []
    min_values = []

    i = 0
    while i < len(values):
        window = values[max(0, i - order): min(len(values), i + order + 1)]
        min_val = min(window)

        if values[i] == min_val:
            # 允许平坦极小值
            start = i
            while i + 1 < len(values) and values[i + 1] == values[i]:
                i += 1  # 跳过平坦区域

            # 只存储平坦区域的第一个点（或最后一个点）
            min_indices.append(start)
            min_values.append(values[start])

        i += 1  # 移动到下一个点

    return min_indices, min_values

class PosePoint:

    def __init__(self, idx, x, y, z):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z

class ProcessWrapper(Process):

    def __init__(self, input_pipe: PipeConnection, output_pipe: PipeConnection):

        if input_pipe is None or output_pipe is None:
            raise ValueError('input_pipe and output_pipe must be provided')

        self.input_pipe = input_pipe
        self.output_pipe = output_pipe

    def init(self):
        pass

    def process(self, data):
        raise NotImplementedError('Subclasses must implement process method')

    def close(self):
        pass

    def run(self):

        self.init()

        try:
            while True:
                data = self.input_pipe.recv()
                if data is None:
                    break

                # 在这里处理数据
                result = self.process(data)

                # 将处理后的数据发送到输出管道
                if result is not None:
                    self.output_pipe.send(result)
        except EOFError:
            pass
        finally:
            self.output_pipe.close()
            self.close()
        
class PointDetectionProcess(ProcessWrapper):

    def init(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                            model_complexity=0,  # 降低复杂度以提升速度（0最快，1中等，2最慢）
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            enable_segmentation=False) 

    # 循环处理函数
    def process(self, data):
        frame_id = data.get('frame_id')
        img = data.get('img')
        drawing = data.get('drawing', False)

        h = img.shape[0]
        w = img.shape[1]
        points : list[PosePoint] = []

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
            marker_circle = _detect_marker_circle_bgr(img)  # 检测脚底圆形标定物，返回(圆心x,y,半径)
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
                    cv2.putText(img, "Marker", (mx + mr + 10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制全身骨骼连线
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

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
                'img': img,
                'frame_id': frame_id,
                'has_person': results.pose_landmarks,
                'points': points,
            }

# 评测状态
class State(IntEnum):
    WAITING = 0
    READY = 1
    FLAT = 2
    RISE = 3
    WAIT_FLAT = 4
    FINISHED = 5

class TestProcess(ProcessWrapper):
    # 评测进程

    def init(self):

        self.state = State.WAITING
        self.start_time = None
        self.num = 0
        self.num_all = 0

        # 观测角度缓存 obsY1~obsY5：各类人体角度时序数据
        self.obsX, self.obsY1, self.obsY2, self.obsY3, self.obsY4, self.obsY5, self.wait = [], [], [], [], [], [], []
        # 单帧作弊标记
        self.is_arm_force_cheat = False                 # 手臂大开借力作弊标记
        self.is_out_of_bounds = False                   # 人员替换/替考作弊标记
        self.is_flat_insufficient = False                 # 肩背未着地作弊标记
        self.is_rise_insufficient = False                 # 上起角度不足作弊标记
        self.flat_threshold = 30
        self.rise_threshold = 40
        self.last_flat_angle = 90
        self.last_rise_angle = 0
        # 流程状态标记
        self.env_ifok = 0            # 环境检测状态：0检测中/1合格/-1~-4各类环境异常
        self.env_ok_count = 0        # 环境检测合格帧计数器
        self.ready_count = 0         # 准备动作帧计数器

    # ==========================
    # 循环处理函数
    # ==========================
    def process(self, data):
        img = data['img']
        frame_id = data['frame_id']
        points : list[PosePoint] = data['points']
        has_person = data['has_person']

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
        k0 = (shoulder_y - heal_y) / (shoulder_x - heal_x) if shoulder_x - heal_x != 0 else 0
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
        time_str = f'{minute}\"{second}\''

        img = cv2ImgAddText(img, f'{max(0, self.num)}', 25 * scaler, 60 * scaler, (0, 0, 255), 40, drawBox=False)
        img = cv2ImgAddText(img, f'{time_str}', 25 * scaler, 105 * scaler, (255, 165, 0), 40, drawBox=False)

        # =================================
        # 计算关键部位角度
        # ================================

        # ========== 计算躯干与地面夹角 body_ground_angle ==========
        A1, B1 = [], []
        A1.append(float(points[11].x)) # 左肩X
        A1.append(float(points[23].x)) # 脚底X
        A1.append(float(640))
        B1.append(float(points[11].y)) # 左肩Y
        B1.append(float(points[23].y)) # 脚底Y
        B1.append(float(points[23].y))
        body_ground_angle = AngleCalculate(A1, B1)
        self.wucha = 8 # 固定角度补偿误差
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
            if points[29].y <= k0 * points[29].x + b0 + 15 and points[23].y <= k0 * points[23].x + b0 + 15 and points[11].y <= k0 * points[11].x + b0 + 15:
                if points[3].y > 0 and points[3].x > 0:
                    # 双脚在画面内，累计合格帧
                    if points[31].y <= 480 and points[32].y <= 480 and points[31].x <= 640 and points[32].x <= 640:
                        self.env_ok_count += 1
                        # 连续10帧环境正常，标记环境合格env_ifok=1
                        if frame_id >= 20 and self.env_ok_count >= 10:
                            self.env_ifok = 1
                    else:
                        self.env_ifok = -1 # 脚出框
                        self.env_ok_count = 0
                else:
                    self.env_ifok = -2 # 头部关键点丢失
                    self.env_ok_count = 0
            else:
                self.env_ifok = -4 # 躯干高于地面基准线，位置偏移
                self.env_ok_count = 0

        # ========== 环境状态文字提示分支（env_ifok区分环境异常） ==========
        if self.state == State.WAITING:
            if self.env_ifok == 0:
                img = cv2ImgAddText(img, '测试环境检测中', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                img = cv2ImgAddText(img, '请准备', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
            elif self.env_ifok == -1:
                img = cv2ImgAddText(img, '脚出画框，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -2:
                img = cv2ImgAddText(img, '头部出画框，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -3:
                img = cv2ImgAddText(img, '斜度大于15°，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -4:
                img = cv2ImgAddText(img, '请调整躯干低于中心线', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
        elif self.state == State.READY:
            img = cv2ImgAddText(img, '检测成功！', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            img = cv2ImgAddText(img, '请开始测试', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)

        # ========== 判定是否进入预备阶段 ==========
        # 躯干脚底夹角差值<=10，环境合格，累计预备帧
        if self.state == State.WAITING and body_ground_angle - base_angle <= 10 and self.env_ifok == 1:
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
                body_ground_angle2 = body_ground_angle - abs((math.atan(k0)) / math.pi * 180) + self.wucha
            if 15 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 30:
                body_ground_angle2 = body_ground_angle - abs((math.atan(k0)) / math.pi * 180) + self.wucha + 10
            if 30 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 40:
                body_ground_angle2 = body_ground_angle - abs((math.atan(k0)) / math.pi * 180) + self.wucha + 20
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
            if (min(points[2].y, points[5].y) < 0 or min(points[7].x, points[8].x) < 0 or max(points[7].x, points[8].x) > 640 or
                    max(points[25].y, points[26].y) > 480 or min(points[25].x, points[26].x) < 0 or max(points[25].x, points[26].x) > 640):
                self.is_out_of_bounds = True

            # 记录波峰波谷
            self.last_flat_angle = min(body_ground_angle2, self.last_flat_angle)
            self.last_rise_angle = max(body_ground_angle2, self.last_rise_angle)

            # 作弊3：波谷角度>30 → 肩背未贴地
            if self.last_flat_angle > self.flat_threshold and body_ground_angle2 > self.last_flat_angle + 10:
                self.is_flat_insufficient = True
            
            # 作弊4：波峰角度<40 → 起身高度不足
            if self.last_rise_angle < self.rise_threshold and body_ground_angle2 < self.last_rise_angle - 10:
                self.is_rise_insufficient = True

            # 平躺判定逻辑
            if self.state == State.READY or self.state == State.RISE or self.state == State.WAIT_FLAT:

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
                img = cv2ImgAddText(img, '手臂打开借力', 25 * scaler, 350 * scaler, (255, 0, 255), 40)
            if self.is_out_of_bounds:
                img = cv2ImgAddText(img, '测试者出画面', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
            if self.is_flat_insufficient:
                img = cv2ImgAddText(img, '肩背未着地', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
            if self.is_rise_insufficient:
                img = cv2ImgAddText(img, '起身角度不足', 25 * scaler, 400 * scaler, (255, 0, 255), 40)


        # ========== 结束阶段主逻辑 ==========
        if self.state == State.FINISHED:
            img = cv2ImgAddText(img, '测试结束！', 25 * scaler, 50 * scaler, (0, 0, 255), 40, drawBox=False)
        
        return {
            'frame': img,
            'frame_id': frame_id,
            'state': self.state,
            'score': self.num,
            'num_all': self.num_all,
            'angle': body_ground_angle2,
            'is_new_rise': is_new_rise,
            'is_arm_force_cheat': self.is_arm_force_cheat,
            'is_out_of_bounds': self.is_out_of_bounds,
            'is_flat_insufficient': self.is_flat_insufficient,
            'is_rise_insufficient': self.is_rise_insufficient,
            'timestamp': int(round(time.time() * 1000)),
        }


# def process_count(kwargs: dict):
#     # 计数进程

def process_frame(img, frame_id, WIDTH, HEIGHT, testing, IF_START):
    """
    单帧人体姿态预处理函数
    :param img: 输入原始BGR图像
    :param frame_id: 当前帧序号
    :param WIDTH: 画面缩放宽
    :param HEIGHT: 画面缩放高
    :param testing: 测试标记 0未开始/1正式测试
    :param IF_START: 每帧测试状态时序列表
    :return: img:绘制骨骼/标志物后的画面; if_existperson:1有人0无人; start_time:帧推理起始时间;
             position:33个人体关键点坐标列表; k0:拟合地面基线斜率; b:地面基线截距;
             k1:预设参考地面线斜率; marker_replaced:是否用标志物覆盖左臀关键点; marker_circle:标志物圆信息
    """
    # position：存储33个mediapipe人体关键点 [关键点编号,像素X,像素Y,深度Z]
    position = []
    # k0:拟合地面基线斜率 b:拟合基线截距 
    k0, b = 0, 0
    # 预设参考地面线两点坐标（左上、右下）
    x01, y001 = 0, 440
    x02, y002 = 640, 240
    # y01/y02：拟合后地面线在x01/x02处的Y坐标
    y01, y02 = y001, y002
    # k1：预设参考地面直线斜率
    k1 = (y001 - y002) / (x01 - x02)
    x3 = 320  # 参考线中点X
    start_time = time.time()  # 记录帧推理开始时间，用于计算FPS
    h, w = img.shape[0], img.shape[1]  # 图像高、宽像素
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe需要RGB输入

    img = img.copy()  # 拷贝原图，避免修改输入图像

    results = pose.process(img_RGB)  # mediapipe人体姿态推理

    # ========== 圆形标志物检测 ==========
    marker_circle = _detect_marker_circle_bgr(img)  # 检测脚底圆形标定物，返回(圆心x,y,半径)
    marker_replaced = False  # 标记是否使用标志物坐标覆盖左臀关键点
    if marker_circle is not None:
        mx, my, mr = marker_circle
        # 绘制标志物外圈+圆心点+文字
        cv2.circle(img, (mx, my), mr, (0, 0, 255), 3)
        cv2.circle(img, (mx, my), 3, (255, 255, 0), -1)
        cv2.putText(img, "Marker", (mx + mr + 10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        marker_replaced = True

    # ========== 检测到人体骨架关键点 ==========
    if results.pose_landmarks:
        if_existperson = 1  # 标记画面存在人体
        # 若检测到标志物，替换左臀关键点(23)归一化坐标，修正地面角度计算
        if marker_replaced:
            results.pose_landmarks.landmark[23].x = mx / w
            results.pose_landmarks.landmark[23].y = my / h
        # 绘制全身骨骼连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 遍历全部33个人体关键点，换算像素坐标存入position并绘制不同颜色关键点
        for i in range(33):
            # 归一化坐标转画面像素
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z  # 关键点深度值
            radius = 2  # 关键点绘制圆点半径
            position.append([i, cx, cy, cz])
            # 按关键点分组填充不同颜色圆点
            if i == 0:
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [11, 12]:
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
            elif i in [23, 24]:
                img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
            elif i in [13, 14]:
                img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
            elif i in [25, 26]:
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [15, 16, 27, 28]:
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
            elif i in [17, 19, 21]:
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            elif i in [18, 20, 22]:
                img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
            elif i in [27, 29, 31]:
                img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
            elif i in [28, 30, 32]:
                img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
            elif i in [9, 10]:
                img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
            elif i in [1, 2, 3, 4, 5, 6, 7, 8]:
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            else:
                img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)

        # 左肩(11)、左脚踝(29)像素坐标，用于拟合地面基准线
        x1 = int(results.pose_landmarks.landmark[11].x * w)
        y1 = int(results.pose_landmarks.landmark[11].y * h)
        x2 = int(results.pose_landmarks.landmark[29].x * w)
        y2 = int(results.pose_landmarks.landmark[29].y * h)

        # 未正式测试时，绘制绿色预设参考地面线+中点标记
        if testing == 0:
            img = cv2.line(img, (x01, y001), (x02, y002), (0, 255, 0), 3)
            img = cv2.circle(img, (x3, int(x3 * k1 + y001)), 5, (255, 0, 0), -1)

        # 参考线与当前肩脚连线夹角差≥5°且未进入测试，缓存肩、脚踝坐标用于拟合地面线
        if abs(math.degrees(math.atan(k0)) - math.degrees(math.atan(k1))) >= 5 and 1 not in IF_START:
            shoulder_x.append(x1)
            shoulder_y.append(y1)
            heal_x.append(x2)
            heal_y.append(y2)
        # 使用最新一组肩、脚踝点计算拟合地面基线 k0斜率 b截距，避免除零
        k0 = (shoulder_y[-1] - heal_y[-1]) / (shoulder_x[-1] - heal_x[-1]) if shoulder_x[-1] - heal_x[-1] != 0 else 0
        b = shoulder_y[-1] - k0 * shoulder_x[-1]
        # 计算拟合基线在x01/x02两点对应的Y值
        y01, y02 = int(k0 * x01 + b), int(k0 * x02 + b)
        # 绘制红色拟合地面基准线
        img = cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)
    else:
        if_existperson = 0  # 未检测到人体

    # 返回全部推理结果
    return img, if_existperson, start_time, position, k0, b, k1, marker_replaced, marker_circle

def update_plot(ax, frame_id, obsY1, num):
    if len(frame_id) == 0 or len(obsY1) == 0:
        return

    min_len = min(len(frame_id), len(obsY1))
    frame_id = frame_id[:min_len]
    obsY1 = obsY1[:min_len]
    num = num[:min_len]

    ax.clear()
    ax.plot(frame_id, obsY1, 'r-', label='ObsY1')

    # 设置横坐标最小值和最大值
    if len(frame_id) > 10:
        ax.set_xlim(frame_id[-10], frame_id[-1] + 1)
    else:
        ax.set_xlim(0, len(frame_id) + 1)

    # 设置纵坐标最小值为0，最大值根据数据动态变化
    ax.set_ylim(0, max(obsY1[-100:]) + 1)
    ax.set_title("实时计数")
    ax.set_xlabel("检测时间")
    ax.set_ylabel("起伏角度")

    # 添加图例
    ax.legend(loc='upper left')

    # ax2.set_xlim(ax.get_xlim())  # 使得新横坐标与原始横坐标共享相同的范围
    # ax2.set_xticks(frame_id)  # 使用 frame_id 作为坐标
    # ax2.set_xticklabels(num)  # 使用 num 作为标签

    # plt.draw()
    # plt.pause(0.1)

def save_plot(filename, frame_id, obsY1, num):
    if len(frame_id) == 0 or len(obsY1) == 0:
        return

    min_len = min(len(frame_id), len(obsY1))
    frame_id = frame_id[:min_len]
    obsY1 = obsY1[:min_len]
    num = num[:min_len]

    # 使用非交互式后端直接创建 Figure，避免主线程检查问题
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    fig_save = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig_save)  # 使用 Agg 后端，不需要主线程
    ax_save = fig_save.add_subplot(111)
    
    ax_save.plot(frame_id, obsY1, 'r-', label='body_ground angle')
    ax_save.set_title("实时计数")
    ax_save.set_xlabel("检测时间")
    ax_save.set_ylabel("起伏角度")

    # 添加图例
    ax_save.legend(loc='upper left')

    # 保存为长图
    fig_save.savefig(filename, bbox_inches='tight')
    canvas.draw()  # 确保图形被渲染

def AngleCalculate(x, y):
    a = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2
    b = (x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2
    c = (x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2
    if a == 0 or c == 0:
        angle = 0
    else:
        angle = abs(math.degrees(math.acos((a + c - b) / math.sqrt(4 * a * c))))
    return angle

def mkdir(path):
    id = 0
    folder = os.path.exists(path)
    filename = os.path.split(path)[1]
    new_path = path
    if folder:
        while folder:
            new_path = ""
            id = id + 1
            new_path = path + f"({id})"
            os.makedirs(new_path)
            folder = os.path.split(new_path)[1]
            print(f"---  新建文件夹{filename}...  ---")
            print("---  完成！  ---")
            break
    else:
        os.makedirs(path)
    return new_path

def ShapeDetection(img):
    if_tool = 0
    img_height, img_width = img.shape[:2]  # 获取图像的高度和宽度
    # 提取轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for obj in contours:
        x, y, w, h = cv2.boundingRect(obj)

        # 判断是否是细长的线段
        if w > 5 * h and w > img_width * 0.5:  # 长宽比大于 5，且高度超过图像的一半
            if x < img_width * 0.3 and x + w > img_width * 0.7:  # 顶部接近上边界，底部接近下边界
                if_tool = 1
                break  # 一旦找到符合条件的线段即可退出循环

    return if_tool

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=10, borderColor=(0, 0, 0), borderWidth=2, drawBox=True, boxColor=(255, 0, 0), boxPadding=4, boxThickness=2):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)  # 创建一个可以在给定图像上绘图的对象
    try:
        fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8", index=1)
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
            fill=None, outline=boxColor, width=boxThickness
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

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式

datapath1 = mkdir(datapath)
FILENAME = datapath1 + f'/{filename}-situp.mp4'


class SITUP(BaseSport):
    def __init__(self, username: Optional[str] = None):
        self.username = username
        # ===================== 曲线存储列表 =====================
        self.L = []                  # 预留曲线缓存
        self.X = []                  # X轴：帧ID，绘图横坐标
        self.X1 = []                 # 每次有效计数num对应的帧标记
        self.Lk_filter = []          # 结果折线图的中间处理曲线（对源数据做平滑处理）
        self.Z = []                  # 外物作弊标记序列（b_l）
        self.frametime = []          # 满足初始姿态的帧序列，用于判定测试开始
        self.leg_lengthr = []         # 右腿长度缓存
        self.leg_lengthl = []        # 左腿长度缓存
        # 绘图辅助线段
        self.line1, self.line2, self.line3, self.line4 = None, None, None, None
        # 观测角度缓存 obsY1~obsY5：各类人体角度时序数据
        self.obsX, self.obsY1, self.obsY2, self.obsY3, self.obsY4, self.obsY5, self.wait = [], [], [], [], [], [], []
        # 画面坐标边界缓存
        self.x1, self.x2, self.y1, self.y2 = [], [], [], []
        # 单帧作弊标记：1=触发作弊，0=正常
        self.hold_head = 0           # 抱头作弊标记（当前代码未启用）
        self.b_l = 0                 # 外物弹力绳作弊标记
        self.s_e = 0                 # 手臂大开借力作弊标记
        self.p = 0                   # 人员替换/替考作弊标记
        self.cl1 = 0                 # 肩背未着地作弊标记
        self.cl2 = 0                 # 上起角度不足作弊标记
        # a~h：每帧状态快照，记录单帧全部作弊+动作状态，用于周期结束汇总作弊
        self.a = []    # 每帧总动作计数num_all快照
        self.b = []    # 躯干状态标记：1=平躺，-1=起身
        self.c = []    # 手臂作弊s_e快照
        self.d = []    # 抱头hold_head快照（未使用）
        self.e = []    # 替考p作弊快照
        self.ff = []   # 起身不足cl2作弊快照
        self.g1, self.g2, self.g3, self.g4 = [], [], [], [] # 预留角度缓存
        self.h = []    # 背未着地cl1作弊快照
        # 逐动作作弊记录列表，保存每一组动作是否违规
        self.body_ground_angle2 = []
        self.shoulder_elbow_angle1 = []
        self.head_holding1 = []
        self.self_texting1 = []
        self.num_l = []              # 每次动作总计数num_all记录
        self.shoulder_ground_angle1 = []
        self.cloth1 = []
        self.if_tool1 = []
        # 全局计数、帧、时间变量
        self.num_index = 0           # 手臂作弊连续帧计数器
        self.time_str = ''           # 格式化计时字符串 分'秒
        self.minute = 0              # 运动分钟
        self.score_index = 0         # 作弊报表图片索引
        self.num = 0                 # 【核心】有效合规仰卧起坐计数（无作弊才算）
        self.frame_id = 0            # 当前处理帧序号，全局自增
        self.index1 = 0
        self.index2 = 0
        self.second = 0              # 运动秒数
        self.num_all = 0             # 【核心】全部动作总计数（含作弊，一落一起即+1）
        # 流程状态标记
        self.if_start = 0            # 0未就绪/2准备完成/1正式测试/-1测试结束
        self.env_ok_count = 0        # 环境检测合格帧计数器
        self.env_ifok = 0            # 环境检测状态：0检测中/1合格/-1~-4各类环境异常
        self.IF_START = []           # 每帧测试状态if_start时序数组
        self.action_time = []        # 单次动作耗时（帧差）
        self.start_frame1 = []       # 每次谷底（平躺）对应的帧ID
        self.start_frame2, self.start_frame3, self.start_frame4 = [], [], []
        self.wucha = 0               # 角度计算误差补偿值
        self.testing = 0             # 1=正在测试，0=未开始
        # 波峰波谷角度检测参数（判断一次完整起落动作）
        self.valley_angle = 0        # 波谷最小躯干角度（平躺最低点）
        self.peak_angle = 0          # 波峰最大躯干角度（起身最高点）
        self.last_count_frame = -999 # 上一次计数帧ID，防重复计数
        self.min_interval = 5        # 动作最小帧间隔，防抖
        self.extremes = [('valley', 0)] # 极值栈：存储(peak/valley, 帧索引)，记录起落拐点
        self.body_ground_angle1 = []  # 躯干角度时序绘图数组
        self.detectsuccess = False   # 环境标志物检测成功标记
        self.nums = []               # 历史有效计数缓存
        self.timestamps = []         # 时间戳缓存
        self.list = []               # 作弊结果汇总表，每一条对应一个动作是否违规
        self.marker_replaced_frames = []  # 每帧标志物坐标替换开关状态缓存
        
        # ===================== 多线程并发队列与锁 =====================
        self.frame_queue = Queue(maxsize=5)       # 待处理帧队列，限制大小防内存溢出
        self.processed_queue = Queue(maxsize=3)   # 处理完成帧队列
        self.io_queue = Queue()                   # 异步IO队列：保存报表、绘图
        self.lock = threading.RLock()             # 可重入锁，保护多线程共享状态
        self.rw_lock = ReadWriteLock()            # 读写分离锁，高频读低频写优化
        self.running = False                      # 线程运行总开关
        self.stop_event = threading.Event()       # 线程停止信号
        self.last_plot_save_time = 0              # 上次绘图保存时间戳
        self.plot_save_interval = 1.0             # 绘图保存间隔(秒)，降低IO负载
        self.last_score_update = 0                # 上次作弊报表更新时间戳
        self.score_update_interval = 0.75        # 报表刷新间隔(秒)
        self.score_overlay = None                 # 报表缩放后贴图缓存
        self.score_image_raw = None               # 报表原始图片缓存
        
        super(SITUP, self).__init__()

    def situp_start(self, frame, WIDTH, HEIGHT):
        """
        单帧仰卧起坐主处理函数
        输入：原图frame、原始画面宽高
        输出：处理后画面、报表图、有效计数num、总计数num_all、测试状态、作弊记录列表
        """
        global body_ground_angle2
        # 画面坐标放大缩放
        WIDTH, HEIGHT = WIDTH * 7, HEIGHT * 15
        window_size, delta_threshold = 5, 5 # 滑动窗口极值检测参数
        s1 = "\""
        s2 = "\'"
        num_old = self.num_all # 处理前总动作数，用于判断是否新完成一次动作
        self.X1.append('0')
        # 调用人体关键点处理函数：返回标注帧、是否检测到人、帧耗时、人体23关键点坐标、地面基线k0/b0、角度基线k1、标志物替换开关
        frame1, if_existperson, start_time, position, k0, b0, k1, marker_replaced, marker_circle = process_frame(
            frame, self.frame_id, WIDTH, HEIGHT, self.testing, self.IF_START
        )
        marker_replace_enabled = False # 当前帧是否启用标志物坐标修正
        # ========== 标志物坐标替换逻辑 ==========
        if if_existperson == 1:
            self.marker_replaced_frames.append(marker_replaced)
            # 仅环境合格+进入预备阶段，且历史无测试帧时，才校验标志物稳定度
            if self.env_ifok == 1 and self.if_start == 2 and 1 not in self.IF_START:
                if len(self.marker_replaced_frames) >= 10:
                    true_count = sum(self.marker_replaced_frames)
                    total_count = len(self.marker_replaced_frames)
                    # 连续10帧90%以上识别到标志物，开启坐标替换
                    if true_count / total_count >= 0.9:
                        marker_replace_enabled = True
            # 开启替换则检测圆形标志物，覆盖脚底关键点坐标
            if marker_replace_enabled:
                if marker_circle is not None:
                    mx, my, mr = marker_circle
                    position[23][1], position[23][2] = mx, my
        # ========== 运动计时逻辑 ==========
        if self.num >= 0:
            if not hasattr(self, 'start_time'):
                self.start_time = time.time()
            run_time = time.time()
            total_seconds = int(run_time - self.start_time)
            self.minute, self.second = divmod(total_seconds, 60)
            self.time_str = f'{self.minute}{s2}{self.second}{s1}'
        # 异步写入关键点坐标，不阻塞主流程
        # self.io_queue.put(('save_position', self.frame_id, position))

        frame2 = frame1.copy() # 绘制用画布副本

        # ========== 分支1：未检测到人体 ==========
        if if_existperson == 0:
            frame2 = cv2.line(frame2, (x01, y002), (x02, y001), (0, 255, 0), 3)
            scaler = 1
            failure_str = '未检测到人!'
            frame2 = cv2ImgAddText(frame2, failure_str, 25 * scaler, 100 * scaler, (255, 0, 255), 40, drawBox=False)
        # ========== 分支2：成功检测到人体，核心角度/作弊/计数全逻辑 ==========
        elif if_existperson == 1:
            scaler = 1
            b_g, b_g_1, b_g_2, b_g_3, b_g_4 = 0, 0, 0, 0, 0 # b_g：1平躺/-1起身，躯干状态标记

            # ========== 计算躯干与地面夹角 body_ground_angle ==========
            A1, B1 = [], []
            line = position
            A1.append(float(line[11][1])) # 左肩X
            A1.append(float(line[23][1])) # 脚底X
            A1.append(float(640))
            B1.append(float(line[11][2])) # 左肩Y
            B1.append(float(line[23][2])) # 脚底Y
            B1.append(float(line[23][2]))
            body_ground_angle = AngleCalculate(A1, B1)
            self.wucha = 8 # 固定角度补偿误差
            self.obsY3.append(body_ground_angle - abs((math.atan(k0)) / math.pi * 180))

            # ========== 计算左手臂肩肘夹角 shoulder_elbow_angle ==========
            A2, B2 = [], []
            A2.append(float(line[16][1]))
            A2.append(float(line[14][1]))
            A2.append(float(line[12][1]))
            B2.append(float(line[16][2]))
            B2.append(float(line[14][2]))
            B2.append(float(line[12][2]))
            shoulder_elbow_angle = AngleCalculate(A2, B2)

            # ========== 计算右手臂肩肘夹角 shoulder_elbow_angle2 ==========
            A6, B6 = [], []
            A6.append(float(line[15][1]))
            A6.append(float(line[13][1]))
            A6.append(float(line[11][1]))
            B6.append(float(line[15][2]))
            B6.append(float(line[13][2]))
            B6.append(float(line[11][2]))
            shoulder_elbow_angle2 = AngleCalculate(A6, B6)
            self.obsY2.append(shoulder_elbow_angle)
            self.obsY4.append(shoulder_elbow_angle2)

            # ========== 脚底地面基准夹角 base_angle ==========
            A4, B4 = [], []
            A4.append(float(line[23][1]))
            A4.append(float(line[29][1]))
            A4.append(float(640))
            B4.append(float(line[23][2]))
            B4.append(float(line[29][2]))
            B4.append(float(line[29][2]))
            base_angle = AngleCalculate(A4, B4)
            self.obsY5.append(body_ground_angle - abs((math.atan(k1)) / math.pi * 180))

            # 画面左上角绘制有效计数、计时文本
            frame2 = cv2ImgAddText(frame2, f'{max(0, self.num)}', 25 * scaler, 60 * scaler, (0, 0, 255), 40, drawBox=False)
            frame2 = cv2ImgAddText(frame2, f'{self.time_str}', 25 * scaler, 105 * scaler, (255, 165, 0), 40, drawBox=False)

            # ========== 环境状态文字提示分支（env_ifok区分环境异常） ==========
            if self.env_ifok == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '测试环境检测中', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '请准备', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
            elif self.env_ifok == 1 and self.if_start == 2 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '检测成功！', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '请开始测试', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
                self.detectsuccess=True
            elif self.env_ifok == -1 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '脚出画框，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -2 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '头部出画框，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -3 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '斜度大于15°，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -4 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '请调整测试位置', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)

            # ========== 环境合格判定逻辑：肩、头、脚关键点在地面线下方 ==========
            if position[29][2] <= k0 * position[29][1] + b0 + 15 and position[23][2] <= k0 * position[23][1] + b0 + 15 and position[11][2] <= k0 * position[11][1] + b0 + 15:
                if position[3][2] > 0 and position[3][1] > 0:
                    # 双脚在画面内，累计合格帧
                    if position[31][2] <= 480 and position[32][2] <= 480 and position[31][1] <= 640 and position[32][1] <= 640:
                        self.env_ok_count += 1
                        # 连续10帧环境正常，标记环境合格env_ifok=1
                        if self.frame_id >= 20 and self.env_ok_count >= 10:
                            self.env_ifok = 1
                    else:
                        self.env_ifok = -1 # 脚出框
                        self.env_ok_count = 0
                else:
                    self.env_ifok = -2 # 头部关键点丢失
                    self.env_ok_count = 0
            else:
                self.env_ifok = -4 # 躯干高于地面基准线，位置偏移
                self.env_ok_count = 0

            # ========== 判定是否进入预备阶段if_start=2 ==========
            # 躯干脚底夹角差值<=10，环境合格，累计预备帧
            if body_ground_angle - base_angle <= 10 and self.env_ifok == 1:
                self.frametime.append(self.frame_id)
                self.if_start = 2
                # 10帧预备姿态，正式开启测试 if_start=1
                if len(self.frametime) >= 10:
                    self.if_start = 1
            else:
                self.if_start = 0
            self.IF_START.append(self.if_start)

            # ========== 正式测试阶段主逻辑（1在IF_START即开始计数+作弊检测） ==========
            if 1 in self.IF_START:
                self.testing = 1
                self.wait.append(self.frame_id)
                # 测试结束文字
                if -1 in self.IF_START:
                    frame2 = cv2ImgAddText(frame2, '测试结束！', 25 * scaler, 50 * scaler, (0, 0, 255), 40, drawBox=False)

                # ========== 修正躯干夹角body_ground_angle2（分段补偿误差） ==========
                if body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 15:
                    body_ground_angle2 = body_ground_angle - abs((math.atan(k0)) / math.pi * 180) + self.wucha
                if 15 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 30:
                    body_ground_angle2 = body_ground_angle - abs((math.atan(k0)) / math.pi * 180) + self.wucha + 10
                if 30 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180) <= 40:
                    body_ground_angle2 = body_ground_angle - abs((math.atan(k0)) / math.pi * 180) + self.wucha + 20
                if 30 < body_ground_angle - abs((math.atan(k0)) / math.pi * 180):
                    body_ground_angle2 = 60
                if body_ground_angle2 >= 60:
                    body_ground_angle2 = 60
                self.obsY1.append(body_ground_angle2)

                # ========== 各类作弊帧检测，赋值作弊标记 ==========
                # 作弊1：手臂夹角≥150，连续3帧判定手臂大开借力 s_e=1
                if shoulder_elbow_angle >= 150:
                    self.num_index = self.num_index + 1
                if self.num_index >= 3:
                    frame2 = cv2ImgAddText(frame2, '手臂打开借力', 25 * scaler, 350 * scaler, (255, 0, 255), 40)
                    self.s_e = 1
                print(shoulder_elbow_angle)
                # 作弊2：人体关键点大面积出画面，判定替考换人 p=1
                if (min(position[2][2], position[5][2]) < 0 or min(position[7][1], position[8][1]) < 0 or max(position[7][1], position[8][1]) > 640 or
                        max(position[25][2], position[26][2]) > 480 or min(position[25][1], position[26][1]) < 0 or max(position[25][1], position[26][1]) > 640):
                    frame2 = cv2ImgAddText(frame2, '测试者被替换', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
                    self.p = 1

                # ========== 滑动窗口极值检测：识别动作波谷(平躺)、波峰(起身) ==========
                if len(self.obsY1) > 2 * window_size + 1:
                    # 窗口为最后的 2 * window_size + 1 个数据
                    idx = len(self.obsY1) - window_size - 1
                    local_window = self.obsY1[idx - window_size: idx + window_size + 1]
                    center_value = self.obsY1[idx]
                    # 判断窗口内最大值=当前值 → 波峰起身
                    is_peak = center_value == max(local_window) and (max(local_window) - min(local_window)) >= delta_threshold
                    # 判断窗口内最小值=当前值 → 波谷平躺落地
                    is_valley = center_value == min(local_window) and (max(local_window) - min(local_window)) >= delta_threshold
                    if is_peak:
                        self.extremes.append(('peak', idx))
                        self.peak_angle = center_value
                    elif is_valley:
                        self.extremes.append(('valley', idx))
                        self.start_frame1.append(self.frame_id)
                        self.valley_angle = center_value

                # 作弊3：波谷角度>30 → 肩背未贴地 cl1=1
                if self.valley_angle > 30:
                    frame2 = cv2ImgAddText(frame2, '肩背未着地', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                    self.cl1 = 1
                # 作弊4：波峰角度8~40 → 起身高度不足 cl2=1
                if 8 < self.peak_angle < 40:
                    frame2 = cv2ImgAddText(frame2, '上起未达角度', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                    self.cl2 = 1

                # 躯干状态标记：≤30平躺(1)，≥40起身(-1)
                if body_ground_angle2 <= 30:
                    b_g = 1
                if body_ground_angle2 >= 40:
                    b_g = -1

                # 每帧快照存入数组，周期结束汇总作弊状态
                self.a.append(self.num_all)
                self.b.append(b_g)
                self.c.append(self.s_e)
                self.d.append(self.hold_head)
                self.e.append(self.p)
                self.ff.append(self.cl2)
                self.h.append(self.cl1)
                self.Z.append(self.b_l)
                self.g1.append(b_g_1)
                self.g2.append(b_g_2)
                self.g3.append(b_g_3)
                self.g4.append(b_g_4)

                # ========== 【有效计数num累加逻辑（当前漏洞核心）】 ==========
                # 动作完成标记：b同时存在平躺1、起身-1，当前帧是起身状态，且瞬时作弊标记全为0
                # 缺陷：仅判断当前帧无作弊，未校验本动作周期历史快照是否存在作弊
                if 1 in self.b and -1 in self.b and self.b[-1] == -1 and self.s_e != 1 and self.cl1 != 1 and self.cl2 != 1:
                    self.num = self.num + 1
                    self.b = [] # 清空躯干状态缓存，等待下一轮起落
                    self.X1.append(self.num)
                # 手臂作弊清零重置计数器
                if shoulder_elbow_angle2 <= 150 and shoulder_elbow_angle <= 150 and self.s_e == 1:
                    self.b = []
                    self.s_e = 0
                    self.num_index = 0
                # 背未着地作弊清零
                if self.valley_angle < 30 and self.cl1 == 1:
                    self.b = []
                    self.cl1 = 0
                    self.index1 = 0
                # 起身不足作弊清零
                if self.peak_angle > 40 and self.cl2 == 1:
                    self.b = []
                    self.cl2 = 0
                    self.index2 = 0

                # ========== 总计数num_all累加逻辑（无作弊校验，只要一谷一峰就+1） ==========
                if len(self.extremes) >= 2:
                    last_three = self.extremes[-2:]
                    # 拐点顺序：先谷(平躺)、后峰(起身)，构成一次完整动作周期
                    if last_three[0][0] == 'valley' and last_three[1][0] == 'peak':
                        # 起落角度差大于阈值，判定有效动作周期
                        if self.peak_angle - self.valley_angle > 10:
                            self.num_all = self.num_all + 1
                            if self.start_frame1:
                                self.action_time.append(self.frame_id - self.start_frame1[0])
                        self.start_frame1 = []
                        self.extremes = [] # 清空极值栈，检测下一组起落
                # 兜底：总计数不低于有效计数
                if self.num_all < self.num:
                    self.num_all = self.num
                    self.extremes = []

        self.obsX.append(self.frame_id / 25)

        # 绘图数组长度对齐，防止维度不匹配报错
        if len(self.X) != len(self.obsY1):
            min_len = min(len(self.X), len(self.obsY1))
            self.X = self.X[:min_len]
            self.obsY1 = self.obsY1[:min_len]

        # ========== 单次新动作完成，生成作弊记录报表 ==========
        if num_old < self.num_all and self.num_all >= 0:
            # 累计9个动作刷新一次报表缓存
            if len(self.num_l) >= 9:
                self.body_ground_angle1, self.shoulder_elbow_angle1, self.head_holding1, self.self_texting1, self.num_l, self.shoulder_ground_angle1, self.cloth1, self.if_tool1 = [], [], [], [], [], [], [], []
                self.score_index = self.score_index + 1
            self.num_l.append(self.num_all)
            # 作弊类型描述映射
            res = ('-', '肘关节是否打开', '是否有替考', '肩背是否着地', '上起是否符合要求', '是否有弹力绳')
            r = 0
            # 遍历本动作周期快照，标记是否存在各类作弊
            if 1 not in self.c and self.cl2 != 1 and self.cl1 != 1:
                self.body_ground_angle1.append('.')
            else:
                self.body_ground_angle1.append('×')
            if 1 in self.c:
                self.shoulder_elbow_angle1.append('×')
            else:
                self.shoulder_elbow_angle1.append('.')
            if 1 in self.e:
                self.self_texting1.append('×')
            else:
                self.self_texting1.append('.')
            if 1 in self.h:
                self.shoulder_ground_angle1.append('×')
            else:
                self.shoulder_ground_angle1.append('.')
            if 1 in self.ff:
                self.cloth1.append('×')
            else:
                self.cloth1.append('.')
            if 1 in self.Z:
                self.if_tool1.append('×')
            else:
                self.if_tool1.append('.')

            # 判断本次动作主要作弊类型
            if self.s_e == 1 or self.cl2 == 1 or self.cl1 == 1:
                if 1 in self.c:
                    r = 1
                if 1 in self.e:
                    r = 2
                if 1 in self.h:
                    r = 3
                if 1 in self.ff:
                    r = 4
                if 1 in self.Z:
                    r = 5
            else:
                r = 0
            # 清空本周期快照缓存，准备下一次动作记录
            self.a, self.c, self.d, self.e, self.Z, self.ff, self.h = [], [], [], [], [], [], []
            # 存入总作弊记录列表
            self.list.append([f'第{self.num_all}个动作', self.body_ground_angle1[-1], res[r], '-'])
        # 组装作弊报表DataFrame
        data = {
            '计数': self.num_l,
            '动作是否计数': self.body_ground_angle1,
            '肘关节是否打开': self.shoulder_elbow_angle1,
            '是否有替考': self.self_texting1,
            '肩背是否着地': self.shoulder_ground_angle1,
            '上起是否符合要求': self.cloth1,
            '是否有弹力绳': self.if_tool1,
        }
        df = pd.DataFrame(data)

        # ========== 报表图片渲染+画面贴报表 ==========
        current_time = time.time()
        score_path = rf"{datapath1}\score{self.score_index}.jpg"
        # 到达刷新间隔则重新生成报表图
        if current_time - self.last_score_update >= self.score_update_interval or self.score_overlay is None:
            dfi.export(df.T, score_path, table_conversion="matplotlib")
            img1 = mpimg.imread(score_path)
            img = (img1 * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_h, img_w = img.shape[:2]
            frame_h, frame_w = frame2.shape[:2]
            max_w = int(frame_w * 0.375)
            max_h = int(frame_h * 0.375)
            scale_ratio = min(max_w / img_w, max_h / img_h)
            new_w = int(img_w * scale_ratio)
            new_h = int(img_h * scale_ratio)
            resized_img = cv2.resize(img, (new_w, new_h))
            self.score_overlay = resized_img
            self.score_image_raw = img1
            self.last_score_update = current_time
        else:
            img1 = self.score_image_raw
        # 报表贴到画面右上角
        if self.score_overlay is not None:
            overlay_h, overlay_w = self.score_overlay.shape[:2]
            frame2[0:overlay_h, -overlay_w:] = self.score_overlay

        self.X.append(self.frame_id)
        self.frame_id += 1

        # 异步批量保存绘图数据，降低IO压力
        current_time = time.time()
        if current_time - self.last_plot_save_time >= self.plot_save_interval:
            self.io_queue.put(('save_plots', {
                'body_ground': (self.X.copy(), self.obsY1.copy(), self.X1.copy()),
                'body_ground2': (self.X.copy(), self.obsY3.copy(), self.X1.copy()),
                'body_ground3': (self.X.copy(), self.obsY5.copy(), self.X1.copy()),
                'exercise_time': (self.X.copy(), self.action_time.copy(), self.X1.copy()),
            }))
            self.last_plot_save_time = current_time
        # 返回输出：处理帧、原图报表、有效计数、总计数、测试状态、作弊明细
        yield frame2, img1, self.num, self.num_all, self.IF_START, self.list

    def _frame_capture_thread(self, cap, WIDTH, HEIGHT):
        """帧采集线程：从摄像头读取帧并放入队列"""
        try:
            while self.running and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 如果队列满了，丢弃最旧的帧，保持最新帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                self.frame_queue.put((frame.copy(), WIDTH, HEIGHT), timeout=0.1)
        except Exception as e:
            print(f"帧采集线程错误: {e}")
        finally:
            print("帧采集线程结束")

    def _frame_process_thread(self):
        """帧处理线程：从队列取帧并处理"""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    frame, WIDTH, HEIGHT = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                try:
                    # 处理帧
                    frame_data_generator = self.situp_start(frame, WIDTH, HEIGHT)
                    frame2, img1, num, num_all, IF_START, list_data = next(frame_data_generator)
                    
                    # 将处理后的帧放入队列
                    if self.processed_queue.full():
                        try:
                            self.processed_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.processed_queue.put((frame2, img1, num, num_all, IF_START, list_data), timeout=0.1)
                except Exception as e:
                    print(f"帧处理错误: {e}")
                    continue
        except Exception as e:
            print(f"帧处理线程错误: {e}")
        finally:
            print("帧处理线程结束")

    def _io_thread(self):
        """后台I/O线程：处理文件读写操作"""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    task = self.io_queue.get(timeout=0.5)
                except Empty:
                    continue
                
                try:
                    if task[0] == 'save_position':
                        _, frame_id, position = task
                        with open(fr"{datapath1}\{frame_id}.txt", 'w') as f:
                            f.write(f'{position}\n')
                    elif task[0] == 'save_plots':
                        _, plots_data = task
                        save_plot(fr'{datapath1}\body_ground.png', 
                                 plots_data['body_ground'][0], 
                                 plots_data['body_ground'][1], 
                                 plots_data['body_ground'][2])
                        save_plot(fr'{datapath1}\body_ground2.png', 
                                 plots_data['body_ground2'][0], 
                                 plots_data['body_ground2'][1], 
                                 plots_data['body_ground2'][2])
                        save_plot(fr'{datapath1}\body_ground3.png', 
                                 plots_data['body_ground3'][0], 
                                 plots_data['body_ground3'][1], 
                                 plots_data['body_ground3'][2])
                        save_plot(fr'{datapath1}\exercise_time.png', 
                                 plots_data['exercise_time'][0], 
                                 plots_data['exercise_time'][1], 
                                 plots_data['exercise_time'][2])
                    elif task[0] == 'save_img':
                        _, img1 = task
                        cv2.imwrite("img1.jpeg", img1)
                except Exception as e:
                    print(f"I/O操作错误: {e}")
        except Exception as e:
            print(f"I/O线程错误: {e}")
        finally:
            print("I/O线程结束")

    def start_video_processing(self, if_open, WIDTH, HEIGHT, video_path: Optional[str] = None):
        """实时/离线统一入口：仿照 tools/situp_video.py 逐帧读取并分析。

        - video_path=None：读取摄像头
        - video_path!=None：读取指定视频文件
        - 保存视频：按输入源自身 fps 写出（不再降速 1/3）
        """
        if if_open == 1:
            cv2.namedWindow('Situp Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Situp Detection', WIDTH, HEIGHT)

            if video_path:
                cap = cv2.VideoCapture(video_path)
            else:
                cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # 打开默认摄像头

            if not cap.isOpened():
                if video_path:
                    print(f"无法打开视频: {video_path}")
                else:
                    print("无法打开摄像头")
                return

            input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or WIDTH
            input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or HEIGHT
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(FILENAME, fourcc, fps, (input_w, input_h))

            # 启动后台 I/O 线程（用于写 txt / 图表 / img1）
            self.running = True
            self.stop_event.clear()
            io_thread = threading.Thread(target=self._io_thread, daemon=True)
            io_thread.start()

            INTERVAL = 0.5  # 秒
            last_exec = time.monotonic() - INTERVAL

            try:
                while self.running and not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 逐帧处理
                    try:
                        frame_data_generator = self.situp_start(frame, input_w, input_h)
                        frame2, img1, num, num_all, IF_START, list_data = next(frame_data_generator)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"处理帧时出错: {e}")
                        continue

                    # 写入共享数据（采集线程 - 写锁）
                    with self.rw_lock.write_lock():
                        self.num = num
                        self.num_all = num_all
                        self.nums.append(num)
                        self.timestamps.append(datetime.now().isoformat(timespec="milliseconds"))

                    # 每 0.5s 记录一次 runtime_data.json + 保存 img1
                    now = time.monotonic()
                    if now - last_exec >= INTERVAL:
                        # 读取共享数据（分析线程 - 读锁）
                        with self.rw_lock.read_lock():
                            record = {
                                "username": self.username,
                                "nums": self.nums.copy(),
                                "num": self.num,
                                "num_all": self.num_all,
                                "timestamps": self.timestamps.copy(),
                                "angles": self.obsY1.copy() if self.obsY1 else [],
                                "detectsuccess": self.detectsuccess,
                                "finished": False,
                            }
                        _append_record(record)
                        # 异步保存图片
                        self.io_queue.put(('save_img', img1))
                        last_exec = now
                    
                    # 显示和保存视频
                    cv2.imshow('Situp Detection', frame2)
                    out.write(frame2)
                    
                    # 检查退出条件
                    if -1 in IF_START:
                        # 读取共享数据（分析线程 - 读锁）
                        with self.rw_lock.read_lock():
                            end_record = {
                                "username": self.username,
                                "nums": self.nums.copy(),
                                "num": self.num,
                                "num_all": self.num_all,
                                "timestamps": self.timestamps.copy(),
                                "angles": self.obsY1.copy() if self.obsY1 else [],
                                "detectsuccess": False,
                                "finished": True,
                            }
                        _append_record(end_record)
                        print("测试结束")
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                self.running = False
                self.stop_event.set()
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                io_thread.join(timeout=2.0)

            # 测试完成，打印结果
            with self.rw_lock.read_lock():
                score0 = int(getattr(self, "num", 0))
                score1 = int(getattr(self, "num_all", 0))
            print(f"测试完成！计数: {score0}, 总数: {score1}")
            print(f"结果视频已保存: {FILENAME}")

        elif if_open == -1:
            save_plot(fr'{datapath1}\body_ground.png', self.X, self.obsY1, self.X1)
            save_plot(fr'{datapath1}\body_ground2.png', self.X, self.obsY3, self.X1)
            save_plot(fr'{datapath1}\body_ground3.png', self.X, self.obsY5, self.X1)
            save_plot(fr'{datapath1}\exercise_time.png', self.X, self.action_time, self.X1)

        elif if_open == 0:
            pass

    def start_video_file(self, file_path: str, WIDTH=640, HEIGHT=480, save_output: bool = True):
        """离线视频文件解析：仿照 tools/situp_video.py 的方式读取视频路径并逐帧分析。

        - 输入：file_path 指定视频文件
        - 输出：默认按输入 fps 保存结果视频到 “原文件名-situp-result.mp4”
        - 处理：逐帧调用 situp_start，逻辑与实时检测保持一致
        """
        if not os.path.exists(file_path):
            print(f"视频不存在: {file_path}")
            return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"无法打开视频: {file_path}")
            return

        input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or WIDTH
        input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or HEIGHT
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

        out = None
        out_path = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = f"{os.path.splitext(file_path)[0]}-situp-result.mp4"
            out = cv2.VideoWriter(out_path, fourcc, fps, (input_w, input_h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理单帧（保持与实时模式一致：不在这里额外翻转输入帧）
                try:
                    frame_data_generator = self.situp_start(frame, input_w, input_h)
                    frame2, img1, num, num_all, IF_START, list_data = next(frame_data_generator)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    continue

                if out is not None:
                    out.write(frame2)

                # 遇到结束标记提前退出
                if -1 in IF_START:
                    break
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            if out_path:
                print(f"结果已保存: {out_path}")

    def start(self):
        # self.interval = 0.5  # 秒
        # self.last_exec = time.monotonic() - self.interval

        print("测试多进程")
        # 多进程
        self.conn_to_detect, self.conn_detect_in = Pipe(duplex=False)
        self.conn_to_test, self.conn_test_in = Pipe(duplex=False)
        self.conn_to_result, self.conn_result_in = Pipe(duplex=False)

        self.detect_process = PointDetectionProcess(self.conn_detect_in, self.conn_to_test)
        self.test_process = TestProcess(self.conn_test_in, self.conn_to_result)

        self.detect_process.start()
        self.test_process.start()
        print("测试多进程-finish")

        return self.conn_to_detect, self.conn_result_in
        

    # def update(self, frame, frame_id):
    #     # 处理帧数据

    #     self.conn_detect_in.send({
    #         "frame": frame,
    #         "frame_id": frame_id,
    #         "drawing": True
    #     })

        # result = {
        #     "uid": self.username,
        #     "score": self.num,
        #     "frame": frame_id,
        #     "num_all": self.num_all,
        #     "timestamp": int(round(time.time() * 1000)),
        #     "angle": self.obsY1[-1] if self.obsY1 else 0,
        # }
    
        # return result, frame2
    
    def stop(self):
        self.conn_to_detect.close()

        # 等待所有进程结束
        wait_time = 3
        while wait_time > 0:
            time.sleep(0.1)
            if self.detect_process and not self.detect_process.is_alive():
                self.detect_process.join()
                self.detect_process = None
            
            if self.test_process and not self.test_process.is_alive():
                self.test_process.join()
                self.test_process = None
            time.sleep(0.4)
            wait_time -= 0.5
        
        # 关闭所有进程
        if self.detect_process:
            self.detect_process.terminate()
            self.detect_process.join()
        if self.test_process:
            self.test_process.terminate()
            self.test_process.join()
            






# 完成个数：self.num
# 完成总个数：self.num_all
# 处理帧数据：frame2
# 完成判据表格：img1
# 数据图：body_ground.png，body_ground2.png， body_ground3.png, (self.X(frame_id), self.obsY1(判断标准、角度))
# 检测视频保存地址：FILENAME（fr'video/{filename}-situp.mp4' filename为当前时刻），保存在本地需上传到数据库

 # 张三——204.4.25 10.09.09——仰卧起坐、立定跳远....、pdf——jpg、txt、MP4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--video",
        default=None,
        help="输入视频路径（可选；传了则分析视频，不传则使用摄像头实时检测）",
    )
    args = parser.parse_args()

    situp_detector = SITUP()
    if args.video:
        # 有视频路径 -> 分析视频文件
        situp_detector.start_video_processing(if_open=1, WIDTH=640, HEIGHT=480, video_path=args.video)
    else:
        # 无视频路径 -> 打开摄像头实时检测
        situp_detector.start_video_processing(if_open=1, WIDTH=640, HEIGHT=480)
