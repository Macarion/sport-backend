import math
import cv2
import mediapipe as mp
import time
import os
import threading
import contextlib


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
from typing import Optional, List, Dict, Any
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
        upper_green = np.array([77, 255, 255])  # 设定绿色的阈值上限

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
            param2=10,
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

mp_pose = mp.solutions.pose
filename = time.strftime('%Y-%m-%d %H %M %S', time.localtime())
datapath = fr'video/{filename}-situp'

mp_drawing = mp.solutions.drawing_utils
global frame_id, K
K, B, Y1, Y2 = [], [], [], []
frame_id_c, shoulder_x, shoulder_y, heal_x, heal_y = [], [], [], [], []

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=0,  # 降低复杂度以提升速度（0最快，1中等，2最慢）
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    enable_segmentation=False) 

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

def process_frame(img, frame_id, WIDTH, HEIGHT, testing, IF_START):
    global position, y2, y1, x2, x1, x01, y01, x02, y02, y001, y002, k0, k, b, x3
    position = []
    k0, k, b = 0, 0, 0
    x01, y001 = 0, 440
    x02, y002 = 640, 240
    y01, y02 = y001, y002
    k1 = (y001-y002)/(x01-x02)
    x3 = 320
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 新建一个空白图像，用于绘制结果
    # img = np.zeros((h, w, 3), dtype=np.uint8)
    img = img.copy()

    results = pose.process(img_RGB)

    # 标志物检测与关键点替换
    marker_circle = _detect_marker_circle_bgr(img)
    marker_replaced = False
    if marker_circle is not None:
        mx, my, mr = marker_circle
        # 绘制标志物圆心和外轮廓（在骨架绘制之前）
        cv2.circle(img, (mx, my), mr, (0, 0, 255), 3)
        cv2.circle(img, (mx, my), 3, (255, 255, 0), -1)
        # 在标志物旁边添加文字标注
        cv2.putText(img, "Marker", (mx + mr + 10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        marker_replaced = True

    if results.pose_landmarks:
        if_existperson = 1

        # 如果检测到标志物，替换 position[23]（左臀关键点）的坐标
        if marker_replaced:
            results.pose_landmarks.landmark[23].x = mx / w
            results.pose_landmarks.landmark[23].y = my / h

        # 使用替换后的关键点绘制骨骼连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):

            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z
            radius = 2
            position.append([i, cx, cy, cz])
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

        x1 = int(results.pose_landmarks.landmark[11].x * w)
        y1 = int(results.pose_landmarks.landmark[11].y * h)
        x2 = int(results.pose_landmarks.landmark[29].x * w)
        y2 = int(results.pose_landmarks.landmark[29].y * h)
        k = (y2 - y1) / (x2 - x1)
        if testing == 0:
            img = cv2.line(img, (x01, y001), (x02, y002), (0, 255, 0), 3)
            img = cv2.circle(img, (x3, int(x3*k1+y001)), 5, (255, 0, 0), -1)
        if abs(math.degrees(math.atan(k0)) - math.degrees(math.atan(k1))) >= 5 and 1 not in IF_START:
            shoulder_x.append(x1)
            shoulder_y.append(y1)
            heal_x.append(x2)
            heal_y.append(y2)
        k0 = (shoulder_y[-1] - heal_y[-1])/(shoulder_x[-1] - heal_x[-1]) if shoulder_x[-1] - heal_x[-1] != 0 else 0
        b = shoulder_y[-1] - k0*shoulder_x[-1]
        y01, y02 = int(k0*x01+b), int(k0*x02+b)
        img = cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)
    else:
        if_existperson = 0
        # if not shoulder_x:
        #     img = cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)
        #     k0 = (y02 - y01) / (x02 - x01)
        # else:
        #     k0 = (shoulder_y[0] - heal_y[0]) / (shoulder_x[0] - heal_x[0])
        #     b = shoulder_y[0] - k * shoulder_x[0]
        #     y01, y02 = int(k * x01 + b), int(k * x02 + b)
        #     img = cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)
    return img, if_existperson, start_time, position, k0, b, k1, marker_replaced

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
        self.L, self.X, self.X1, self.Lk_filter, self.HX, self.Z, self.frametime, self.leg_lengthr, self.leg_lengthl, self.HY = [], [], [], [], [0], [], [], [], [], [450]
        self.line1, self.line2, self.line3, self.line4 = None, None, None, None
        self.obsX, self.obsY1, self.obsY2, self.obsY3, self.obsY4, self.obsY5, self.wait = [], [], [], [], [], [], []
        self.x1, self.x2, self.y1, self.y2 = [], [], [], []
        self.time_list = []
        self.hold_head, self.b_l, self.s_e, self.p, self.cl1, self.cl2 = 0, 0, 0, 0, 0, 0
        self.a, self.b, self.c, self.d, self.e, self.ff, self.g1, self.g2, self.g3, self.g4, self.h = [], [], [], [], [], [], [], [], [], [], []
        self.body_ground_angle2, self.shoulder_elbow_angle1, self.head_holding1, self.self_texting1, self.num_l, self.shoulder_ground_angle1, self.cloth1, self.if_tool1 = [], [], [], [], [], [], [], []
        self.num_index, self.time_str, self.minute, self.score_index, self.num, self.frame_id, self.index1, self.index2, self.second, self.num_all = 0, '', 0, 0, 0, 0, 0, 0, 0, 0
        self.if_start, self.env_ifok = 0, 0
        self.last, self.IF_START, self.doing, self.action_time, self.start_frame1, self.start_frame2, self.start_frame3, self.start_frame4 = [], [], [], [], [], [], [], []
        self.wucha, self.testing = 0, 0
        self.valley_angle, self.peak_angle, self.last_count_frame, self.min_interval = 0, 0, -999, 5
        self.extremes = [('valley', 0)]
        self.body_ground_angle1 = []
        self.detectsuccess=False
        self.nums = []
        self.timestamps = []
        self.body_ground_angle1, self.list = [], []
        self.marker_replaced_frames = []  # 记录每帧的 marker_replaced 状态
        
        # 并发处理相关
        self.frame_queue = Queue(maxsize=5)  # 限制队列大小，避免内存溢出
        self.processed_queue = Queue(maxsize=3)  # 处理后的帧队列
        self.io_queue = Queue()  # 文件I/O操作队列
        self.lock = threading.RLock()  # 保护共享状态的锁（可重入锁，支持同一线程多次获取）
        self.rw_lock = ReadWriteLock()  # 读写锁，保护共享数据的读写
        self.running = False
        self.stop_event = threading.Event()
        self.last_plot_save_time = 0
        self.plot_save_interval = 1.0  # 图表保存间隔（秒），减少I/O频率
        self.last_score_update = 0
        self.score_update_interval = 0.75
        self.score_overlay = None
        self.score_image_raw = None
        
        super(SITUP, self).__init__()
    def situp_start(self, frame, WIDTH, HEIGHT):
        global sl, bl, sr, br, body_ground_angle2
        # sl =bl=sr=br=
        WIDTH, HEIGHT = WIDTH * 7, HEIGHT * 15
        window_size, delta_threshold = 5, 5
        s1 = "\""
        s2 = "\'"
        num_old = self.num_all
        self.X1.append('0')
        frame1, if_existperson, start_time, position, k0, b0, k1, marker_replaced = process_frame(frame, self.frame_id, WIDTH, HEIGHT,
                                                                                self.testing,self.IF_START)
        # 记录每帧的 marker_replaced 状态
        marker_replace_enabled = False
        if if_existperson == 1:
            self.marker_replaced_frames.append(marker_replaced)
            # 在环境检测成功之前，判断是否超过90%的帧检测到标志物
            # 只有超过90%才进行坐标替换
            if self.env_ifok == 1 and self.if_start == 2 and 1 not in self.IF_START:
                if len(self.marker_replaced_frames) >= 10:
                    true_count = sum(self.marker_replaced_frames)
                    total_count = len(self.marker_replaced_frames)
                    if true_count / total_count >= 0.9:
                        marker_replace_enabled = True
            # 只有在 marker_replace_enabled 为 True 时才替换坐标
            if marker_replace_enabled:
                marker_circle = _detect_marker_circle_bgr(frame)
                if marker_circle is not None:
                    mx, my, mr = marker_circle
                    position[23][1], position[23][2] = mx, my
        # 计时
        if self.num >= 0:
            if not hasattr(self, 'start_time'):
                self.start_time = time.time()
            run_time = time.time()
            total_seconds = int(run_time - self.start_time)
            self.minute, self.second = divmod(total_seconds, 60)
            self.time_str = f'{self.minute}{s2}{self.second}{s1}'
        # 存运动数据 - 改为异步I/O
        self.io_queue.put(('save_position', self.frame_id, position))

        frame2 = cv2.flip(frame1, 1)

        if if_existperson == 0:
            frame2 = cv2.line(frame2, (x01, y002), (x02, y001), (0, 255, 0), 3)
            scaler = 1
            failure_str = '未检测到人!'
            frame2 = cv2ImgAddText(frame2, failure_str, 25 * scaler, 100 * scaler, (255, 0, 255), 40, drawBox=False)

        elif if_existperson == 1:
            end_time = time.time()
            FPS = 1 / (end_time - start_time)
            scaler = 1
            b_g, b_g_1, b_g_2, b_g_3, b_g_4 = 0, 0, 0, 0, 0

            if self.frame_id == 1:
                sl, sr = position[11], position[12]
                bl, br = position[31], position[32]

            A1, B1 = [], []
            line = position
            A1.append(float(line[11][1]))
            A1.append(float(line[23][1]))
            A1.append(float(640))
            B1.append(float(line[11][2]))
            B1.append(float(line[23][2]))
            B1.append(float(line[23][2]))
            body_ground_angle = AngleCalculate(A1, B1)

            # if self.frame_id <= 20:
            #     self.wucha = abs(body_ground_angle - abs((math.atan(k0)) / math.pi * 180))
            self.wucha = 8
            self.obsY3.append(body_ground_angle - abs((math.atan(k0)) / math.pi * 180))

            A2, B2 = [], []
            A2.append(float(line[16][1]))
            A2.append(float(line[14][1]))
            A2.append(float(line[12][1]))
            B2.append(float(line[16][2]))
            B2.append(float(line[14][2]))
            B2.append(float(line[12][2]))
            shoulder_elbow_angle = AngleCalculate(A2, B2)

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

            A4, B4 = [], []
            A4.append(float(line[23][1]))
            A4.append(float(line[29][1]))
            A4.append(float(640))
            B4.append(float(line[23][2]))
            B4.append(float(line[29][2]))
            B4.append(float(line[29][2]))
            base_angle = AngleCalculate(A4, B4)
            self.obsY5.append(body_ground_angle - abs((math.atan(k1)) / math.pi * 180))

            # ROTATE_180 = 1
            # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

            # cv2.putText(frame2, '检测速率  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
            frame2 = cv2ImgAddText(frame2, f'{max(0, self.num)}', 25 * scaler, 60 * scaler, (0, 0, 255), 40,
                                   drawBox=False)
            frame2 = cv2ImgAddText(frame2, f'{self.time_str}', 25 * scaler, 105 * scaler, (255, 165, 0), 40,
                                   drawBox=False)
            # frame2 = cv2ImgAddText(frame2, f'{round(self.frame_id /FPS)}: {int(foot_ground_angle)}', 500 * scaler, 50 * scaler, (255, 255, 255), 40)
            # frame2 = cv2ImgAddText(frame2, f'{round(self.frame_id  / FPS)}: {int(body_ground_angle)}', 500 * scaler, 50 * scaler, (255, 255, 255), 40)
            # frame2 = cv2ImgAddText(frame2, f'{round(self.frame_id  / FPS)}: {int(shoulder_elbow_angle)}', 500 * scaler, 50 * scaler, (255, 255, 255), 40)

            if self.env_ifok == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '测试环境检测中', 25 * scaler, 15 * scaler, (0, 0, 255), 40,
                                       drawBox=False)
                frame2 = cv2ImgAddText(frame2, '请准备', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
            elif self.env_ifok == 1 and self.if_start == 2 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '检测成功！', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '请开始测试', 400 * scaler, 50 * scaler, (255, 165, 0), 40,
                                       drawBox=False)
                self.detectsuccess=True
                if len(self.marker_replaced_frames) >= 10:
                    if true_count / total_count >= 0.9:
                        marker_replace_enabled = True
            elif self.env_ifok == -1 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '脚出画框，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40,
                                       drawBox=False)
            elif self.env_ifok == -2 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '头部出画框，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40,
                                       drawBox=False)
            elif self.env_ifok == -3 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '斜度大于15°，请重新调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40,
                                       drawBox=False)
            elif self.env_ifok == -4 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '请调整测试位置', 25 * scaler, 15 * scaler, (0, 0, 255), 40,
                                       drawBox=False)

            if position[29][2] <= k0 * position[29][1] + b0 + 15 and position[23][2] <= k0 * position[23][
                1] + b0 + 15 and position[11][2] <= k0 * position[11][1] + b0 + 15:
                if position[3][2] > 0 and position[3][1] > 0:
                    if position[31][2] <= 480 and position[32][2] <= 480 and position[31][1] <= 640 and position[32][
                        1] <= 640:
                        # if body_ground_angle - abs((math.atan(k1)) / math.pi * 180) <= 10:

                        self.last.append(self.frame_id)
                        if self.frame_id >= 20 and len(self.last) >= 10:
                            self.env_ifok = 1

                        # else:
                            # self.env_ifok = -3
                            # self.last = []
                    else:
                        self.env_ifok = -1
                        self.last = []
                else:
                    self.env_ifok = -2
                    self.last = []
            else:
                self.env_ifok = -4
                self.last = []

            if body_ground_angle - base_angle <= 10 and self.env_ifok == 1:
                self.frametime.append(self.frame_id)
                self.if_start = 2
                if len(self.frametime) >= 10:
                    self.if_start = 1
            else:
                self.if_start = 0
            self.IF_START.append(self.if_start)

            if 1 in self.IF_START:
                self.testing = 1
                self.wait.append(self.frame_id)
                self.time_list.append([self.minute, self.second])
                self.doing.append(self.time_list[0])
                if (self.minute * 60 + self.second) - (self.doing[0][0] * 60 + self.doing[0][1]) == 1000000:
                    self.if_start = -1
                self.IF_START.append(self.if_start)

                if -1 in self.IF_START:
                    frame2 = cv2ImgAddText(frame2, '测试结束！', 25 * scaler, 50 * scaler, (0, 0, 255), 40,
                                           drawBox=False)

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

                # if position[14][1] - 15 > position[12][1]:
                #     self.hold_head = 1
                #     frame2 = cv2ImgAddText(frame2, '两手须在胸前折叠，不可抱头', 25 * scaler, 375 * scaler,
                #                            (255, 0, 255), 40)
                #     sign = 1
                if shoulder_elbow_angle >= 150 and self.num >= 1:
                    self.num_index = self.num_index + 1
                if self.num_index >= 3:
                    frame2 = cv2ImgAddText(frame2, '手臂打开借力', 25 * scaler, 350 * scaler, (255, 0, 255), 40)
                    # break
                    self.s_e = 1
                if (min(position[2][2], position[5][2]) < 0 or min(position[7][1], position[8][1]) < 0 or max(
                        position[7][1], position[8][1]) > 640 or
                        max(position[25][2], position[26][2]) > 480 or min(position[25][1], position[26][1]) < 0 or
                        max(position[25][1], position[26][1]) > 640):
                    frame2 = cv2ImgAddText(frame2, '测试者被替换', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
                    self.p = 1
                if self.num > 1:
                    self.HX.append(position[31][1])
                    self.HY.append(position[31][2])
                    if position[31][1] < self.HX[-2] - 640 / 10 and position[31][1] > self.HX[-2] + 640 / 10 and \
                            position[31][2] > self.HY[-2] + 480 / 10 and position[31][2] < self.HY[-2] - 480 / 10:
                        print(1)
                #       frame2 = cv2ImgAddText(frame2, '镜头中还有他人，请停止测试', 25 * scaler, 25 * scaler, (255, 0, 255),40)

                # img = frame2[
                #       min(sl[2], bl[2])-30:max(sr[2], br[2]+30),
                #       min(bl[1], br[1])-30:max(sl[1], sr[1])+30]
                # imgContour = img.copy()
                # imgGray = cv2.cvtColor(imgContour, cv2.COLOR_RGB2GRAY)  # 转灰度图
                # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
                # imgCanny = cv2.Canny(imgBlur, 100, 150)  # Canny算子边缘检测
                # if_tool = ShapeDetection(imgCanny)  # 形状检测
                # if if_tool == 1:
                #     frame2 = cv2ImgAddText(frame2, '有绳带物借力', 25 * scaler, 375 * scaler, (255, 0, 255),40)
                #     b_l = 1
                #     # break

                if len(self.obsY1) > 2 * window_size + 1:
                    idx = len(self.obsY1) - window_size - 1
                    local_window = self.obsY1[idx - window_size: idx + window_size + 1]
                    center_value = self.obsY1[idx]
                    is_peak = center_value == max(local_window) and (
                                max(local_window) - min(local_window)) >= delta_threshold
                    is_valley = center_value == min(local_window) and (
                                max(local_window) - min(local_window)) >= delta_threshold
                    if is_peak:
                        self.extremes.append(('peak', idx))
                        self.peak_angle = center_value
                    elif is_valley:
                        self.extremes.append(('valley', idx))
                        self.start_frame1.append(self.frame_id)
                        self.valley_angle = center_value

                if  self.valley_angle > 30:
                    frame2 = cv2ImgAddText(frame2, '肩背未着地', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                    self.cl1 = 1
                if 8 < self.peak_angle < 40 and self.num_all >= 1:
                    frame2 = cv2ImgAddText(frame2, '上起未达角度', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                    self.cl2 = 1

                if body_ground_angle2 <= 30:
                    b_g = 1
                if body_ground_angle2 >= 40:
                    b_g = -1

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

                if 1 in self.b and -1 in self.b and self.b[
                    -1] == -1 and self.s_e != 1 and self.cl1 != 1 and self.cl2 != 1:
                    self.num = self.num + 1
                    self.b = []
                    self.X1.append(self.num)
                if shoulder_elbow_angle2 <= 160 and shoulder_elbow_angle <= 160 and self.s_e == 1:
                    self.b = []
                    self.s_e = 0
                    self.num_index = 0
                if self.valley_angle < 30 and self.cl1 == 1:
                    self.b = []
                    self.cl1 = 0
                    self.index1 = 0
                if self.peak_angle > 40 and self.cl2 == 1:
                    self.b = []
                    self.cl2 = 0
                    self.index2 = 0

                print(self.extremes)
                if len(self.extremes) >= 2:
                    last_three = self.extremes[-2:]
                    if last_three[0][0] == 'valley' and last_three[1][0] == 'peak':
                        if self.peak_angle - self.valley_angle > 10:
                            self.num_all = self.num_all + 1
                            if self.start_frame1:
                                self.action_time.append(self.frame_id - self.start_frame1[0])
                        self.start_frame1 = []
                        self.extremes = []

                if self.num_all < self.num:
                    self.num_all = self.num
                    self.extremes = []

        self.obsX.append(self.frame_id / 25)

        if len(self.X) != len(self.obsY1):
            min_len = min(len(self.X), len(self.obsY1))
            self.X = self.X[:min_len]
            self.obsY1 = self.obsY1[:min_len]

        if num_old < self.num_all and self.num_all >= 0:
            if len(self.num_l) >= 9:
                self.body_ground_angle1, self.shoulder_elbow_angle1, self.head_holding1, self.self_texting1, self.num_l, self.shoulder_ground_angle1, self.cloth1, self.if_tool1 = [], [], [], [], [], [], [], []
                self.score_index = self.score_index + 1
            self.num_l.append(self.num_all)
            # if int(len(self.X1)) >= 2 and int(self.X1[-1]) > int(max(self.X1[:-1])):
            res = ('-', '肘关节是否打开', '是否有替考', '肩背是否着地', '上起是否符合要求', '是否有弹力绳')
            r = 0
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

            self.a, self.c, self.d, self.e, self.Z, self.ff, self.h = [], [], [], [], [], [], []

            self.list.append([f'第{self.num_all}个动作', self.body_ground_angle1[-1], res[r], '-'])
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

        current_time = time.time()
        score_path = rf"{datapath1}\score{self.score_index}.jpg"

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

        if self.score_overlay is not None:
            overlay_h, overlay_w = self.score_overlay.shape[:2]
            frame2[0:overlay_h, -overlay_w:] = self.score_overlay

        self.X.append(self.frame_id)
        self.frame_id += 1

        # 降低图表保存频率，改为异步保存
        current_time = time.time()
        if current_time - self.last_plot_save_time >= self.plot_save_interval:
            self.io_queue.put(('save_plots', {
                'body_ground': (self.X.copy(), self.obsY1.copy(), self.X1.copy()),
                'body_ground2': (self.X.copy(), self.obsY3.copy(), self.X1.copy()),
                'body_ground3': (self.X.copy(), self.obsY5.copy(), self.X1.copy()),
                'exercise_time': (self.X.copy(), self.action_time.copy(), self.X1.copy()),
            }))
            self.last_plot_save_time = current_time

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
        self.interval = 0.5  # 秒
        self.last_exec = time.monotonic() - self.interval

    def update(self, frame, frame_id):
        # 处理帧数据

        frame_data_generator = self.situp_start(frame, 0, 0)
        frame2, img1, num, num_all, IF_START, list_data = next(frame_data_generator)

        # 更新共享状态（从处理结果中获取）
        with self.lock:
            self.num = num
            self.num_all = num_all
            self.nums.append(num)
            self.timestamps.append(datetime.now().isoformat(timespec="milliseconds"))
        
        # 每 0.5 s 执行一次数据记录
        now = time.monotonic()
        if now - self.last_exec >= self.interval:
            with self.lock:
                record = {
                    "username": self.username,
                    "nums": self.nums.copy(),
                    "num": self.num,
                    "num_all": self.num_all,
                    "timestamps": self.timestamps.copy(),
                    "angles": self.obsY1.copy() if self.obsY1 else [],
                    # 检测成功条件：环境通过且处于准备状态，并且尚未进入正式开始信号
                    "detectsuccess": self.detectsuccess,
                    "finished": False,
                }
            _append_record(record)
            # 异步保存图片
            self.io_queue.put(('save_img', img1))
            self.last_exec = now
        

        result = {
            "uid": self.username,
            "score": self.num,
            "frame": frame_id,
            "num_all": self.num_all,
            "timestamp": int(round(time.time() * 1000)),
            "angle": self.obsY1[-1] if self.obsY1 else 0,
        }
    
        return result, frame2
    
    def stop(self):
        pass


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
