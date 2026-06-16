import operator
import argparse
import math
import cv2
import mediapipe as mp
import time
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import dataframe_image as dfi
import pandas as pd
import matplotlib.image as mpimg
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading
from queue import Queue, Empty
import json
from contextlib import ExitStack
import requests
import signal

DATA_FILE: Path = Path(__file__).resolve().parent / "pullup_runtime_data.json"
PULLUP_UPLOAD_LOG_FILE = Path(__file__).resolve().parent / "pullup_upload_log.txt"
STOP_FLAG = Path(__file__).with_suffix(".stop.flag")
_JSON_LOCK = threading.Lock()
stopping = {"flag": False}


def _request_stop(*_):
    stopping["flag"] = True


signal.signal(signal.SIGINT, _request_stop)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _request_stop)


def _append_pullup_upload_log(message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    try:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        if payload is not None:
            line += f" | {json.dumps(payload, ensure_ascii=False)}"
        with PULLUP_UPLOAD_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
    except Exception as e:
        print(f"Write pullup upload log failed: {e}")


def _append_record(record: Dict[str, Any]) -> None:
    """用新记录覆盖原有 JSON 内容（线程安全，仅保留最新一条）。"""
    with _JSON_LOCK:
        # 仍以数组形式保存，便于后续统一解析
        DATA_FILE.write_text(
            json.dumps([record], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

def _auth_remote_media_upload_url() -> str:
    explicit = (os.getenv("AUTH_REMOTE_MEDIA_UPLOAD") or "").strip()
    if explicit:
        return explicit
    base = (os.getenv("AUTH_REMOTE_BASE") or "").strip() or "http://121.196.163.155:8081"
    return base.rstrip("/") + "/media/upload/"


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _upload_pullup_media_to_auth_remote(
    video_path: str,
    *,
    userid: int,
    itemid: int = 1,
    score0: int,
    score1: int,
    testtime: str,
    shoulder_image: Optional[Path] = None,
    score_image: Optional[Path] = None,
    timeout_s: float = 10.0,
) -> None:
    video_file = Path(video_path)
    if not video_file.exists() or video_file.stat().st_size <= 0:
        print(f"Upload skipped, video missing/empty: {video_file}")
        return

    url = _auth_remote_media_upload_url()
    data: Dict[str, Any] = {
        "userid": str(userid),
        "itemid": str(itemid),
        "score0": str(score0),
        "score1": str(score1),
        "testtime": str(testtime),
    }

    upload_payload = {
        "url": url,
        "data": {k: str(v) for k, v in data.items()},
        "video": {
            "path": str(video_file.resolve()),
            "name": video_file.name,
            "exists": video_file.exists(),
            "size_bytes": int(video_file.stat().st_size),
        },
        "shoulder_image": str(shoulder_image.resolve()) if shoulder_image and shoulder_image.exists() else "",
        "score_image": str(score_image.resolve()) if score_image and score_image.exists() else "",
        "timeout_s": timeout_s,
    }
    _append_pullup_upload_log("[PullupUpload] 准备上传", upload_payload)

    http = requests.Session()
    http.trust_env = False

    try:
        with ExitStack() as stack:
            files: list[tuple[str, tuple[str, Any, str]]] = []
            f_video = stack.enter_context(video_file.open("rb"))
            files.append(("video", (video_file.name, f_video, "video/mp4")))

            if shoulder_image and shoulder_image.exists() and shoulder_image.stat().st_size > 0:
                f_img = stack.enter_context(shoulder_image.open("rb"))
                files.append(("images", ("shoudler.png", f_img, _image_mime(shoulder_image))))

            if score_image and score_image.exists() and score_image.stat().st_size > 0:
                f_img = stack.enter_context(score_image.open("rb"))
                files.append(("images", ("score.jpg", f_img, _image_mime(score_image))))

            resp = http.post(url, data=data, files=files, timeout=timeout_s)

        response_preview = (resp.text or "")[:500]
        if 200 <= resp.status_code < 300:
            print(f"Uploaded pullup media to auth remote: {url}")
            _append_pullup_upload_log(
                "[PullupUpload] 上传成功",
                {"status": resp.status_code, "url": url, "response": response_preview},
            )
        else:
            print(f"Upload failed ({resp.status_code}): {resp.text[:500]}")
            _append_pullup_upload_log(
                "[PullupUpload] 上传失败",
                {"status": resp.status_code, "url": url, "response": response_preview},
            )
    except Exception as e:
        print(f"Upload error: {e}")
        _append_pullup_upload_log(
            "[PullupUpload] 上传异常",
            {"url": url, "error": str(e), "request": upload_payload},
        )

mp_pose = mp.solutions.pose
filename = time.strftime('%Y-%m-%d %H %M %S', time.localtime())
datapath = fr'video/{filename}-pullup'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
global frame_id,yr_min, yl_min, yrw, ylw, kl, kr
yr_min, yl_min, yrw, ylw, kl, kr = 0, 0, 0, 0, 0, 0
K, XR, XL, YR, YL, YRW, YLW, YR_max, BB, XRW, XLW = [], [], [], [], [], [], [], [], [], [], []
foot_y = []
COUNT = []
# 导入模型
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=1,            # 选择人体姿态关键点检测模型，选1
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)   # 追踪阈值

fig, ax = plt.subplots()
fig.set_size_inches(6, 6)  # 设置图像大小
plt.ion()  # 打开交互模式

def get_maxima(values, order=10):
    if not values:
        return [], []

    values = np.asarray(values)
    max_indices = []
    max_values = []

    for i in range(0, len(values)):
        window = values[max(0, i - order):min(len(values), i + order + 1)]
        if values[i] == max(window) and values[i] != values[max(0, i - 1)] and values[i] != values[
            min(len(values) - 1, i + 1)]:
            max_indices.append(i)
            max_values.append(values[i])

    return max_indices, max_values

def get_minima(values, order=10):
    if not values:
        return [], []

    values = np.asarray(values)
    min_indices = []
    min_values = []

    for i in range(len(values)):
        window_start = max(0, i - order)
        window_end = min(len(values), i + order + 1)
        window = values[window_start:window_end]

        # Ensure the element is less than its immediate neighbors
        is_local_min = (values[i] == min(window)) and (i == 0 or values[i] != values[i - 1]) and (
                    i == len(values) - 1 or values[i] != values[i + 1])

        if is_local_min:
            min_indices.append(i)
            min_values.append(values[i])

    return min_indices, min_values
def process_frame(img, num, frame_id, hand_off, IF_START, WIDTH, HEIGHT, corrected_keypoints=None):
    global position, yr_min, yl_min, yrw, ylw, kl, kr, y01, x01, x02, foot_y, y1, y2
    position = []
    x01, x02, y01 = 0, 0, 450
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 结果
    results = pose.process(img_RGB)

    if results.pose_landmarks:
        if_existperson = 1
        # 可视化关键点和连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):

            # 获取关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z

            radius = 2
            position.append([i, cx, cy, cz])

            # 着色，美观
            if i == 0: # 鼻 - 使用矫正后的位置（如果有的话）
                if corrected_keypoints and 'nose' in corrected_keypoints:
                    # 使用矫正后的鼻子位置
                    corrected_nose = corrected_keypoints['nose']
                    nose_cx = int(corrected_nose['x'])
                    nose_cy = int(corrected_nose['y'])
                    img = cv2.circle(img, (nose_cx, nose_cy), radius, (0, 0, 255), -1)
                else:
                    # 使用原始位置
                    img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [11,12]: # 肩
                img = cv2.circle(img,(cx,cy), radius, (223,155,6), -1)
            elif i in [23,24]: # 髋
                img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
            elif i in [13,14]: # 胳膊肘
                img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
            elif i in [25,26]: # 膝
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [15,16,27,28]: # 手腕、脚腕
                img = cv2.circle(img,(cx,cy), radius, (223,155,60), -1)
            elif i in [17,19,21]: # 左手
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            elif i in [18,20,22]: # 右手
                img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
            elif i in [27,29,31]: # 左脚
                img = cv2.circle(img,(cx,cy), radius, (29,123,243), -1)
            elif i in [28,30,32]: # 右脚
                img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
            elif i in [9,10]: # 嘴
                img = cv2.circle(img,(cx,cy), radius, (205,235,255), -1)
            elif i in [1,2,3,4,5,6,7,8]: # 眼、脸颊
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            else: # 其它关键点
                img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)

            # 地面线 420——412
            if 5 >= frame_id >= 0:
                y1 = int(results.pose_landmarks.landmark[31].y * h)
                y2 = int(results.pose_landmarks.landmark[32].y * h)
                foot_y.append(max(y1, y2))
            if foot_y:
                y01 = foot_y[-1]+20
            img = cv2.line(img, (0, y01),(640, y01), (0, 255, 0), 3)
            yr_min = yl_min = y01

            # 待测者位置 320（+-80）
            x01 = 320 - 100
            x02 = 320 + 100
            if 1 not in IF_START:
                # img = cv2.line(img, (x01, y01), (x01, 0), (255, 0, 0), 3)
                # img = cv2.line(img, (x02, y01), (x02, 0), (255, 0, 0), 3)
                img = cv2.line(img, (int((x01+x02)/2), y01), (int((x01+x02)/2), 0), (255, 0, 0), 3)

            # 杠线
            y1 = int(results.pose_landmarks.landmark[20].y * h)
            y2 = int(results.pose_landmarks.landmark[19].y * h)
            if 1 not in IF_START:
                YLW.append(y2)
                YRW.append(y1)
            yrw = min(YRW)
            ylw = min(YLW)
            if 1 in IF_START and -1 not in IF_START:
                COUNT.append(frame_id)
                if frame_id - COUNT[0] >= 5 and hand_off == 0 and num >= 0:
                    img = cv2.line(img, (0, int(min(yrw, ylw))), (int(WIDTH*2), int(min(yrw, ylw))), (0, 0, 255), 3)
    else:
        if_existperson = 0
        y01 = 450
        img = cv2.line(img, (0, y01), (640, y01), (0, 255, 0), 3)

    return img, if_existperson, start_time, position, yr_min, yl_min, yrw, ylw, x01, x02, y01

def update_plot(ax, frame_id, obsY1):
    if len(frame_id) == 0 or len(obsY1) == 0:
        return

    min_len = min(len(frame_id), len(obsY1))
    frame_id = frame_id[:min_len]
    obsY1 = obsY1[:min_len]
    # num = num[:min_len]

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
    ax.set_ylabel("手肘角度")

    # 添加图例
    ax.legend(loc='upper left')

    # ax2 = ax.twiny()  # 创建一个共享纵坐标的第二横坐标
    # ax2.set_xlim(ax.get_xlim())  # 使得新横坐标与原始横坐标共享相同的范围
    # ax2.set_xticks(frame_id)  # 使用 frame_id 作为坐标
    # ax2.set_xticklabels(num)  # 使用 num 作为标签

    # plt.draw()
    # plt.pause(0.001)
def save_plot(filename, frame_id, obsY1):
    if len(frame_id) == 0 or len(obsY1) == 0:
        return

    min_len = min(len(frame_id), len(obsY1))
    frame_id = frame_id[:min_len]
    obsY1 = obsY1[:min_len]

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig_save = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig_save)
    ax_save = fig_save.add_subplot(111)

    ax_save.plot(frame_id, obsY1, 'r-', label='body_ground angle')
    ax_save.set_title("实时计数")
    ax_save.set_xlabel("检测时间")
    ax_save.set_ylabel("手肘角度")
    ax_save.legend(loc='upper left')

    fig_save.savefig(filename, bbox_inches='tight')
    canvas.draw()
def AngleCalculate(x, y):
    a = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2
    b = (x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2
    c = (x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2
    angle = abs(math.degrees(math.acos((a+c-b)/math.sqrt(4*a*c))))
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
def ShapeDetection(img):
    if_tool = 0
    img_height, img_width = img.shape[:2]  # 获取图像的高度和宽度
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 提取轮廓

    for obj in contours:
        x, y, w, h = cv2.boundingRect(obj)
        if h > 5 * w and h > img_height * 0.5:  # 判断是否是细长的线段（长宽比大于 5，且高度超过图像的一半）
            if y < img_height * 0.3 and y + h > img_height * 0.6:  # 顶部接近上边界，底部接近下边界
                if_tool = 1
                break  # 一旦找到符合条件的线段即可退出循环

    return if_tool

datapath1 = mkdir(datapath)
FILENAME = datapath1 + f'/{filename}-pullup.mp4'

class PULL():
    def __init__(self, userid: Optional[int] = None):
        self.userid = userid
        self.hand_off, self.num = 0, 0
        self.swr, self.swl, self.ewr, self.ewl, self.nose_x, self.body = 0, 0, 0, 0, 0, 0
        self.N, self.N2, self.X, self.B_L, self.L, self.HX, self.HY, self.A, self.Y, self.Z, self.P_A, self.F_H, self.START, self.frametime, self.H_D, self.O_G = [], [], [],  [], [], [], [], [], [], [], [], [], [], [], [], []
        self.a, self.b, self.c, self.d, self.e, self.ff, self.h, self.g = [], [], [], [], [], [], [], []
        self.num_l, self.if_count, self.elbow_angle, self.false_hold, self.hold_pot, self.on_ground, self.hand_distance, self.body_length1, self.pot_assisted, self.multi_user = [], [], [], [], [], [], [], [], [], []
        self.num_index, self.time_str, self.minute, self.num_index1, self.num_index2, self.second, self.num_all = 0, '', 0, 0, 0, 0, 0
        self.time_list, self.AA, self.hand_point1x, self.hand_point1y, self.hand_radius1, self.hand_point2x, self.hand_point2y, self.hand_radius2, self.leg_lengthr, self.leg_lengthl = [], [], [], [], [], [], [], [], [], []
        self.if_start, self.frame_id = 0, 0
        self.hands, self.head, self.last, self.IF_START, self.last_pot, self.last_box, self.foot_l_y, self.foot_r_y = [], [], [], [], [], [], [], []
        self.env_ifok = 0
        self.guo_gang, self.nose_count, self.action_time, self.start_frame = [], [], [], []
        self.x1, self.x2, self.y1, self.y2 = [], [], [], []
        self.valley_angle, self.peak_angle, self.last_count_frame, self.min_interval = 0, 0, -999, 5
        self.extremes = [('valley', 0)]
        self.body_ground_angle1 = []
        self.nums = []
        self.timestamps = []
        self.list = []
        self.last_score_update = 0
        self.score_update_interval = 0.75
        self.score_overlay = None
        self.score_image_raw = None
        self.frame_queue = Queue(maxsize=5)
        self.processed_queue = Queue(maxsize=3)
        self.io_queue = Queue()
        self.lock = threading.Lock()
        self.running = False
        self.stop_event = threading.Event()
        self.last_plot_save_time = 0
        self.plot_save_interval = 1.0
        # Track the most recent detected local minimum for the nose Y coordinate
        self.last_nose_min: Optional[float] = None
        # Track the most recent detected local maximum for the elbow angle
        self.last_elbow_max: Optional[float] = None
        self.detectsuccess=False
        self.readytohang=False

        # 关键点记录相关属性
        self.keypoints_buffer = []  # 存储最近5帧的关键点数据
        self.max_buffer_size = 5    # 缓冲区最大大小

        # 头肩比矫正相关属性
        self.head_shoulder_ratio = None  # 固定的头肩比参数
        self.correction_threshold = 3   # 矫正阈值
        self.is_detection_started = False  # 检测是否已开始

        super(PULL, self).__init__()

    def record_keypoints(self, position):
        """
        记录关键点坐标和置信度
        要求：鼻子、左肩、右肩、左胯、右胯的置信度都大于等于90%
        一直记录最新帧，保持缓冲区为最新5帧
        """
        # 定义需要记录的关键点索引
        required_keypoints = {
            'nose': 0,      # 鼻子
            'left_shoulder': 11,   # 左肩
            'right_shoulder': 12,  # 右肩
            'left_hip': 23,  # 左胯
            'right_hip': 24  # 右胯
        }

        try:
            # 检查所有必需关键点是否存在且置信度足够
            frame_data = {}
            all_confident = True

            for name, idx in required_keypoints.items():
                if idx >= len(position):
                    all_confident = False
                    break

                # position格式: [idx, x, y, z, visibility]
                if len(position[idx]) < 5:
                    all_confident = False
                    break

                x, y, z, visibility = position[idx][1], position[idx][2], position[idx][3], position[idx][4]
                confidence = visibility * 100  # 转换为百分比

                if confidence < 90:
                    all_confident = False
                    break

                frame_data[name] = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'confidence': confidence
                }

            # 如果所有关键点都满足置信度要求，则记录这一帧
            if all_confident:
                self.keypoints_buffer.append({
                    'frame_id': self.frame_id,
                    'timestamp': time.time(),
                    'keypoints': frame_data
                })

                # 保持缓冲区大小为5，移除最旧的帧
                if len(self.keypoints_buffer) > self.max_buffer_size:
                    self.keypoints_buffer.pop(0)

        except Exception as e:
            # 静默处理错误，不输出到控制台
            pass

    def get_keypoints_data(self):
        """
        获取记录的关键点数据
        返回最新的5帧数据，每帧包含5个关键点的坐标和置信度

        返回格式:
        [
            {
                'frame_id': 帧ID,
                'timestamp': 时间戳,
                'keypoints': {
                    'nose': {'x': x, 'y': y, 'z': z, 'confidence': conf},
                    'left_shoulder': {...},
                    'right_shoulder': {...},
                    'left_hip': {...},
                    'right_hip': {...}
                }
            },
            ... (最多5帧)
        ]
        """
        return self.keypoints_buffer.copy()

    def calculate_head_shoulder_ratio(self, keypoints):
        """
        计算头肩比：(鼻子纵坐标与左右肩中点纵坐标插值) / (左右肩横坐标差值)

        参数:
        keypoints: 关键点数据字典

        返回:
        头肩比值，如果计算失败返回None
        """
        try:
            nose = keypoints.get('nose')
            left_shoulder = keypoints.get('left_shoulder')
            right_shoulder = keypoints.get('right_shoulder')

            if not all([nose, left_shoulder, right_shoulder]):
                return None

            # 计算左右肩中点纵坐标
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2

            # 计算鼻子与肩中点纵坐标差值
            nose_shoulder_diff = abs(nose['y'] - shoulder_center_y)

            # 计算左右肩横坐标差值
            shoulder_width = abs(right_shoulder['x'] - left_shoulder['x'])

            if shoulder_width == 0:
                return None

            # 计算头肩比
            ratio = nose_shoulder_diff / shoulder_width
            return ratio

        except Exception:
            return None

    def calculate_fixed_head_shoulder_ratio(self):
        """
        计算固定头肩比参数：对最新5帧数据计算头肩比的平均值
        """
        keypoints_data = self.get_keypoints_data()

        if len(keypoints_data) < 5:
            return False

        ratios = []
        for frame_data in keypoints_data:
            ratio = self.calculate_head_shoulder_ratio(frame_data['keypoints'])
            if ratio is not None:
                ratios.append(ratio)

        if len(ratios) >= 3:  # 至少需要3帧有效数据
            self.head_shoulder_ratio = sum(ratios) / len(ratios)
            return True

        return False

    def correct_nose_position(self, keypoints):
        """
        矫正鼻子纵坐标
        使用固定头肩比参数和当前左右肩横坐标差值计算理想鼻子纵坐标

        参数:
        keypoints: 当前帧的关键点数据（会被修改）
        """
        if self.head_shoulder_ratio is None:
            return False

        try:
            nose = keypoints.get('nose')
            left_shoulder = keypoints.get('left_shoulder')
            right_shoulder = keypoints.get('right_shoulder')

            if not all([nose, left_shoulder, right_shoulder]):
                return False

            # 计算当前左右肩横坐标差值
            shoulder_width = abs(right_shoulder['x'] - left_shoulder['x'])

            if shoulder_width == 0:
                return False

            # 计算左右肩中点纵坐标
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2

            # 使用固定头肩比计算理想鼻子纵坐标
            ideal_nose_y = shoulder_center_y + (self.head_shoulder_ratio * shoulder_width)

            # 检查差值是否超过阈值
            current_nose_y = nose['y']
            diff = abs(current_nose_y - ideal_nose_y)

            if diff > self.correction_threshold:
                # 更新鼻子纵坐标为理想值
                keypoints['nose']['y'] = ideal_nose_y
                return True

            return False

        except Exception:
            return False

    def pullup_start(self, frame, WIDTH, HEIGHT):
        WIDTH, HEIGHT = 640, 480
        s1 = "\""
        s2 = "\'"
        num_old = self.num_all

        # 首先进行基本的关键点处理，获取位置信息
        frame1, if_existperson, start_time,position, yr_min, yl_min, yrw, ylw, x01, x02, y01 = process_frame(frame, self.num, self.frame_id, self.hand_off, self.IF_START, WIDTH, HEIGHT)

        # 检查是否需要进行鼻子矫正，并准备矫正后的关键点数据
        corrected_keypoints = None
        if if_existperson == 1 and 1 in self.IF_START and -1 not in self.IF_START:
            if self.is_detection_started and self.head_shoulder_ratio is not None:
                # 创建当前帧的关键点字典用于矫正
                current_keypoints = {
                    'nose': {'x': position[0][1], 'y': position[0][2], 'z': position[0][3], 'confidence': position[0][4] * 100},
                    'left_shoulder': {'x': position[11][1], 'y': position[11][2], 'z': position[11][3], 'confidence': position[11][4] * 100},
                    'right_shoulder': {'x': position[12][1], 'y': position[12][2], 'z': position[12][3], 'confidence': position[12][4] * 100},
                    'left_hip': {'x': position[23][1], 'y': position[23][2], 'z': position[23][3], 'confidence': position[23][4] * 100},
                    'right_hip': {'x': position[24][1], 'y': position[24][2], 'z': position[24][3], 'confidence': position[24][4] * 100}
                }

                # 矫正鼻子位置
                if self.correct_nose_position(current_keypoints):
                    # 更新position数组中的鼻子坐标
                    position[0][2] = current_keypoints['nose']['y']
                    # 重新处理帧，使用矫正后的关键点
                    corrected_keypoints = current_keypoints
                    frame1, if_existperson, start_time,position, yr_min, yl_min, yrw, ylw, x01, x02, y01 = process_frame(frame, self.num, self.frame_id, self.hand_off, self.IF_START, WIDTH, HEIGHT, corrected_keypoints)
        img1 = np.zeros_like(frame1)
        if self.num >= 0:
            if not hasattr(self, 'start_time'):
                self.start_time = time.time()
            run_time = time.time()
            total_seconds = int(run_time - self.start_time)
            self.minute, self.second = divmod(total_seconds, 60)
            self.time_str = f'{self.minute}{s2}{self.second}{s1}'

        self.io_queue.put(('save_position', self.frame_id, position.copy()))

        frame2 = cv2.flip(frame1, 1)
        scaler = 1
        # 显示数值时间
        frame2 = cv2ImgAddText(frame2, f'{max(0, self.num)}', 25 * scaler, 60 * scaler, (0, 0, 255), 40, drawBox=False)
        frame2 = cv2ImgAddText(frame2, f'{self.time_str}', 25 * scaler, 105 * scaler, (255, 165, 0), 40)

        if if_existperson == 0 and 1 not in self.IF_START:
            frame2 = cv2ImgAddText(frame2, '未检测到人!', 25 * scaler, 100 * scaler, (255, 0, 255), 40, drawBox=False)
        elif if_existperson == 0 and 1 in self.IF_START:
            frame2 = cv2ImgAddText(frame2, '摄像头像素太低', 25 * scaler, 50 * scaler, (0, 0, 255), 40, drawBox=False)
            frame2 = cv2ImgAddText(frame2, '停止测试', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
            frame2 = cv2ImgAddText(frame2, '重新更换设备', 350 * scaler, 100 * scaler, (255, 165, 0), 40, drawBox=False)
        elif if_existperson == 1:
            # 记录该帧处理完毕的时间
            end_time = time.time()
            # 计算每秒处理图像帧数FPS
            FPS = 1 / (end_time - start_time)
            e_a, f_h, h_p, o_g, h_d, b_l, p_a, m_u, z, k1, k2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            A1, B1 = [], []
            A1.append(float(position[16][1]))
            A1.append(float(position[14][1]))
            A1.append(float(position[12][1]))
            B1.append(float(position[16][2]))
            B1.append(float(position[14][2]))
            B1.append(float(position[12][2]))
            self.elbow_angle = AngleCalculate(A1, B1)
            self.Y.append(self.elbow_angle)
            
            # 检测波峰（窗口10，看大趋势，过滤3-4帧的小浅峰）
            if len(self.Y) >= 20:
                window = self.Y[-20:]
                max_val = max(window)
                max_pos = window.index(max_val)

                # 要求左右都有样本，避免窗口边缘的小抖动
                if 2 <= max_pos <= 17:
                    left_min = min(window[:max_pos]) if max_pos > 0 else max_val
                    right_min = min(window[max_pos+1:]) if max_pos < len(window)-1 else max_val
                    rise = max_val - left_min
                    fall = max_val - right_min
                    main_amp = max(window) - min(window)

                    rise_th = 5   # 上升幅度阈值
                    fall_th = 5   # 下降幅度阈值

                    # 需明显上升且下降，并且上升占主振幅一定比例，过滤小扰动；
                    # 若新峰值比当前记录低太多（>25），视为抖动不更新。
                    if rise >= rise_th and fall >= fall_th and rise >= 0.4 * main_amp:
                        if self.last_elbow_max is not None and max_val < self.last_elbow_max - 25:
                            pass
                        else:
                            self.last_elbow_max = max_val

            if self.frame_id == 1:
                self.nose_x = position[0][1]
                self.swr = position[12][1]
                self.swl = position[11][1]
                self.ewr = position[8][1]
                self.ewl = position[7][1]
                self.body = abs(position[24][1] - position[23][1])
            body_length = position[23][2] - position[11][2]
            self.B_L.append(body_length)
            self.L.append(body_length / max(self.B_L))

            self.START.append(self.if_start)
            if (self.env_ifok == 0 or self.env_ifok == 2) and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '请上杠悬挂', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '请上杠悬挂', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
                self.readytohang=True
            elif self.env_ifok == 1 and self.if_start == 2 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '检测成功！', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '请开始测试', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
                self.detectsuccess=True
            elif self.env_ifok == -1 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '脚出画框，请停止动作', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -2 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '头部出画框，请停止动作', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
            elif self.env_ifok == -3 and self.if_start == 0 and 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '请调整位置', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)

            self.hands.append(position[19][2])
            self.hands.append(position[20][2])
            self.head.append(abs(position[11][2] - position[1][2]))
            useless, hands_top = min(enumerate(self.hands), key=operator.itemgetter(1))
            useless, head_long = max(enumerate(self.head), key=operator.itemgetter(1))
            if (x01+x02)/2 <= position[19][1] and position[20][1] <= (x01+x02)/2:
                self.env_ifok = 2
                if position[19][2] < position[1][2] and position[20][2] < position[4][2]:
                    # if position[31][2] < y01 + 3 and position[32][2] < y01 +3:
                        # if hands_top >= head_long-20:
                                self.last.append(self.frame_id)
                                if self.frame_id >= 30:
                                    if len(self.last) >= 10:
                                        self.env_ifok = 1
                                if -1 not in self.IF_START and 1 not in self.IF_START:
                                    self.record_keypoints(position)
                                if self.if_start == 0 and 1 not in self.IF_START:
                                    if abs(self.ewl - self.ewr) >= abs(position[16][1] - position[15][1]):
                                        frame2 = cv2ImgAddText(frame2, '握距太窄，请调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                                    if abs(position[16][1] - position[15][1]) >= 3*abs(self.swl - self.swr):
                                        frame2 = cv2ImgAddText(frame2, '握距太宽，请调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                        # else:
                                # self.env_ifok = -2
                                # self.last = []
                    # else:
                        # self.env_ifok = -1
                        # self.last = []
            else:
                self.env_ifok = -3
                self.last = []

            if position[19][2] < position[11][2] and position[20][2] < position[12][2] and self.env_ifok == 1:
                self.frametime.append(self.frame_id)
                self.if_start = 2
                if len(self.frametime) >= 10 and position[31][2] < yr_min and position[32][2] < yl_min:
                    self.if_start = 1
                    # 检测开始：计算固定头肩比参数
                    if not self.is_detection_started:
                        if self.calculate_fixed_head_shoulder_ratio():
                            self.is_detection_started = True
            elif self.hand_off == 1:
                self.if_start = -1
            else:
                self.if_start = 0
            self.IF_START.append(self.if_start)

            if -1 in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '测试结束！', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)

            if 1 in self.IF_START and -1 not in self.IF_START:
                # if abs(position[15][1] - position[16][1]) <= 1.5*abs(self.ewl - self.ewr):
                    # frame2 = cv2ImgAddText(frame2, '异常显示', 25 * scaler, 150 * scaler, (255, 0, 255), 40)
                    # h_d = 1
                    # self.H_D.append(h_d)
                # elif abs(position[24][1] - position[23][1]) <= 0.5*self.body:
                    # frame2 = cv2ImgAddText(frame2, '异常显示', 25 * scaler, 150 * scaler, (255, 0, 255), 40)
                    # h_d = 1
                    # self.H_D.append(h_d)
                # elif abs(position[12][1] - position[11][1]) <= 0.5*abs(self.swl - self.swr):
                    # frame2 = cv2ImgAddText(frame2, '异常显示', 25 * scaler, 150 * scaler, (255, 0, 255), 40)
                    # h_d = 1
                    # self.H_D.append(h_d)
                # else:
                    self.hand_point1x.append((position[17][1] + position[21][1]) / 2)  # 左手
                    self.hand_point1y.append((position[19][2] + position[15][2]) / 2)  # 左手
                    self.hand_point2x.append((position[18][1] + position[22][1]) / 2)
                    self.hand_point2y.append((position[20][2] + position[16][2]) / 2)
                    y1 = min(self.hand_point1y)
                    handpoint1 = [self.hand_point1x[self.hand_point1y.index(y1)], y1]
                    y2 = min(self.hand_point2y)
                    handpoint2 = [self.hand_point2x[self.hand_point2y.index(y2)], y2]
                    self.hand_radius1.append(1.5 * max(position[19][2] - handpoint1[1], position[17][1] - handpoint1[0]))
                    self.hand_radius2.append(1.5 * max(position[20][2] - handpoint2[1], position[18][1] - handpoint2[0]))
                    handradius1 = max(self.hand_radius1)
                    handradius2 = max(self.hand_radius2)
                    if position[22][1] < position[18][1] and position[21][1] > position[17][1] and self.num > 1:
                        # frame2 = cv2ImgAddText(frame2, '须两手正握杠，请停止测试', 25 * scaler, 325 * scaler, (255, 0, 255), 40)
                        # f_h = 1
                        self.F_H.append(f_h)
                        # break
                    for i in [15, 17, 19, 21]:
                        if (position[i][1] - handpoint1[0])**2 + (position[i][2] - handpoint1[1])**2 >= handradius1**2:
                            self.num_index1 = self.num_index1 + 1
                            if self.num_index1 >= 30:
                                # frame2 = cv2ImgAddText(frame2, '须两手正握杠，请停止测试', 25 * scaler, 325 * scaler, (255, 0, 255), 40)
                                # f_h = 1
                                self.F_H.append(f_h)
                                # break
                    for i in [22, 20, 16, 18]:
                        if (position[i][1] - handpoint1[0])**2 + (position[i][2] - handpoint1[1])**2 >= handradius2**2:
                            self.num_index2 = self.num_index2 + 1
                            if self.num_index1 >= 30:
                                # frame2 = cv2ImgAddText(frame2, '须两手正握杠，请停止测试', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                                # f_h = 1
                                self.F_H.append(f_h)
                                # break
                    # if min(self.B_L)/max(self.B_L) > 0.6 :
                    if (position[31][2] >= yl_min+3 or position[32][2] >= yr_min+3):
                            frame2 = cv2ImgAddText(frame2, '脚沾地', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
                            o_g = 1
                            self.O_G.append(o_g)
                            # break
                    # if (position[20][2] > yrw + 25 or position[19][2] > ylw + 25) and self.num > 1:
                    #     self.A.append(self.frame_id)
                    #     if len(self.A) >= 5:
                    #         # frame2 = cv2ImgAddText(frame2, '双手离杠', 25 * scaler, 200 * scaler, (255, 0, 255), 40)
                    #         h_p = 1
                    if position[20][2] > yrw + 50 and position[19][2] > ylw + 50:
                        self.A.append(self.frame_id)
                        if len(self.A) >= 5:
                            self.hand_off = 1

                    # if 1 in self.IF_START:
                    #     img = frame2[min(position[12][2]-10, position[11][2]-10):max(position[23][2]-20, position[24][2]-20), min(position[11][1]-10, position[12][1]-10):max(position[12][1]+10, position[11][1]+10)]
                    #     imgContour = img.copy()
                    #     imgGray = cv2.cvtColor(imgContour, cv2.COLOR_RGB2GRAY)  # 转灰度图
                    #     imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
                    #     imgCanny = cv2.Canny(imgBlur, 100, 1500)  # Canny算子边缘检测
                    #     if_tool = ShapeDetection(imgCanny, imgContour)  # 形状检测
                    #     if if_tool == 1:
                    #         frame2 = cv2ImgAddText(frame2, '借助弹力绳，此次动作不计', 25 * scaler, 350 * scaler,(255, 0, 255), 40)
                    #         b_l = 1
                    #         self.Z.append(b_l)
                    #         # break
                    #     self.leg_lengthr.append(abs(position[26][2] - position[24][2]))
                    #     self.leg_lengthl.append(abs(position[25][2] - position[23][2]))
                    #     leglengthr = max(self.leg_lengthr)
                    #     leglengthl = max(self.leg_lengthl)
                    #     if abs(position[25][2] - position[23][2]) <= 0.3*leglengthl or abs(position[26][2] - position[24][2]) <= 0.3*leglengthr:
                    #         # frame2 = cv2ImgAddText(frame2, '摆动幅度过大，请停止动作', 25 * scaler, 100 * scaler, (255, 0, 255), 40)
                    #         # b_l = 1
                    #         self.Z.append(b_l)
                            # break

                    if position[31][1] > 2.5*self.swl-1.5*self.nose_x or position[32][1] < 2.5*self.swr-1.5*self.nose_x:
                        self.last_pot.append(self.frame_id)
                        if len(self.last_pot) >= 10:
                            frame2 = cv2ImgAddText(frame2, '借助侧柱支撑', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
                            p_a = 1
                            self.P_A.append(p_a)
                            # break
                    else:
                        self.last_pot = []

                    self.foot_r_y.append(position[31][2])
                    self.foot_l_y.append(position[32][2])
                    if max(self.foot_r_y) - min(self.foot_r_y) <= 3 and max(self.foot_l_y) - min(self.foot_l_y) <= 3:
                        self.last_box.append(self.frame_id)
                        if len(self.last_pot) >= 10:
                            frame2 = cv2ImgAddText(frame2, '检测到支撑物', 25 * scaler, 350 * scaler, (255, 0, 255), 40)
                            m_u = 1
                            # break
                    else:
                        self.last_box = []

                    self.b.append(f_h)
                    self.c.append(h_p)
                    self.d.append(o_g)
                    self.e.append(h_d)
                    self.ff.append(b_l)
                    self.h.append(p_a)
                    self.g.append(m_u)

                    self.nose_count.append(position[0][2])
                    
                    # 检测波谷（窗口10，看大趋势，过滤3-4帧的小浅谷）
                    if len(self.nose_count) >= 20:  
                        window = self.nose_count[-20:]
                        min_val = min(window)
                        min_pos = window.index(min_val)

                        # 要求左右都有样本，避免窗口边缘小抖动
                        if 2 <= min_pos <= 17:
                            left_max = max(window[:min_pos]) if min_pos > 0 else min_val
                            right_max = max(window[min_pos+1:]) if min_pos < len(window)-1 else min_val
                            drop = left_max - min_val
                            rise = right_max - min_val
                            main_amp = max(window) - min(window)

                            drop_th = 5   # 下跌幅度阈值
                            rise_th = 5   # 回升幅度阈值

                            # 需明显跌落且回升，并且跌幅占主振幅一定比例，过滤小扰动；
                            # 若新谷值比当前记录高出过多（>25），视为抖动不更新。
                            if drop >= drop_th and rise >= rise_th and drop >= 0.4 * main_amp:
                                if self.last_nose_min is not None and min_val > self.last_nose_min + 25:
                                    pass
                                else:
                                    self.last_nose_min = min_val
                    
                    current_elbow_max = self.last_elbow_max if self.last_elbow_max is not None else 180
                    current_nose_y = self.last_nose_min if self.last_nose_min is not None else 0
                    
                    if self.elbow_angle >= 120:
                        self.N.append(0)
                    if position[0][2] <= min(ylw, yrw) + 20:
                        self.N.append(1)
                        self.guo_gang = []
                    if (current_nose_y > yrw+20) or (current_nose_y > ylw+20):
                        self.guo_gang.append(self.frame_id)
                        if len(self.guo_gang) >= 5:
                            frame2 = cv2ImgAddText(frame2, '下巴未过杠', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                            # k1 = 1

                    # if current_elbow_max < 120:
                        # frame2 = cv2ImgAddText(frame2, '手臂未放直', 25 * scaler, 400 * scaler, (255, 0, 255),40)
                        # k2 = 1

                    if 0 in self.N and self.N[-1] == 1 and self.if_start == 1:
                        if 1 not in self.P_A:
                            # if 1 not in self.Z:
                                # if 1 not in self.O_G:
                                #   if 1 not in self.F_H:
                                #       if 1 not in self.H_D:
                                            self.num = self.num + 1
                                            self.N = []
                                            self.a.append(self.num)
                        else:
                            self.N = []
                            self.H_D = []
                            self.P_A = []
                            self.Z = []
                            self.F_H = []

                    # 趋势判定：连续下行记 1，连续上行记 0，避免单点噪声
                    if len(self.nose_count) >= 4:
                        recent = self.nose_count[-4:]
                        total_delta = recent[-1] - recent[0]
                        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

                        # 明显下降趋势（总降幅≥5，且连续为负斜率）
                        if total_delta <= -5 and all(d < 0 for d in deltas):
                                self.N2.append(1)
                                if not self.start_frame:
                                    self.start_frame.append(self.frame_id)
                        # 明显上升趋势（总升幅≥5，且连续为正斜率）
                        elif total_delta >= 5 and all(d > 0 for d in deltas):
                                self.N2.append(0)
                    if 1 in self.N2 and self.N2[-1] == 0 and self.if_start == 1:
                        self.num_all = self.num_all + 1
                        if self.start_frame:
                            self.action_time.append(self.frame_id - self.start_frame[0])
                        else:
                            self.action_time.append(0)
                        self.N2 = []
                        self.start_frame = []


                    if len(self.X) != len(self.Y):
                        min_len = min(len(self.X), len(self.Y))
                        self.X = self.X[:min_len]
                        self.Y = self.Y[:min_len]

                    if num_old < self.num_all and self.num_all >= 1:
                        res = ('-', '反握', '脚沾地', '握距过长或过短', '下巴未过杠', '手臂未放直',  '借力')
                        r = 0
                        self.num_l.append(self.num_all)
                        if self.a:
                            self.if_count.append('.')
                        else:
                            self.if_count.append('×')
                        if 1 in self.b:
                            self.false_hold.append('×')
                        else:
                            self.false_hold.append('.')
                        if 1 in self.c:
                            self.hold_pot.append('×')
                        else:
                            self.hold_pot.append('.')
                        if 1 in self.d:
                            self.on_ground.append('×')
                        else:
                            self.on_ground.append('.')
                        if 1 in self.e:
                            self.hand_distance.append('×')
                        else:
                            self.hand_distance.append('.')
                        if 1 in self.ff:
                            self.body_length1.append('×')
                        else:
                            self.body_length1.append('.')
                        if 1 in self.h:
                            self.pot_assisted.append('×')
                        else:
                            self.pot_assisted.append('.')
                        if 1 in self.g:
                            self.multi_user.append('×')
                        else:
                            self.multi_user.append('.')

                        if not self.a:
                            if 1 in self.b:
                                r = 1
                            if 1 in self.d:
                                r = 2
                            if 1 in self.e:
                                r = 3
                            if k1 == 1:
                                r = 4
                            if k2 == 1:
                                r = 5
                            if 1 in self.h:
                                r = 6
                        else:
                            r = 0
                        self.a, self.b, self.c, self.d, self.e, self.ff, self.h, self.g = [], [], [], [], [], [], [], []

                        self.list.append([f'第{self.num_all}个动作', self.if_count[-1], res[r], '-'])

        data = {
            '计数': self.num_l,
            '是否计数': self.if_count,
            '是否反握': self.false_hold,
            # '是否握杠': self.hold_pot,
            '是否脚沾地': self.on_ground,
            '握距是否正常': self.hand_distance,
            '是否有摆动': self.body_length1,
            '是否借力': self.pot_assisted,
            '是否有多人': self.multi_user,
        }
        df = pd.DataFrame(data)

        current_time = time.time()
        score_path = rf"{datapath1}\score.jpg"

        if current_time - self.last_score_update >= self.score_update_interval or self.score_overlay is None:
            dfi.export(df.T, score_path, table_conversion="matplotlib")
            img1 = mpimg.imread(score_path)
            plt.xticks([])
            plt.yticks([])

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
        self.frame_id+= 1

        current_time = time.time()
        if current_time - self.last_plot_save_time >= self.plot_save_interval:
            self.io_queue.put(('save_plots', {
                'shoulder': (self.X.copy(), self.Y.copy()),
                'exercise_time': (self.X.copy(), self.action_time.copy()),
            }))
            self.last_plot_save_time = current_time

        yield frame, frame2, img1, self.num , self.num_all, self.IF_START, self.list

    def _frame_capture_thread(self, cap, WIDTH, HEIGHT):
        try:
            while self.running and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
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
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    frame, WIDTH, HEIGHT = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    frame_data_generator = self.pullup_start(frame, WIDTH, HEIGHT)
                    frame, frame2, img1, num, num_all, IF_START, list_data = next(frame_data_generator)
                    if self.processed_queue.full():
                        try:
                            self.processed_queue.get_nowait()
                        except Empty:
                            pass
                    self.processed_queue.put((frame, frame2, img1, num, num_all, IF_START, list_data), timeout=0.1)
                except Exception as e:
                    print(f"帧处理错误: {e}")
                    continue
        except Exception as e:
            print(f"帧处理线程错误: {e}")
        finally:
            print("帧处理线程结束")

    def _io_thread(self):
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
                        shoulder_data = plots_data.get('shoulder', ([], []))
                        exercise_data = plots_data.get('exercise_time', ([], []))
                        save_plot(fr'{datapath1}\shoulder.png',
                                  shoulder_data[0], shoulder_data[1])
                        save_plot(fr'{datapath1}\exercise_time.png',
                                  exercise_data[0], exercise_data[1])
                    elif task[0] == 'save_img':
                        _, img1 = task
                        cv2.imwrite("img1.jpeg", img1)
                except Exception as e:
                    print(f"I/O操作错误: {e}")
        except Exception as e:
            print(f"I/O线程错误: {e}")
        finally:
            print("I/O线程结束")

    def start_video_processing(self, if_open, WIDTH, HEIGHT):
        if if_open == 1:
            try:
                STOP_FLAG.unlink(missing_ok=True)
            except Exception:
                pass
            stopping["flag"] = False
            cv2.namedWindow('Pullup Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pullup Detection', WIDTH, HEIGHT)

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                print("无法打开摄像头")
                return

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            capture_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            record_fps = capture_fps * (2 / 3)

            # 处理后的视频
            out = cv2.VideoWriter(FILENAME, fourcc, record_fps, (WIDTH, HEIGHT))

            # 原视频（未处理的原始帧）
            original_filename = FILENAME.replace('-pullup.mp4', '-pullup-original.mp4')
            out_original = cv2.VideoWriter(original_filename, fourcc, capture_fps, (WIDTH, HEIGHT))

            self.running = True
            self.stop_event.clear()

            capture_thread = threading.Thread(target=self._frame_capture_thread, args=(cap, WIDTH, HEIGHT), daemon=True)
            process_thread = threading.Thread(target=self._frame_process_thread, daemon=True)
            io_thread = threading.Thread(target=self._io_thread, daemon=True)

            capture_thread.start()
            process_thread.start()
            io_thread.start()

            INTERVAL = 0.5
            last_exec = time.monotonic() - INTERVAL
            last_frame_time = time.time()

            try:
                while self.running:
                    if stopping["flag"] or STOP_FLAG.exists():
                        end_record = {
                            "nums": self.nums.copy(),
                            "num": self.num,
                            "num_all": self.num_all,
                            "timestamps": self.timestamps.copy(),
                            "angles": self.Y.copy() if self.Y else [],
                            "detectsuccess": False,
                            "readytohang": False,
                            "finished": True,
                        }
                        _append_record(end_record)
                        _append_pullup_upload_log(
                            "[PullupUpload] 收到停止信号，准备结束测试",
                            {
                                "userid": self.userid,
                                "num": self.num,
                                "num_all": self.num_all,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        print("测试结束")
                        break

                    try:
                        frame, frame2, img1, num, num_all, IF_START, list_data = self.processed_queue.get(timeout=0.1)
                    except Empty:
                        if time.time() - last_frame_time > 0.1:
                            pass
                        continue

                    last_frame_time = time.time()

                    with self.lock:
                        self.num = num
                        self.num_all = num_all
                        self.list = list_data
                        self.nums.append(num)
                        self.timestamps.append(datetime.now().isoformat(timespec="milliseconds"))

                    now = time.monotonic()
                    if now - last_exec >= INTERVAL:
                        record = {
                            "nums": self.nums.copy(),
                            "num": self.num,
                            "num_all": self.num_all,
                            "timestamps": self.timestamps.copy(),
                            "angles": self.Y.copy() if self.Y else [],
                            # 检测成功条件：环境通过且处于准备状态，并且尚未进入正式开始信号
                            "detectsuccess": self.detectsuccess,
                            "readytohang": self.readytohang,
                            "finished": False,
                        }
                        _append_record(record)
                        self.io_queue.put(('save_img', img1))
                        last_exec = now

                    cv2.imshow('Pullup Detection', frame2)
                    out.write(frame2)  # 保存处理后的视频
                    out_original.write(frame)  # 保存原始视频

                    if -1 in IF_START:
                        end_record  = {
                            "nums": self.nums.copy(),
                            "num": self.num,
                            "num_all": self.num_all,
                            "timestamps": self.timestamps.copy(),
                            "angles": self.Y.copy() if self.Y else [],
                            "detectsuccess": False,
                            "readytohang": False,
                            "finished": True,
                        }
                        _append_record(end_record)

                        print("测试结束")
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        _append_pullup_upload_log(
                            "[PullupUpload] 检测到按下 q，准备结束测试",
                            {
                                "userid": self.userid,
                                "num": self.num,
                                "num_all": self.num_all,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        break
            finally:
                self.running = False
                self.stop_event.set()

                capture_thread.join(timeout=1.0)
                process_thread.join(timeout=1.0)
                io_thread.join(timeout=2.0)

                cap.release()
                out.release()  # 释放处理后的视频
                out_original.release()  # 释放原始视频
                cv2.destroyAllWindows()
                
                datapath_dir = Path(datapath1)
                try:
                    save_plot(str(datapath_dir / "shoulder.png"), self.X, self.Y)
                except Exception:
                    pass

                try:
                    score_path = datapath_dir / "score.jpg"
                    if not score_path.exists() or score_path.stat().st_size <= 0:
                        df = pd.DataFrame(
                            {
                                "count": getattr(self, "num_l", []),
                                "if_count": getattr(self, "if_count", []),
                                "false_hold": getattr(self, "false_hold", []),
                                "on_ground": getattr(self, "on_ground", []),
                                "hand_distance": getattr(self, "hand_distance", []),
                                "body_length": getattr(self, "body_length1", []),
                                "pot_assisted": getattr(self, "pot_assisted", []),
                                "multi_user": getattr(self, "multi_user", []),
                            }
                        )
                        dfi.export(df.T, str(score_path), table_conversion="matplotlib")
                except Exception:
                    pass

                if self.userid is not None:
                    with self.lock:
                        score0 = int(getattr(self, "num", 0))
                        score1 = int(getattr(self, "num_all", 0))
                        testtime = (
                            self.timestamps[-1]
                            if self.timestamps
                            else datetime.now().isoformat(timespec="seconds")
                        )
                    # 异步上传，不阻塞主线程
                    upload_thread = threading.Thread(
                        target=_upload_pullup_media_to_auth_remote,
                        kwargs={
                            "video_path": FILENAME,
                            "userid": self.userid,
                            "itemid": 1,
                            "score0": score0,
                            "score1": score1,
                            "testtime": testtime,
                            "shoulder_image": datapath_dir / "shoulder.png",
                            "score_image": datapath_dir / "score.jpg",
                        },
                        daemon=False
                    )
                    upload_thread.start()
                    print("后台正在上传视频...")
                    _append_pullup_upload_log(
                        "[PullupUpload] 后台开始上传",
                        {
                            "userid": self.userid,
                            "itemid": 1,
                            "score0": score0,
                            "score1": score1,
                            "testtime": testtime,
                            "video_path": FILENAME,
                            "shoulder_image": str((datapath_dir / "shoulder.png").resolve()),
                            "score_image": str((datapath_dir / "score.jpg").resolve()),
                            "log_file": str(PULLUP_UPLOAD_LOG_FILE.resolve()),
                        },
                    )
                else:
                    _append_pullup_upload_log(
                        "[PullupUpload] 跳过上传，userid 缺失",
                        {"log_file": str(PULLUP_UPLOAD_LOG_FILE.resolve())},
                    )

                try:
                    STOP_FLAG.unlink(missing_ok=True)
                except Exception:
                    pass
                stopping["flag"] = False

        elif if_open == -1:
            save_plot(fr'{datapath1}\shoulder.png', self.X, self.Y)
            save_plot(fr'{datapath1}\exercise_time.png', self.X, self.action_time)

        elif if_open == 0:
            pass

# 完成个数：self.num
# 完成总个数：self.num_all
# 处理帧数据：frame2
# 完成判据表格：img1
# 数据图：shoulder.png，(self.X(frame_id), self.obsY(判断标准、手肘角度))
# 检测视频保存地址：FILENAME（fr'video/{filename}-situp' filename为当前时刻），保存在本地需上传到数据库
# 结束信号：self.IF_START(数组中存在-1就结束)

 # 张三——204.4.25 10.09.09——仰卧起坐、立定跳远....、pdf——jpg、txt、MP4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--userid",
        default=os.environ.get("PULLUP_USERID"),
        help="userid (optional; if provided, upload media to auth remote server)",
    )
    args = parser.parse_args()

    pullup_detector = PULL(userid=_safe_int(args.userid))
    pullup_detector.start_video_processing(if_open=1, WIDTH=640, HEIGHT=480)
