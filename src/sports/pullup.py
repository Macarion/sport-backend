import math
import cv2
import time
import numpy as np
from enum import IntEnum
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from core.config import FONT_DIR, PULLUP_STATIC_IMAGE_MODE
from core.task import SportTask

class PosePoint:

    def __init__(self, idx, x, y, z, visibility=1.0):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class Inferencer:

    def __init__(self):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=PULLUP_STATIC_IMAGE_MODE,
            model_complexity=0,  # 降低复杂度以提升速度（0最快，1中等，2最慢）
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
        )

    # 循环处理函数
    def process(self, task: SportTask):
        img = task.get_frame()
        drawing = task.get_config().get("drawing", True)

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_RGB)
        points = []

        h = img.shape[0]
        w = img.shape[1]

        if results.pose_landmarks:
            if drawing:
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            for i in range(33):
                # 归一化坐标转画面像素
                cx = int(results.pose_landmarks.landmark[i].x * w)
                cy = int(results.pose_landmarks.landmark[i].y * h)
                cz = results.pose_landmarks.landmark[i].z  # 关键点深度值
                visibility = results.pose_landmarks.landmark[i].visibility  # 关键点可见性
                points.append(PosePoint(i, cx, cy, cz, visibility))

        return {
            "has_person": results.pose_landmarks is not None,
            "points": points,
        }


# 评测状态
class State(IntEnum):
    WAITING = 0     # 等待开始
    READY = 1       # 准备就绪
    FLAT = 2        # 平躺
    RISE = 3        # 起身
    WAIT_FLAT = 4   # 作弊后等待平躺
    FINISHED = 5    # 评测结束


class Evaluator:
    # 评测算法

    def __init__(self):
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
        self.running = False
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

        self.foot_y = []
        self.COUNT = []
        self.yr_min, self.yl_min, self.yrw, self.ylw, self.kl, self.kr = 0, 0, 0, 0, 0, 0
        self.K, self.XR, self.XL, self.YR, self.YL, self.YRW, self.YLW, self.YR_max, self.BB, self.XRW, self.XLW = [], [], [], [], [], [], [], [], [], [], []
    

    def img_put_text(
        self,
        img,
        text,
        left,
        top,
        textColor=(0, 255, 0),
        textSize=40,
        borderColor=(0, 0, 0),
        borderWidth=2,
        drawBox=False,
    ):
        img_copy = img.copy()
        pil_img = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            fontStyle = ImageFont.truetype(
                str(FONT_DIR / "msyh.ttf"), textSize, encoding="utf-8", index=1
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
            boxPadding = 4
            boxThickness = 2
            boxColor = (255, 0, 0)
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

        np.copyto(img, cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR))
        return img

    # ==========================
    # 循环处理函数
    # ==========================
    def process(self, task: SportTask):

        img = task.get_frame()
        data = task.get_data()
        config = task.get_config()
        frame_id = task.get_frame_id()

        points: list[PosePoint] = data["points"]
        has_person = data["has_person"]
        drawing = config.get("drawing", True)

        self.frame_id = frame_id
        HEIGHT = img.shape[0]
        WIDTH = img.shape[1]

        frame2 = img
        scaler = 1
        x01, x02, y01 = 0, 0, 450

        if not has_person:
            img = cv2.line(img, (0, y01), (640, y01), (0, 255, 0), 3)

            if 1 not in self.IF_START:
                frame2 = cv2ImgAddText(frame2, '未检测到人!', 25 * scaler, 100 * scaler, (255, 0, 255), 40, drawBox=False)
            else:
                frame2 = cv2ImgAddText(frame2, '摄像头像素太低', 25 * scaler, 50 * scaler, (0, 0, 255), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '停止测试', 400 * scaler, 50 * scaler, (255, 165, 0), 40, drawBox=False)
                frame2 = cv2ImgAddText(frame2, '重新更换设备', 350 * scaler, 100 * scaler, (255, 165, 0), 40, drawBox=False)

            return None

        # 检查是否需要进行鼻子矫正，并准备矫正后的关键点数据
        corrected_keypoints = None
        # 会导致错误判定为下巴没有过杠，注释了
        # if 1 in self.IF_START and -1 not in self.IF_START:
        #     if self.is_detection_started and self.head_shoulder_ratio is not None:
        #         # 创建当前帧的关键点字典用于矫正
        #         current_keypoints = {
        #             'nose': {'x': points[0].x, 'y': points[0].y, 'z': points[0].z},
        #             'left_shoulder': {'x': points[11].x, 'y': points[11].y, 'z': points[11].z},
        #             'right_shoulder': {'x': points[12].x, 'y': points[12].y, 'z': points[12].z},
        #             'left_hip': {'x': points[23].x, 'y': points[23].y, 'z': points[23].z},
        #             'right_hip': {'x': points[24].x, 'y': points[24].y, 'z': points[24].z}
        #         }

        #         # 矫正鼻子位置
        #         if self.correct_nose_points(current_keypoints):
        #             # 更新points数组中的鼻子坐标
        #             points[0].y = current_keypoints['nose']['y']
        #             # 重新处理帧，使用矫正后的关键点
        #             corrected_keypoints = current_keypoints

        if drawing:
            for i in range(33):

                # 获取关键点的三维坐标
                cx = points[i].x
                cy = points[i].y

                radius = 2

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
            y1 = points[31].y
            y2 = points[32].y
            self.foot_y.append(max(y1, y2))
        if self.foot_y:
            y01 = self.foot_y[-1]+20
        img = cv2.line(img, (0, y01),(640, y01), (0, 255, 0), 3)
        self.yr_min = self.yl_min = y01

        # 待测者位置 320（+-80）
        x01 = 320 - 100
        x02 = 320 + 100
        if 1 not in self.IF_START:
            # img = cv2.line(img, (x01, y01), (x01, 0), (255, 0, 0), 3)
            # img = cv2.line(img, (x02, y01), (x02, 0), (255, 0, 0), 3)
            img = cv2.line(img, (int((x01+x02)/2), y01), (int((x01+x02)/2), 0), (255, 0, 0), 3)

        # 杠线
        y1 = points[20].y
        y2 = points[19].y
        if 1 not in self.IF_START:
            self.YLW.append(y2)
            self.YRW.append(y1)
        self.yrw = min(self.YRW)
        self.ylw = min(self.YLW)
        if 1 in self.IF_START and -1 not in self.IF_START:
            self.COUNT.append(frame_id)
            if frame_id - self.COUNT[0] >= 5 and self.hand_off == 0 and self.num >= 0:
                img = cv2.line(img, (0, int(min(self.yrw, self.ylw))), (int(WIDTH*2), int(min(self.yrw, self.ylw))), (0, 0, 255), 3)

        # 旧版的pullup_start
        # s1 = "\""
        # s2 = "\'"
        num_old = self.num_all

        # img1 = np.zeros_like(img)
        # if self.num >= 0:
        #     if not hasattr(self, 'start_time'):
        #         self.start_time = time.time()
        #     run_time = time.time()
        #     total_seconds = int(run_time - self.start_time)
        #     self.minute, self.second = divmod(total_seconds, 60)
        #     self.time_str = f'{self.minute}{s2}{self.second}{s1}'

        # 显示数值时间
        # frame2 = cv2ImgAddText(frame2, f'{max(0, self.num)}', 25 * scaler, 60 * scaler, (0, 0, 255), 40, drawBox=False)
        # frame2 = cv2ImgAddText(frame2, f'{self.time_str}', 25 * scaler, 105 * scaler, (255, 165, 0), 40)

        e_a, f_h, h_p, o_g, h_d, b_l, p_a, m_u, z, k1, k2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        A1, B1 = [], []
        A1.append(float(points[16].x))
        A1.append(float(points[14].x))
        A1.append(float(points[12].x))
        B1.append(float(points[16].y))
        B1.append(float(points[14].y))
        B1.append(float(points[12].y))
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
            self.nose_x = points[0].x
            self.swr = points[12].x
            self.swl = points[11].x
            self.ewr = points[8].x
            self.ewl = points[7].x
            self.body = abs(points[24].x - points[23].x)
        body_length = points[23].y - points[11].y
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

        self.hands.append(points[19].y)
        self.hands.append(points[20].y)
        self.head.append(abs(points[11].y - points[1].y))
        if (x01+x02)/2 <= points[19].x and points[20].x <= (x01+x02)/2:
            self.env_ifok = 2
            if points[19].y < points[1].y and points[20].y < points[4].y:
                # if points[31].y < y01 + 3 and points[32].y < y01 +3:
                    # if hands_top >= head_long-20:
                            self.last.append(self.frame_id)
                            if self.frame_id >= 30:
                                if len(self.last) >= 10:
                                    self.env_ifok = 1
                            if -1 not in self.IF_START and 1 not in self.IF_START:
                                self.record_keypoints(points)
                            if self.if_start == 0 and 1 not in self.IF_START:
                                if abs(self.ewl - self.ewr) >= abs(points[16].x - points[15].x):
                                    frame2 = cv2ImgAddText(frame2, '握距太窄，请调整', 25 * scaler, 15 * scaler, (0, 0, 255), 40, drawBox=False)
                                if abs(points[16].x - points[15].x) >= 3*abs(self.swl - self.swr):
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

        if points[19].y < points[11].y and points[20].y < points[12].y and self.env_ifok == 1:
            self.frametime.append(self.frame_id)
            self.if_start = 2
            if len(self.frametime) >= 10 and points[31].y < self.yr_min and points[32].y < self.yl_min:
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
            # if abs(points[15].x - points[16].x) <= 1.5*abs(self.ewl - self.ewr):
                # frame2 = cv2ImgAddText(frame2, '异常显示', 25 * scaler, 150 * scaler, (255, 0, 255), 40)
                # h_d = 1
                # self.H_D.append(h_d)
            # elif abs(points[24].x - points[23].x) <= 0.5*self.body:
                # frame2 = cv2ImgAddText(frame2, '异常显示', 25 * scaler, 150 * scaler, (255, 0, 255), 40)
                # h_d = 1
                # self.H_D.append(h_d)
            # elif abs(points[12].x - points[11].x) <= 0.5*abs(self.swl - self.swr):
                # frame2 = cv2ImgAddText(frame2, '异常显示', 25 * scaler, 150 * scaler, (255, 0, 255), 40)
                # h_d = 1
                # self.H_D.append(h_d)
            # else:
                self.hand_point1x.append((points[17].x + points[21].x) / 2)  # 左手
                self.hand_point1y.append((points[19].y + points[15].y) / 2)  # 左手
                self.hand_point2x.append((points[18].x + points[22].x) / 2)
                self.hand_point2y.append((points[20].y + points[16].y) / 2)
                y1 = min(self.hand_point1y)
                handpoint1 = [self.hand_point1x[self.hand_point1y.index(y1)], y1]
                y2 = min(self.hand_point2y)
                handpoint2 = [self.hand_point2x[self.hand_point2y.index(y2)], y2]
                self.hand_radius1.append(1.5 * max(points[19].y - handpoint1[1], points[17].x - handpoint1[0]))
                self.hand_radius2.append(1.5 * max(points[20].y - handpoint2[1], points[18].x - handpoint2[0]))
                handradius1 = max(self.hand_radius1)
                handradius2 = max(self.hand_radius2)
                if points[22].x < points[18].x and points[21].x > points[17].x and self.num > 1:
                    # frame2 = cv2ImgAddText(frame2, '须两手正握杠，请停止测试', 25 * scaler, 325 * scaler, (255, 0, 255), 40)
                    # f_h = 1
                    self.F_H.append(f_h)
                    # break
                for i in [15, 17, 19, 21]:
                    if (points[i].x - handpoint1[0])**2 + (points[i].y - handpoint1[1])**2 >= handradius1**2:
                        self.num_index1 = self.num_index1 + 1
                        if self.num_index1 >= 30:
                            # frame2 = cv2ImgAddText(frame2, '须两手正握杠，请停止测试', 25 * scaler, 325 * scaler, (255, 0, 255), 40)
                            # f_h = 1
                            self.F_H.append(f_h)
                            # break
                for i in [22, 20, 16, 18]:
                    if (points[i].x - handpoint1[0])**2 + (points[i].y - handpoint1[1])**2 >= handradius2**2:
                        self.num_index2 = self.num_index2 + 1
                        if self.num_index1 >= 30:
                            # frame2 = cv2ImgAddText(frame2, '须两手正握杠，请停止测试', 25 * scaler, 400 * scaler, (255, 0, 255), 40)
                            # f_h = 1
                            self.F_H.append(f_h)
                            # break
                # if min(self.B_L)/max(self.B_L) > 0.6 :
                if (points[31].y >= self.yl_min+3 or points[32].y >= self.yr_min+3):
                        frame2 = cv2ImgAddText(frame2, '脚沾地', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
                        o_g = 1
                        self.O_G.append(o_g)
                        # break
                # if (points[20].y > yrw + 25 or points[19].y > ylw + 25) and self.num > 1:
                #     self.A.append(self.frame_id)
                #     if len(self.A) >= 5:
                #         # frame2 = cv2ImgAddText(frame2, '双手离杠', 25 * scaler, 200 * scaler, (255, 0, 255), 40)
                #         h_p = 1
                if points[20].y > self.yrw + 50 and points[19].y > self.ylw + 50:
                    self.A.append(self.frame_id)
                    if len(self.A) >= 5:
                        self.hand_off = 1

                # if 1 in self.IF_START:
                #     img = frame2[min(points[12].y-10, points[11].y-10):max(points[23].y-20, points[24].y-20), min(points[11].x-10, points[12].x-10):max(points[12].x+10, points[11].x+10)]
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
                #     self.leg_lengthr.append(abs(points[26].y - points[24].y))
                #     self.leg_lengthl.append(abs(points[25].y - points[23].y))
                #     leglengthr = max(self.leg_lengthr)
                #     leglengthl = max(self.leg_lengthl)
                #     if abs(points[25].y - points[23].y) <= 0.3*leglengthl or abs(points[26].y - points[24].y) <= 0.3*leglengthr:
                #         # frame2 = cv2ImgAddText(frame2, '摆动幅度过大，请停止动作', 25 * scaler, 100 * scaler, (255, 0, 255), 40)
                #         # b_l = 1
                #         self.Z.append(b_l)
                        # break

                if points[31].x > 2.5*self.swl-1.5*self.nose_x or points[32].x < 2.5*self.swr-1.5*self.nose_x:
                    self.last_pot.append(self.frame_id)
                    if len(self.last_pot) >= 10:
                        frame2 = cv2ImgAddText(frame2, '借助侧柱支撑', 25 * scaler, 375 * scaler, (255, 0, 255), 40)
                        p_a = 1
                        self.P_A.append(p_a)
                        # break
                else:
                    self.last_pot = []

                self.foot_r_y.append(points[31].y)
                self.foot_l_y.append(points[32].y)
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

                self.nose_count.append(points[0].y)
                
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
                if points[0].y <= min(self.ylw, self.yrw) + 20:
                    self.N.append(1)
                    self.guo_gang = []
                if (current_nose_y > self.yrw+20) or (current_nose_y > self.ylw+20):
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

        # data = {
        #     '计数': self.num,
        #     '是否计数': self.if_count,
        #     '是否反握': self.false_hold,
        #     # '是否握杠': self.hold_pot,
        #     '是否脚沾地': self.on_ground,
        #     '握距是否正常': self.hand_distance,
        #     '是否有摆动': self.body_length1,
        #     '是否借力': self.pot_assisted,
        #     '是否有多人': self.multi_user,
        # }

        np.copyto(img, frame2)

        return {
            "frame_id": frame_id,
            "score": self.num,
            "num_all": self.num_all,
            "angle": HEIGHT - points[0].y,

            # "ready": self.state >= State.READY,
            # "angle": body_ground_angle2,
            # "hints": hints,
            # "is_new_rise": is_new_rise,
            # "is_arm_force_cheat": self.is_arm_force_cheat,
            # "is_out_of_bounds": self.is_out_of_bounds,
            # "is_flat_insufficient": self.is_flat_insufficient,
            # "is_rise_insufficient": self.is_rise_insufficient,
            # "is_knee_move": self.is_knee_moved,
            "timestamp": int(round(time.time() * 1000)),
        }


    def correct_nose_points(self, keypoints):
        """
        矫正鼻子纵坐标
        使用固定头肩比参数和当前左右肩横坐标差值计算理想鼻子纵坐标

        参数:
        keypoints: 当前帧的关键点数据（会被修改）
        """
        if self.head_shoulder_ratio is None:
            return False

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

    def record_keypoints(self, points):
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

        # 检查所有必需关键点是否存在且置信度足够
        frame_data = {}
        all_confident = True

        for name, idx in required_keypoints.items():
            if idx >= len(points):
                all_confident = False
                break

            # position格式: [idx, x, y, z, visibility]
            x, y, z, visibility = points[idx].x, points[idx].y, points[idx].z, points[idx].visibility
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

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=10, borderColor=(0, 0, 0), borderWidth=2, drawBox=True, boxColor=(255, 0, 0), boxPadding=4, boxThickness=2):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)  # 创建一个可以在给定图像上绘图的对象
    try:
        fontStyle = ImageFont.truetype(str(FONT_DIR / "msyh.ttf"), textSize, encoding="utf-8", index=1)
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

def AngleCalculate(x, y):
    a = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2
    b = (x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2
    c = (x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2
    angle = abs(math.degrees(math.acos((a+c-b)/math.sqrt(4*a*c))))
    return angle


from .base import BaseSport  # 包装类依赖（置于尾部，不打断原文件结构）


class SportPullup(BaseSport):
    """引体向上：MediaPipe Pose 推理 + 状态机评测的链式封装。"""

    def __init__(self):
        self._inferencer = None
        self._evaluator = None

    def start(self):
        """加载模型与评测状态机（用户进程内调用）。"""
        self._inferencer = Inferencer()
        self._evaluator = Evaluator()

    def process(self, frame, config, frame_id, uid=0):
        """处理一帧（就地绘制），返回评测数据 dict；本帧无有效数据返回 None。"""
        if self._inferencer is None:
            raise RuntimeError('SportPullup 未调用 start()')
        task = SportTask(uid=uid, frame_id=frame_id, sport_type='pullup',
                         config=config if config is not None else {})
        task.set_frame(frame)
        task.set_data(self._inferencer.process(task))
        return self._evaluator.process(task)

    def stop(self):
        self._inferencer = None
        self._evaluator = None