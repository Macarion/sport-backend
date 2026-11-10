import math
import cv2
import time
import numpy as np
from enum import IntEnum
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from core.config import (
    FONT_DIR, POSE_YOLO_BOX_CONF,
    POSE_YOLO_IMG_SZ, POSE_YOLO_KPT_CONF, POSE_YOLO_MODEL_PATH,
    SITUP_MARKER_ENABLED, SITUP_MARKER_LOWER, SITUP_MARKER_UPPER,
)
from core.task import SportTask


class PosePoint:

    def __init__(self, idx, x, y, z):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z


class _BasePoseInferencer:

    def __init__(self):
        

        self.lower_green = np.array(SITUP_MARKER_LOWER)  # 设定绿色的阈值下限
        self.upper_green = np.array(SITUP_MARKER_UPPER)  # 设定绿色的阈值上限
        self.marker_enabled = SITUP_MARKER_ENABLED
        self._mark_kernel = np.ones((5, 5), np.uint8)

    # 标志物检测函数
    def _detect_marker_circle_bgr(self, frame: np.ndarray) -> Optional[tuple[int, int, int]]:
        """
        将 backend/api/mark_cv.py 的标志物识别逻辑封装为函数：
        - HSV 绿色阈值分割
        - 形态学开运算去噪
        - Canny 边缘
        - HoughCircles 圆检测
        返回 (x, y, r) 像素坐标与半径；未检测到返回 None。
        """
        if frame is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)  # 设定掩膜取值范围
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._mark_kernel)  # 开运算去噪
        edges = cv2.Canny(opening, 80, 160)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            1,
            100,
            param1=100,
            param2=20,
            minRadius=6,
            maxRadius=100,
        )
        if circles is None:
            return None

        c = circles[0][0]
        return (int(c[0]), int(c[1]), int(c[2]))

    def _apply_marker(self, img: np.ndarray, points: list[PosePoint]) -> Optional[tuple[int, int, int]]:
        """可选的圆形标志物检测；命中时替换左髋(11)坐标，返回 (x, y, r) 或 None。"""
        if not self.marker_enabled:
            return None

        marker_circle = self._detect_marker_circle_bgr(img)
        if marker_circle is not None:
            mx, my, _mr = marker_circle
            points[11].x = mx
            points[11].y = my
        return marker_circle

    def _draw_marker(self, img: np.ndarray, marker_circle: tuple[int, int, int]):
        mx, my, mr = marker_circle
        # 绘制标志物外圈+圆心点+文字
        cv2.circle(img, (mx, my), mr, (0, 0, 255), 3)
        cv2.circle(img, (mx, my), 3, (255, 255, 0), -1)

    def _draw_keypoints(self, img: np.ndarray, points: list[PosePoint]):
        """遍历 COCO 17 个人体关键点，按部位绘制不同颜色圆点。"""
        radius = 2  # 关键点绘制圆点半径
        for p in points:
            # 按关键点分组填充不同颜色圆点
            if p.idx == 0:
                cv2.circle(img, (p.x, p.y), radius, (0, 0, 255), -1)      # 鼻
            elif p.idx in [1, 2, 3, 4]:
                cv2.circle(img, (p.x, p.y), radius, (94, 218, 121), -1)   # 眼、耳
            elif p.idx in [5, 6]:
                cv2.circle(img, (p.x, p.y), radius, (223, 155, 6), -1)    # 肩
            elif p.idx in [7, 8]:
                cv2.circle(img, (p.x, p.y), radius, (140, 47, 240), -1)   # 肘
            elif p.idx in [9, 10]:
                cv2.circle(img, (p.x, p.y), radius, (223, 155, 60), -1)   # 腕
            elif p.idx in [11, 12]:
                cv2.circle(img, (p.x, p.y), radius, (1, 240, 255), -1)    # 髋
            elif p.idx in [13, 14]:
                cv2.circle(img, (p.x, p.y), radius, (0, 0, 255), -1)      # 膝
            elif p.idx == 15:
                cv2.circle(img, (p.x, p.y), radius, (29, 123, 243), -1)   # 左踝
            elif p.idx == 16:
                cv2.circle(img, (p.x, p.y), radius, (193, 182, 255), -1)  # 右踝
            else:
                cv2.circle(img, (p.x, p.y), radius, (0, 255, 0), -1)


class Inferencer(_BasePoseInferencer):
    """Ultralytics YOLO Pose 姿态推理（多人，COCO 17 关键点）。

    输出 COCO 17 关键点，供下游 Evaluator 直接复用；
    选人策略：按"靠画面中心 + 框面积大"加权得分最高者（不依赖平躺姿态）。
    """

    # COCO 17 关键点骨架连接（绘制所有人用）
    COCO_SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),            # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # 双臂
        (5, 11), (6, 12), (11, 12),                # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16),    # 双腿
    ]

    # 选人打分权重：靠画面中心 + 框面积大
    CENTER_WEIGHT = 0.9
    AREA_WEIGHT = 0.1

    def __init__(self):
        super().__init__()
        from ultralytics import YOLO
        

        self.model = YOLO(POSE_YOLO_MODEL_PATH)  # 首次运行自动下载权重
        self.img_sz = POSE_YOLO_IMG_SZ
        self.box_conf = POSE_YOLO_BOX_CONF
        self.kpt_conf = POSE_YOLO_KPT_CONF

    def _select_person(self, r, frame_shape) -> int:
        """按"靠画面中心 + 框面积大"加权选人，返回检测索引。

        两者各自归一化到 [0,1] 后加权求和，得分最高者当选。
        不依赖平躺姿态，测试开始前也可稳定选人。
        """
        h, w = frame_shape[:2]
        cx0, cy0 = w / 2.0, h * 2 / 3.0
        max_dist = math.hypot(cx0, cy0)

        boxes = r.boxes.xyxy.cpu().numpy()
        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
        max_area = areas.max() if areas.size else 0.0

        best_idx, best_score = 0, float("-inf")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            bcx, bcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # 距中心越近分越高
            center_score = 1.0 - math.hypot(bcx - cx0, bcy - cy0) / max_dist
            # 面积越大分越高（相对当前最大框归一化）
            area_score = float(areas[i] / max_area) if max_area > 0 else 0.0
            score = self.CENTER_WEIGHT * center_score + self.AREA_WEIGHT * area_score
            if score > best_score:
                best_score, best_idx = score, i
        return best_idx

    def _draw_persons(self, img: np.ndarray, r, selected_idx: int):
        """选中者：框+骨架+关键点；其他人：仅画框。"""
        if r.boxes is None or len(r.boxes) == 0:
            return

        kpt_conf_all = r.keypoints.conf if r.keypoints is not None else None
        n = len(r.boxes)
        for i in range(n):
            box = r.boxes.xyxy[i].cpu().numpy()
            box_conf = float(r.boxes.conf[i])
            x1, y1, x2, y2 = (int(v) for v in box)
            selected = (i == selected_idx)

            if selected:
                color = (255, 0, 0)  # 蓝框标记被选中者
                label = f"target {box_conf:.2f}"
            else:
                color = (0, 255, 0)
                label = f"person {box_conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label,
                (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA,
            )

            if not selected or r.keypoints is None:
                continue

            kpt_xy = r.keypoints.xy[i].cpu().numpy()
            if kpt_conf_all is not None:
                kpt_conf = kpt_conf_all[i].cpu().numpy()
            else:
                kpt_conf = np.ones(len(kpt_xy), dtype=np.float32)

            def pt(j, kpt=kpt_xy):
                return (int(kpt[j][0]), int(kpt[j][1]))

            # 选中者绘制 COCO 骨架连线（彩色关键点由调用方在 points 上绘制）
            for a, b in self.COCO_SKELETON:
                if kpt_conf[a] > self.kpt_conf and kpt_conf[b] > self.kpt_conf:
                    cv2.line(img, pt(a), pt(b), (0, 255, 255), 2, cv2.LINE_AA)

    # 循环处理函数
    def process(self, task: SportTask):
        img = task.get_frame()
        config = task.get_config()

        drawing = config.get("drawing", True)

        points: list[PosePoint] = []
        person_boxes: list[list[float]] = []
        selected_idx = -1

        # YOLO 推理直接使用 BGR 帧
        results = self.model.predict(
            img, imgsz=self.img_sz, conf=self.box_conf, verbose=False,
        )
        r = results[0]

        has_person = r.keypoints is not None and len(r.keypoints) > 0

        if r.boxes is not None and len(r.boxes) > 0:
            for bx in r.boxes.xyxy.cpu().numpy():
                person_boxes.append([float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])])

        if has_person:
            # 选人：画面中心距离 + 框面积加权
            sel = self._select_person(r, img.shape)
            selected_idx = sel
            kpt_xy = r.keypoints.xy[sel].cpu().numpy()  # (17, 2) 像素坐标

            # 直接构造 COCO 17 关键点
            points = [
                PosePoint(i, int(xy[0]), int(xy[1]), 0.0)
                for i, xy in enumerate(kpt_xy)
            ]

            # ========== 圆形标志物检测 ==========
            marker_circle = self._apply_marker(img, points)

            if drawing:
                if marker_circle is not None:
                    self._draw_marker(img, marker_circle)

                # 选中者画框+骨架，其他人只画框
                self._draw_persons(img, r, sel)
                # 被评测者的彩色关键点
                self._draw_keypoints(img, points)

        return {
            "has_person": has_person,
            "points": points,
            "person_boxes": person_boxes,
            "selected_idx": selected_idx,
        }


# 评测状态
class State(IntEnum):
    WAIT = 0        # 等待坐下
    FACE = 1        # 正在进行人脸识别
    READY = 2       # 识别通过，等待躺下
    FLAT = 3        # 平躺（躺下合格进入测试）
    RISE = 4        # 起身
    WAIT_FLAT = 5   # 作弊后等待平躺
    FINISHED = 6    # 评测结束


class Evaluator:
    # 评测算法

    def __init__(self, face_client=None):
        self.face_client = face_client  # 人脸识别客户端（None=门禁关闭，直接放行）
        self.state = State.WAIT
        self.num = 0
        

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
        self.is_knee_moved = False  # 膝盖移动作弊标记
        self.is_assist_cheat = False  # 头后方有人助力作弊标记
        self.knee_move_alpha = 0.05  # 膝盖移动权重，权重越大，平均值越接近当前帧
        self.knee_move_threshold = 0.15  # 膝盖移动阈值(1代表视频宽度)
        self.weighted_knee_point = None # 膝盖移动加权平均点
        self.assist_count = 0  # 头后方有人驻留的连续帧计数
        self.assist_hold_frames = 75  # 驻留多少帧判定助力作弊
        self.assist_behind_dx = 0.25  # 头后方横轴检测范围(相对画面宽,沿脚→头方向)
        self.assist_band_dy = 0.15  # 纵坐标带容差(相对画面宽,与被测者框中心y比)
        self.flat_threshold = 20
        self.rise_threshold = 50
        self.last_flat_angle = 180
        self.last_rise_angle = 0

        # 流程状态标记
        self.env_ifok = 0  # 环境检测状态：0检测中/1合格/-1~-4各类环境异常
        self.env_ok_count = 0  # 环境检测合格帧计数器
        self.ready_count = 0  # 准备动作帧计数器
        self.arm_force_count = 0
        self.ready_time = None
    
    def is_flat(self, angle):
        return angle <= self.flat_threshold
    
    def is_rise(self, angle):
        return angle >= self.rise_threshold
    
    def is_mid(self, angle):
        return self.flat_threshold < angle < self.rise_threshold

    def calculate_angle(self, x, y):
        a = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2
        b = (x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2
        c = (x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2
        if a == 0 or c == 0:
            angle = 0
        else:
            angle = abs(math.degrees(math.acos((a + c - b) / math.sqrt(4 * a * c))))
        return angle

    def _has_assist_behind_head(self, points, person_boxes, selected_idx, w) -> bool:
        """检测头部后方是否出现其他人（疑似推背助力）。

        脚→头横轴几何：dir_x = sign(head_x - heal_x)，头后方横轴范围(assist_behind_dx)
        内、且框中心 y 落在被测者框中心 y 容差带(assist_band_dy)内的非被测者框即判为
        后方有人。纯横纵轴判定，不用身体方向投影——起身时方向向量转竖直会把上方行人
        误判为头后方。任何参考量缺失（heal_x 未缓存、方向退化、被测者框越界）一律
        返回 False：无法判定时不误报。
        """
        if not person_boxes or not (0 <= selected_idx < len(person_boxes)):
            return False
        if self.heal_x == 0:
            return False  # 脚踝参考点未缓存，无法定方向

        head_x = points[0].x
        if head_x == self.heal_x:
            return False  # 方向退化为 0
        dir_x = 1 if head_x > self.heal_x else -1  # 脚→头横轴方向

        behind_max = w * self.assist_behind_dx
        band_max = w * self.assist_band_dy
        subj = person_boxes[selected_idx]
        subj_cy = (subj[1] + subj[3]) / 2.0

        for i, box in enumerate(person_boxes):
            if i == selected_idx:
                continue
            bcx = (box[0] + box[2]) / 2.0
            bcy = (box[1] + box[3]) / 2.0
            # 横轴：在头部后方，0 < 横向距离 ≤ behind_max
            if not (0 < (bcx - head_x) * dir_x <= behind_max):
                continue
            # 纵轴：与被测者框中心的纵坐标带重合
            if abs(bcy - subj_cy) <= band_max:
                return True
        return False


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

    def _sit_ok(self, points) -> bool:
        """坐姿判定（阶段一）：臀部贴地（不高于脚踝40px）且双腿屈膝（双膝高于脚踝20px，非伸直）。"""
        return (
            points[11].y >= points[15].y - 40
            and points[12].y >= points[16].y - 40
            and points[13].y <= points[15].y - 20
            and points[14].y <= points[16].y - 20
        )

    def _face_admitted(self, task) -> bool:
        """阶段一坐姿合格后的人脸核验：门禁未接 → 放行；否则裁剪选中者并询 FaceClient。"""
        if self.face_client is None:
            return True
        data = task.get_data() or {}
        boxes = data.get("person_boxes") or []
        sel = data.get("selected_idx", -1)
        crop = None
        if 0 <= sel < len(boxes):
            crop = self._crop(task.get_frame(), boxes[sel])
        return self.face_client.face_recognition(task.get_uid(), crop)

    @staticmethod
    def _crop(frame, box):
        """按 YOLO 框裁剪当前帧（同坐标系直接切片；copy 脱离 SHM，pickle 安全）。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2].copy()

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
        person_boxes = data.get("person_boxes") or []
        selected_idx = data.get("selected_idx", -1)
        drawing = config.get("drawing", True)
        start_count = config.get("start", False)

        if not has_person:
            return None

        hints = []

        # =================================
        # 参考线标识
        # ================================

        # 预设参考地面线两点坐标（左上、右下）
        h, w = img.shape[:2]
        x01, y001 = 0, h
        x02, y002 = w, h // 2

        # k0:拟合地面基线斜率 b0:拟合基线截距
        k0, b0 = 0, 0
        # k1：预设参考地面直线斜率
        k1 = (y001 - y002) / (x01 - x02)
        x3 = w // 2  # 参考线中点X

        # 左肩(5)、左脚踝(15)像素坐标，用于拟合地面基准线
        x1 = int(points[5].x)
        y1 = int(points[5].y)
        x2 = int(points[15].x)
        y2 = int(points[15].y)

        # 未正式测试时，绘制绿色预设参考地面线+中点标记
        if self.state < State.FLAT:
            if drawing:
                cv2.line(img, (x01, y001), (x02, y002), (0, 255, 0), 3)
                cv2.circle(img, (x3, int(x3 * k1 + y001)), 5, (255, 0, 0), -1)

            # 参考线与当前肩脚连线夹角差≥5°且未进入测试，缓存肩、脚踝坐标用于拟合地面线
            # （等待坐下/人脸阶段冻结：拟合只在等躺下阶段更新，测试期冻结沿用原行为）
            if self.state == State.READY and abs(math.degrees(math.atan(k0)) - math.degrees(math.atan(k1))) >= 5:
                self.shoulder_x = x1
                self.shoulder_y = y1
                self.heal_x = x2
                self.heal_y = y2

        # 使用最新一组肩、脚踝点计算拟合地面基线 k0斜率 b截距，避免除零
        k0 = (
            (self.shoulder_y - self.heal_y) / (self.shoulder_x - self.heal_x)
            if self.shoulder_x - self.heal_x != 0
            else 0
        )
        b0 = self.shoulder_y - k0 * self.shoulder_x
        # 计算拟合基线在x01/x02两点对应的Y值
        y01, y02 = int(k0 * x01 + b0), int(k0 * x02 + b0)
        
        if drawing and self.state >= State.READY:
            # 绘制红色拟合地面基准线（等躺下之前拟合值未更新，不画）
            cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)

        # =================================
        # 计算关键部位角度
        # ================================

        # ========== 计算躯干与地面夹角 body_ground_angle ==========
        A, B = [0.0] * 3, [0.0] * 3
        A[0] = points[5].x   # 左肩X
        A[1] = points[11].x  # 左髋X（脚底侧）
        A[2] = w
        B[0] = points[5].y   # 左肩Y
        B[1] = points[11].y  # 左髋Y
        B[2] = points[11].y
        body_ground_angle = self.calculate_angle(A, B)
        self.wucha = 8  # 固定角度补偿误差
        self.obsY3.append(body_ground_angle - abs((math.atan(k0)) / math.pi * 180))

        # ========== 计算右手臂肩肘夹角 shoulder_elbow_angle ==========
        A[0] = points[10].x  # 右腕X
        A[1] = points[8].x   # 右肘X
        A[2] = points[6].x   # 右肩X
        B[0] = points[10].y  # 右腕Y
        B[1] = points[8].y   # 右肘Y
        B[2] = points[6].y   # 右肩Y
        shoulder_elbow_angle = self.calculate_angle(A, B)

        # ========== 计算左手臂肩肘夹角 shoulder_elbow_angle2 ==========
        A[0] = points[9].x   # 左腕X
        A[1] = points[7].x   # 左肘X
        A[2] = points[5].x   # 左肩X
        B[0] = points[9].y   # 左腕Y
        B[1] = points[7].y   # 左肘Y
        B[2] = points[5].y   # 左肩Y
        shoulder_elbow_angle2 = self.calculate_angle(A, B)

        self.obsY2.append(shoulder_elbow_angle)
        self.obsY4.append(shoulder_elbow_angle2)

        # ========== 脚底地面基准夹角 base_angle ==========
        A[0] = points[11].x  # 左臀部X
        A[1] = points[15].x  # 左脚踝X
        A[2] = w
        B[0] = points[11].y  # 左臀部Y
        B[1] = points[15].y  # 左脚踝Y
        B[2] = points[15].y  # 左脚踝Y
        base_angle = self.calculate_angle(A, B)

        self.obsY5.append(body_ground_angle - abs((math.atan(k1)) / math.pi * 180))

        # ========== 阶段一：等待坐下 → 人脸核验 → 通过转等躺下（READY） ==========
        if self.state == State.WAIT:
            if self._sit_ok(points):
                if points[1].y > 0 and points[1].x > 0:
                    # 双脚在画面内，累计合格帧
                    if (
                        points[15].y <= h
                        and points[16].y <= h
                        and points[15].x <= w
                        and points[16].x <= w
                    ):
                        self.env_ok_count += 1
                        # 连续10帧环境正常，标记环境合格env_ifok=1
                        if self.env_ok_count >= 10:
                            self.env_ifok = 1
                            self.state = State.FACE
                            self.env_ok_count = 0
                    else:
                        self.env_ifok = -1  # 脚出框
                        self.env_ok_count = 0
                else:
                    self.env_ifok = -2  # 头部关键点丢失
                    self.env_ok_count = 0
        if self.state == State.FACE:
            if self._sit_ok(points) and self._face_admitted(task):
                self.state = State.READY

        # ========== 环境状态文字提示分支（按状态/环境异常） ==========
        if self.state == State.WAIT:
            # 等待坐下（人脸提示由 data.face 驱动前端 faceHint）
            hints.append("请坐下，双腿屈膝固定，准备测试")
            if drawing:
                img = self.img_put_text(img, "请坐下，双腿屈膝固定，准备测试", 25, 15, (0, 0, 255))
        elif self.state == State.READY:
            # 人脸已过，等待躺下
            if self.env_ifok == 1:
                hints.append("请平躺，双腿保持不动")
                if drawing:
                    img = self.img_put_text(img, "请平躺，双腿保持不动", 25, 15, (0, 0, 255))
            elif self.env_ifok == -1:
                hints.append("脚出画框，请重新调整")
                if drawing:
                    img = self.img_put_text(img, "脚出画框，请重新调整", 25, 15, (0, 0, 255))
            elif self.env_ifok == -2:
                hints.append("头部出画框，请重新调整")
                if drawing:
                    img = self.img_put_text(img, "头部出画框，请重新调整", 25, 15, (0, 0, 255))
            elif self.env_ifok == -3:
                hints.append("斜度大于15°，请重新调整")
                if drawing:
                    img = self.img_put_text(img, "斜度大于15°，请重新调整", 25, 15, (0, 0, 255))

        # ========== 等躺下阶段：躺下合格进入测试（FLAT） ==========
        # 躯干脚底夹角差值<=10，肩、头、脚关键点在地面线下方
        if (
            self.state == State.READY
            and body_ground_angle - base_angle <= 10
            and self.env_ifok == 1
            and points[15].y <= k0 * points[15].x + b0 + 15
            and points[11].y <= k0 * points[11].x + b0 + 15
            and points[5].y <= k0 * points[5].x + b0 + 15
        ):
            self.ready_count += 1
            # 10帧预备姿态，正式开启测试
            if self.ready_count >= 10:
                self.state = State.FLAT
                self.ready_time = time.time()

        if (
            self.ready_time is not None
            and time.time() - self.ready_time <= 3
        ):
            hints.clear()
            hints.append("检测成功！")
            hints.append("请开始测试")
            if drawing:
                img = self.img_put_text(img, "检测成功！", 25, 15, (0, 0, 255))
                img = self.img_put_text(img, "请开始测试", 400, 50, (255, 165, 0))

        body_ground_angle2 = body_ground_angle
        is_new_rise = False
       
        # ========== 正式测试阶段主逻辑 ==========
        if self.state >= State.FLAT and self.state < State.FINISHED:

            # ========== 修正躯干夹角body_ground_angle2（分段补偿误差） ==========
            # 不清楚为什么这么做
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
            # 作弊1：手臂夹角≥135，连续3帧判定手臂打开借力
            if shoulder_elbow_angle >= 135 and self.is_mid(body_ground_angle2):
                self.arm_force_count += 1
            else:
                self.arm_force_count = 0

            if self.arm_force_count >= 3 and self.state == State.FLAT:
                self.is_arm_force_cheat = True

            # 作弊2：人体关键点大面积出画面，判定替考换人
            if (
                min(points[1].y, points[2].y) < 0
                or min(points[3].x, points[4].x) < 0
                or max(points[3].x, points[4].x) > w
                or max(points[13].y, points[14].y) > h
                or min(points[13].x, points[14].x) < 0
                or max(points[13].x, points[14].x) > w
            ):
                self.is_out_of_bounds = True

            # 记录波峰波谷
            self.last_flat_angle = min(body_ground_angle2, self.last_flat_angle)
            self.last_rise_angle = max(body_ground_angle2, self.last_rise_angle)

            # 作弊3：波谷角度>30 → 肩背未贴地
            if (
                self.is_rise(body_ground_angle2)
                and self.is_mid(self.last_flat_angle)
            ):
                self.is_flat_insufficient = True

            # 作弊4：波峰角度<40 → 起身高度不足
            if (
                self.is_flat(body_ground_angle2)
                and self.is_mid(self.last_rise_angle)
            ):
                self.is_rise_insufficient = True

            # 作弊5：腿部移动/其他人抢走关键点
            if self.weighted_knee_point == None:
                self.weighted_knee_point = points[13]

            if (
                self.weighted_knee_point.x - points[13].x > w * self.knee_move_threshold
                or self.weighted_knee_point.y - points[13].y > w * self.knee_move_threshold
            ):
                self.is_knee_moved = True

            self.weighted_knee_point.x = self.weighted_knee_point.x * (1 - self.knee_move_alpha) + points[13].x * self.knee_move_alpha
            self.weighted_knee_point.y = self.weighted_knee_point.y * (1 - self.knee_move_alpha) + points[13].y * self.knee_move_alpha

            # 作弊6：头部后方有人长时间驻留，判定推背助力
            if self._has_assist_behind_head(points, person_boxes, selected_idx, w):
                self.assist_count += 1
            else:
                self.assist_count = 0

            if self.assist_count >= self.assist_hold_frames:
                self.is_assist_cheat = True

            # 平躺判定逻辑
            if (
                self.state == State.RISE
                or self.state == State.WAIT_FLAT
            ):
                if self.is_flat(body_ground_angle2):
                    self.state = State.FLAT
                    self.is_flat_insufficient = False
                    self.is_arm_force_cheat = False
                    self.last_rise_angle = 0

            # 起身判定逻辑
            if self.state == State.FLAT:
                if self.is_rise(body_ground_angle2):
                    self.state = State.RISE
                    self.is_rise_insufficient = False
                    self.last_flat_angle = 180
                    is_new_rise = True

            # ========== 有效计数num累加逻辑 ==========
            if self.is_out_of_bounds or self.is_knee_moved or self.is_assist_cheat:
                self.state = State.FINISHED
                is_new_rise = False

            elif self.is_arm_force_cheat:
                self.state = State.WAIT_FLAT
                is_new_rise = False

            if is_new_rise and start_count:
                self.num += 1

        if self.state >= State.FLAT:
            # ========== 作弊状态提示 ==========
            if self.is_arm_force_cheat:
                hints.append("手臂打开借力")
            if self.is_flat_insufficient:
                hints.append("肩背未着地")
            if self.is_rise_insufficient:
                hints.append("起身角度不足")
            if self.is_out_of_bounds:
                hints.append("测试者出画面")
            if self.is_knee_moved:
                hints.append("测试者大幅移动")
            if 0 < self.assist_count < self.assist_hold_frames:
                hints.append("后方有人")  # 驻留计数未满 75 帧的预警
            if self.is_assist_cheat:
                hints.append("疑似助力作弊")

            # ========== 作弊状态绘制 ==========
            if drawing:
                hint_y = 350
                if self.is_arm_force_cheat:
                    img = self.img_put_text(img, "手臂打开借力", 25, hint_y, (255, 0, 255))
                    hint_y += 25
                if self.is_flat_insufficient:
                    img = self.img_put_text(img, "肩背未着地", 25, hint_y, (255, 0, 255))
                    hint_y += 25
                if self.is_rise_insufficient:
                    img = self.img_put_text(img, "起身角度不足", 25, hint_y, (255, 0, 255))
                    hint_y += 25
                if self.is_out_of_bounds:
                    img = self.img_put_text(img, "测试者出画面", 25, hint_y, (255, 0, 255))
                    hint_y += 25
                if self.is_knee_moved:
                    img = self.img_put_text(img, "测试者大幅移动", 25, hint_y, (255, 0, 255))
                    hint_y += 25
                if 0 < self.assist_count < self.assist_hold_frames:
                    img = self.img_put_text(img, "后方有人", 25, hint_y, (0, 255, 255))
                    hint_y += 25
                if self.is_assist_cheat:
                    img = self.img_put_text(img, "疑似助力作弊", 25, hint_y, (255, 0, 255))
                    hint_y += 25

        # ========== 结束阶段主逻辑 ==========
        if self.state == State.FINISHED:
            hints.append("测试结束！")
            if drawing:
                img = self.img_put_text(img, "测试结束！", 25, 50, (0, 0, 255))

        return {
            "frame_id": frame_id,
            "ready": self.state >= State.FLAT,
            "score": self.num,
            "angle": body_ground_angle2,
            "hints": hints,
            "face": self.face_client.status() if self.face_client else None,
            "is_waiting_for_flat": self.env_ifok == -4,
            "is_new_rise": is_new_rise,
            "is_arm_force_cheat": self.is_arm_force_cheat,
            "is_out_of_bounds": self.is_out_of_bounds,
            "is_flat_insufficient": self.is_flat_insufficient,
            "is_rise_insufficient": self.is_rise_insufficient,
            "is_knee_move": self.is_knee_moved,
            "is_assist_cheat": self.is_assist_cheat,
            "timestamp": int(round(time.time() * 1000)),
        }


from .base import BaseSport  # 包装类依赖（置于尾部，不打断原文件结构）


class SportSitup(BaseSport):
    """仰卧起坐：YOLO Pose 推理 + 状态机评测的链式封装。"""

    def __init__(self):
        self._inferencer = None
        self._evaluator = None
        self._face_client = None     # 人脸识别客户端（set_face_client 注入）

    def load(self):
        """加载模型与评测状态机（用户进程内调用，耗时秒级，每进程一次）。"""
        self._inferencer = Inferencer()
        self._evaluator = Evaluator(face_client=self._face_client)

    def reset(self):
        """换会话复位：重建评测状态机，推理模型保留（进程复用，ADR-0003）。"""
        self._evaluator = Evaluator(face_client=self._face_client)

    def process(self, frame, config, frame_id, uid=0, timestamp_ms=None):
        """处理一帧（就地绘制），返回评测数据 dict；本帧无有效数据返回 None。"""
        if self._inferencer is None:
            raise RuntimeError('SportSitup 未调用 load()')
        task = SportTask(uid=uid, frame_id=frame_id, sport_type='situp',
                         config=config if config is not None else {})
        task.set_frame(frame)
        task.set_data(self._inferencer.process(task))
        return self._evaluator.process(task)

    def close(self):
        self._inferencer = None
        self._evaluator = None

