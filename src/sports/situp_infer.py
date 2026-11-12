# sports/situp_infer.py
"""仰卧起坐推理侧：PosePoint 数据结构 + YOLO Pose 推理与绘制（评测见 situp_eval.py）。"""
import math

import cv2
import numpy as np
from typing import Optional

from core.config import (
    POSE_YOLO_BOX_CONF, POSE_YOLO_IMG_SZ, POSE_YOLO_KPT_CONF,
    POSE_YOLO_MODEL_PATH, SITUP_MARKER_ENABLED, SITUP_MARKER_LOWER,
    SITUP_MARKER_UPPER,
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
                PosePoint(i, int(kpt[0]), int(kpt[1]), 0.0)
                for i, kpt in enumerate(kpt_xy)
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
