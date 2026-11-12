# sports/situp_eval.py
"""仰卧起坐评测：状态机 + 表驱动提示。

几何计算走模块级 angle_deg（顶点夹角）与 compensate_angle（测试期分段补偿）；
提示文案与绘制统一由 _emit_hints 按表生成（一次 PIL 往返画完）。
推理侧见 situp_infer.py，装配见 situp.py。
"""
import math
import time

import cv2
import numpy as np
from enum import IntEnum
from PIL import Image, ImageDraw, ImageFont

from core.config import FONT_DIR
from core.task import SportTask

from .situp_infer import PosePoint


WUCHA = 8  # 固定角度补偿误差（原 self.wucha）


def xy(p: PosePoint) -> tuple:
    """PosePoint → (x, y) 元组（angle_deg 入参用）。"""
    return (p.x, p.y)


def angle_deg(p0, p1, p2) -> float:
    """三点夹角（度），顶点为中间参数 p1；入参均为 (x, y) 元组。

    余弦定理（与原 calculate_angle 同公式）：a=|p0-p1|²、b=|p0-p2|²、c=|p1-p2|²；
    acos 参数 clamp 到 [-1, 1] 防浮点越界；退化边长（a 或 c 为 0）返回 0。
    """
    a = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
    b = (p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2
    c = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    if a == 0 or c == 0:
        return 0.0
    cos_t = (a + c - b) / math.sqrt(4 * a * c)
    return abs(math.degrees(math.acos(max(-1.0, min(1.0, cos_t)))))


def compensate_angle(body_ground: float, ground_slope: float) -> float:
    """躯干角分段补偿（测试期专用）：先减去拟合地面线倾角，再按区间修正。

    等价表（由原 4 个互相覆盖的 if 化简，行为逐位一致，含末尾封顶 60）：
        t = body_ground - |atan(ground_slope)|（度）
        t ≤ 15      → t + WUCHA
        15 < t ≤ 30 → t + WUCHA + 10
        t > 30      → 60
    """
    t = body_ground - abs(math.degrees(math.atan(ground_slope)))
    if t <= 15:
        v = t + WUCHA
    elif t <= 30:
        v = t + WUCHA + 10
    else:
        v = 60
    return min(v, 60)


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
    """仰卧起坐评测状态机：process 每帧驱动，结论随返回 dict 承载（含 hints）。"""

    def __init__(self, face_client=None):
        self.face_client = face_client  # 人脸识别客户端（None=门禁关闭，直接放行）
        self.state = State.WAIT
        self.num = 0

        # 画面尺寸（process 每帧更新，各函数经 self.w/self.h 读取）
        self.w = 0
        self.h = 0

        # 地面基准线：FLAT 前每帧拟合更新，FLAT 后冻结只读
        self.ground_slope = 0.0   # 拟合地面基线斜率（原 k0）
        self.ground_offset = 0.0  # 拟合基线截距（原 b0）
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
        self.weighted_knee_point = None  # 膝盖移动加权平均点
        self.assist_count = 0  # 头后方有人驻留的连续帧计数
        self.assist_hold_frames = 75  # 驻留多少帧判定助力作弊
        self.assist_behind_dx = 0.25  # 头后方横轴检测范围(相对画面宽,沿脚→头方向)
        self.assist_band_dy = 0.15  # 纵坐标带容差(相对画面宽,与被测者框中心y比)
        self.flat_threshold = 20
        self.rise_threshold = 50
        self.last_flat_angle = 180
        self.last_rise_angle = 0

        # 流程状态标记
        self.env_ifok = 0  # 环境检测状态：0检测中/1合格/-1/-2各类环境异常
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

    def _has_assist_behind_head(self, points, person_boxes, selected_idx) -> bool:
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

        behind_max = self.w * self.assist_behind_dx
        band_max = self.w * self.assist_band_dy
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

    def _sit_ok(self, points) -> bool:
        """坐姿判定（阶段一）：臀部贴地（不高于脚踝 0.1h）且双腿屈膝（双膝高于脚踝 0.05h，非伸直）。"""
        return (
            points[11].y >= points[15].y - self.h * 0.1
            and points[12].y >= points[16].y - self.h * 0.1
            and points[13].y <= points[15].y - self.h * 0.05
            and points[14].y <= points[16].y - self.h * 0.05
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

    def _fit_ground_line(self, img, points, drawing):
        """预设/拟合地面参考线：FLAT 前每帧更新 ground_slope/offset，FLAT 后冻结只读。

        肩、脚踝坐标缓存只在 READY 阶段更新（等待/人脸阶段冻结），与原实现逐帧行为一致；
        拟合后的红色基准线在 READY 起绘制（含测试期，用冻结值）。
        """
        h, w = self.h, self.w

        # 预设参考地面线两点坐标（左上、右下）
        x01, y001 = 0, h
        x02, y002 = w, h // 2
        # 预设参考地面直线斜率（原 k1，仅本函数使用）
        preset_slope = (y001 - y002) / (x01 - x02)
        x3 = w // 2  # 参考线中点X

        # 左肩(5)、左脚踝(15)像素坐标，用于拟合地面基准线
        x1 = int(points[5].x)
        y1 = int(points[5].y)
        x2 = int(points[15].x)
        y2 = int(points[15].y)

        if self.state < State.FLAT:
            # 未正式测试时，绘制绿色预设参考地面线+中点标记
            if drawing:
                cv2.line(img, (x01, y001), (x02, y002), (0, 255, 0), 3)
                cv2.circle(img, (x3, int(x3 * preset_slope + y001)), 5, (255, 0, 0), -1)

            # 参考线与当前肩脚连线夹角差≥5°且未进入测试，缓存肩、脚踝坐标用于拟合地面线
            # （等待坐下/人脸阶段冻结：缓存只在等躺下阶段更新；判断沿用原语义——此处用
            # 拟合前的 0 与预设斜率比较，原代码 k0 在该点恒为 0）
            if self.state == State.READY and abs(math.degrees(math.atan(0)) - math.degrees(math.atan(preset_slope))) >= 5:
                self.shoulder_x = x1
                self.shoulder_y = y1
                self.heal_x = x2
                self.heal_y = y2

            # 用肩、脚踝缓存拟合地面基线斜率与截距，避免除零
            self.ground_slope = (
                (self.shoulder_y - self.heal_y) / (self.shoulder_x - self.heal_x)
                if self.shoulder_x - self.heal_x != 0
                else 0
            )
            self.ground_offset = self.shoulder_y - self.ground_slope * self.shoulder_x

        # 拟合基线在 x01/x02 两点对应的 Y 值（FLAT 后用冻结值）
        y01 = int(self.ground_slope * x01 + self.ground_offset)
        y02 = int(self.ground_slope * x02 + self.ground_offset)
        if drawing and self.state >= State.READY:
            # 绘制红色拟合地面基准线（等躺下之前拟合值未更新，不画）
            cv2.line(img, (x01, y01), (x02, y02), (0, 0, 255), 3)

    def _run_prep_phase(self, task, points):
        """阶段一状态推进：等待坐下 → 人脸核验 → 等躺下 → 躺下合格转测试（FLAT）。"""
        h, w = self.h, self.w

        # ========== 等待坐下：坐姿合格 + 双脚在画面内，连续10帧环境合格转人脸 ==========
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

        # ========== 人脸核验通过转等躺下（READY） ==========
        if self.state == State.FACE:
            if self._sit_ok(points) and self._face_admitted(task):
                self.state = State.READY

        # ========== 等躺下阶段：躺下合格进入测试（FLAT） ==========
        # 躯干与地面基准夹角差<=10，肩、头、脚关键点在拟合地面线下方（容差 0.05h）
        if self.state == State.READY:
            body_ground = angle_deg(xy(points[5]), xy(points[11]), (w, points[11].y))
            base_angle = angle_deg(xy(points[11]), xy(points[15]), (w, points[15].y))
            if (
                body_ground - base_angle <= 10
                and self.env_ifok == 1
                and points[15].y <= self.ground_slope * points[15].x + self.ground_offset + h * 0.05
                and points[11].y <= self.ground_slope * points[11].x + self.ground_offset + h * 0.05
                and points[5].y <= self.ground_slope * points[5].x + self.ground_offset + h * 0.05
            ):
                self.ready_count += 1
                # 10帧预备姿态，正式开启测试
                if self.ready_count >= 10:
                    self.state = State.FLAT
                    self.ready_time = time.time()

    def _run_test_phase(self, points, person_boxes, selected_idx, start_count, body_angle2):
        """正式测试主逻辑（FLAT..FINISHED）：补偿躯干角 → 作弊检测 → 转移与计数。

        返回 (is_new_rise, body_angle2)，body_angle2 为补偿后的值供返回值 angle 使用。
        """
        # ========== 分段补偿躯干角（等价表见 compensate_angle） ==========
        body_angle2 = compensate_angle(body_angle2, self.ground_slope)

        # ========== 各类作弊帧检测，赋值作弊标记 ==========
        self._cheat_arm_force(points, body_angle2)
        self._cheat_out_of_bounds(points)
        # 记录波峰波谷（须在肩背/起身检测之前，与原顺序一致）
        self.last_flat_angle = min(body_angle2, self.last_flat_angle)
        self.last_rise_angle = max(body_angle2, self.last_rise_angle)
        self._cheat_flat_insufficient(body_angle2)
        self._cheat_rise_insufficient(body_angle2)
        self._cheat_knee_move(points)
        self._cheat_assist(points, person_boxes, selected_idx)

        is_new_rise = False

        # ========== 平躺判定逻辑 ==========
        if (
            self.state == State.RISE
            or self.state == State.WAIT_FLAT
        ):
            if self.is_flat(body_angle2):
                self.state = State.FLAT
                self.is_flat_insufficient = False
                self.is_arm_force_cheat = False
                self.last_rise_angle = 0

        # ========== 起身判定逻辑 ==========
        if self.state == State.FLAT:
            if self.is_rise(body_angle2):
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

        return is_new_rise, body_angle2

    def _cheat_arm_force(self, points, body_angle2):
        """作弊1：手臂夹角≥135，连续3帧判定手臂打开借力。"""
        shoulder_elbow_angle = angle_deg(xy(points[10]), xy(points[8]), xy(points[6]))
        if shoulder_elbow_angle >= 135 and self.is_mid(body_angle2):
            self.arm_force_count += 1
        else:
            self.arm_force_count = 0

        if self.arm_force_count >= 3 and self.state == State.FLAT:
            self.is_arm_force_cheat = True

    def _cheat_out_of_bounds(self, points):
        """作弊2：人体关键点大面积出画面，判定替考换人。"""
        if (
            min(points[1].y, points[2].y) < 0
            or min(points[3].x, points[4].x) < 0
            or max(points[3].x, points[4].x) > self.w
            or max(points[13].y, points[14].y) > self.h
            or min(points[13].x, points[14].x) < 0
            or max(points[13].x, points[14].x) > self.w
        ):
            self.is_out_of_bounds = True

    def _cheat_flat_insufficient(self, body_angle2):
        """作弊3：波谷角度>30 → 肩背未着地。"""
        if self.is_rise(body_angle2) and self.is_mid(self.last_flat_angle):
            self.is_flat_insufficient = True

    def _cheat_rise_insufficient(self, body_angle2):
        """作弊4：波峰角度<40 → 起身高度不足。"""
        if self.is_flat(body_angle2) and self.is_mid(self.last_rise_angle):
            self.is_rise_insufficient = True

    def _cheat_knee_move(self, points):
        """作弊5：腿部移动/其他人抢走关键点（膝盖点加权漂移超阈值即判）。"""
        if self.weighted_knee_point is None:
            self.weighted_knee_point = points[13]

        if (
            self.weighted_knee_point.x - points[13].x > self.w * self.knee_move_threshold
            or self.weighted_knee_point.y - points[13].y > self.w * self.knee_move_threshold
        ):
            self.is_knee_moved = True

        self.weighted_knee_point.x = self.weighted_knee_point.x * (1 - self.knee_move_alpha) + points[13].x * self.knee_move_alpha
        self.weighted_knee_point.y = self.weighted_knee_point.y * (1 - self.knee_move_alpha) + points[13].y * self.knee_move_alpha

    def _cheat_assist(self, points, person_boxes, selected_idx):
        """作弊6：头部后方有人长时间驻留，判定推背助力。"""
        if self._has_assist_behind_head(points, person_boxes, selected_idx):
            self.assist_count += 1
        else:
            self.assist_count = 0

        if self.assist_count >= self.assist_hold_frames:
            self.is_assist_cheat = True

    def _draw_texts(self, img, rows):
        """在帧上绘制 (text, left, top, color) 列表：单次 PIL 往返画完全部文本。"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            fontStyle = ImageFont.truetype(str(FONT_DIR / "msyhbd.ttf"), 30)
        except:
            fontStyle = ImageFont.load_default()

        for text, left, top, textColor in rows:
            # 绘制边框（黑底描边）
            # for x in range(-2, 3):
            #     for y in range(-2, 3):
            #         if x != 0 or y != 0:
            #             draw.text((left + x, top + y), text, (0, 0, 0), font=fontStyle)

            # 绘制加粗的文本（通过多次绘制模拟加粗效果）
            # for i in range(1, 3):
            #     draw.text((left + i, top), text, textColor, font=fontStyle)
            #     draw.text((left - i, top), text, textColor, font=fontStyle)
            #     draw.text((left, top + i), text, textColor, font=fontStyle)
            #     draw.text((left, top - i), text, textColor, font=fontStyle)

            # 再绘制一次文本以覆盖边框的部分
            draw.text((left, top), text, textColor, font=fontStyle)

        np.copyto(img, cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR))
        return img

    def _emit_hints(self, img, drawing):
        """按状态/标记生成提示列表并（drawing 时）绘制全部文本。

        表驱动：每条提示只定义一次；顺序与原实现一致——状态类 → 成功 3 秒窗口
        （整体替换状态类，原 hints.clear() 语义）→ 作弊类（state≥FLAT，y 从 350
        起按激活行每条 +25 累加）→ 结束。
        """
        hints = []
        rows = []  # (text, left, top, color)

        # ---------- 状态类提示（成功窗口激活时整体替换） ----------
        success = self.ready_time is not None and time.time() - self.ready_time <= 3
        if success:
            hints.extend(["检测成功！", "请开始测试"])
            rows.extend([
                ("检测成功！", 25, 15, (0, 0, 255)),
                ("请开始测试", 400, 50, (255, 165, 0)),
            ])
        elif self.state == State.WAIT:
            # 等待坐下（人脸提示由 data.face 驱动前端 faceHint）
            hints.append("请坐下，双腿屈膝固定，准备测试")
            rows.append(("请坐下，双腿屈膝固定，准备测试", 25, 15, (0, 0, 255)))
        elif self.state == State.READY:
            # 人脸已过，等待躺下
            # 原 env_ifok==-3 分支"斜度大于15°，请重新调整"已随表重构移除：
            # 活代码从不产出 -3（只写 0/1/-1/-2），后续补上斜度检测逻辑时再加回此行。
            if self.env_ifok == 1:
                hints.append("请平躺，双腿保持不动")
                rows.append(("请平躺，双腿保持不动", 25, 15, (0, 0, 255)))
            elif self.env_ifok == -1:
                hints.append("脚出画框，请重新调整")
                rows.append(("脚出画框，请重新调整", 25, 15, (0, 0, 255)))
            elif self.env_ifok == -2:
                hints.append("头部出画框，请重新调整")
                rows.append(("头部出画框，请重新调整", 25, 15, (0, 0, 255)))

        # ---------- 作弊类提示（state >= FLAT 才出现） ----------
        if self.state >= State.FLAT:
            hint_y = 350
            cheat_rows = [
                (self.is_arm_force_cheat, "手臂打开借力", (255, 0, 255)),
                (self.is_flat_insufficient, "肩背未着地", (255, 0, 255)),
                (self.is_rise_insufficient, "起身角度不足", (255, 0, 255)),
                (self.is_out_of_bounds, "测试者出画面", (255, 0, 255)),
                (self.is_knee_moved, "测试者大幅移动", (255, 0, 255)),
                (0 < self.assist_count < self.assist_hold_frames, "后方有人", (0, 255, 255)),
                (self.is_assist_cheat, "疑似助力作弊", (255, 0, 255)),
            ]
            for active, text, color in cheat_rows:
                if active:
                    hints.append(text)
                    rows.append((text, 25, hint_y, color))
                    hint_y += 25

        # ---------- 结束 ----------
        if self.state == State.FINISHED:
            hints.append("测试结束！")
            rows.append(("测试结束！", 25, 50, (0, 0, 255)))

        if drawing and rows:
            img = self._draw_texts(img, rows)
        return hints, img

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

        # 每帧更新画面尺寸
        self.h, self.w = img.shape[:2]

        # 地面参考线：FLAT 前每帧拟合更新，之后冻结只读
        self._fit_ground_line(img, points, drawing)

        # 阶段一状态推进（等待/人脸/等躺下）
        self._run_prep_phase(task, points)

        # 躯干与地面夹角（raw；测试期内由 _run_test_phase 补偿后返回）
        body_angle2 = angle_deg(xy(points[5]), xy(points[11]), (self.w, points[11].y))

        # 正式测试阶段（FLAT..FINISHED）
        is_new_rise = False
        if self.state >= State.FLAT and self.state < State.FINISHED:
            is_new_rise, body_angle2 = self._run_test_phase(
                points, person_boxes, selected_idx, start_count, body_angle2,
            )

        # 提示：表驱动生成 hints 并（drawing 时）一次绘制
        hints, img = self._emit_hints(img, drawing)

        return {
            "frame_id": frame_id,
            "ready": self.state >= State.FLAT,
            "score": self.num,
            "angle": body_angle2,
            "hints": hints,
            "face": self.face_client.status() if self.face_client else None,
            "is_waiting_for_flat": self.state == State.READY,
            "is_new_rise": is_new_rise,
            "is_arm_force_cheat": self.is_arm_force_cheat,
            "is_out_of_bounds": self.is_out_of_bounds,
            "is_flat_insufficient": self.is_flat_insufficient,
            "is_rise_insufficient": self.is_rise_insufficient,
            "is_knee_move": self.is_knee_moved,
            "is_assist_cheat": self.is_assist_cheat,
            "timestamp": int(round(time.time() * 1000)),
        }
