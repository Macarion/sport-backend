"""
detector/measure.py

"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


# =============================================================================
# 1. 手部检测模块 (封装 MediaPipe，专门针对中指)
# =============================================================================
class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, detection_con=0.65, track_con=0.65):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 用于坐标平滑的历史记录
        self.prev_pos = None
        self.alpha = 0.6  # 平滑系数 (0~1)，越大越灵敏，越小越平滑

    def find_middle_finger(self, image: np.ndarray):
        """
        检测双手中指指尖。
        如果检测到两只手，返回两个中指的平均点；
        如果只检测到一只手，返回这一只手的中指点。
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        points = []

        if results.multi_hand_landmarks:
            h, w, c = image.shape

            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                cx, cy = float(lm.x * w), float(lm.y * h)
                points.append(np.array([cx, cy], dtype=np.float32))

        if not points:
            return None

        if len(points) == 1:
            finger_pos = points[0]
        else:
            finger_pos = (points[0] + points[1]) / 2.0

        # EMA 平滑
        if self.prev_pos is None:
            self.prev_pos = finger_pos
        else:
            self.prev_pos = self.prev_pos * (1 - self.alpha) + finger_pos * self.alpha

        return tuple(self.prev_pos)


# =============================================================================
# 2. 透视标定模块 (核心算法)
# =============================================================================
class PerspectiveCalibrator:
    def __init__(
            self,
            marker_size_cm: float,
            marker_gap_cm: float,
            marker_ids: Tuple[int, int] = (0, 1),
            zero_offset_cm: float = 0.0
    ):
        self.marker_size = marker_size_cm
        self.marker_gap = marker_gap_cm
        self.ids_target = marker_ids
        self.zero_offset_cm = zero_offset_cm


        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # === 改进 1: 优化检测参数 ===
        self.aruco_params = cv2.aruco.DetectorParameters()
        # 增加二值化尝试范围，应对阴影和光照不均
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 63
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        # 提高边缘修正精度
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # 略微放宽透视后的格网检查（针对侧拍大角度）
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.10

        # === 改进 2: 状态记忆变量 ===
        self.H = None
        self.H_inv = None
        self.last_H = None  # 上一次成功的 H 矩阵
        self.missing_frames = 0  # 连续丢失帧数计数
        self.MAX_MISSING_FRAMES = 30  # 允许丢失的最大帧数 (约1秒)

        # 构建物理坐标 (不变)
        obj_top = np.array([
            [0, 0], [marker_size_cm, 0],
            [marker_size_cm, marker_size_cm], [0, marker_size_cm]
        ], dtype=np.float32)

        offset_y = marker_size_cm + marker_gap_cm
        obj_bot = np.array([
            [0, offset_y], [marker_size_cm, offset_y],
            [marker_size_cm, offset_y + marker_size_cm], [0, offset_y + marker_size_cm]
        ], dtype=np.float32)

        self.obj_points = np.vstack((obj_top, obj_bot))

    def calibrate(self, image: np.ndarray) -> bool:
        """
        返回 True 表示当前处于已标定状态（可能是当前帧检测到的，也可能是沿用历史的）
        """
        # 1. 图像预处理：转灰度，适当增强对比度（可选）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. 检测
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        found_current_frame = False

        if ids is not None and len(ids) >= 2:
            ids = ids.flatten().tolist()
            id_top, id_bot = self.ids_target

            if id_top in ids and id_bot in ids:
                # 找到两个码，进行计算
                idx_top = ids.index(id_top)
                idx_bot = ids.index(id_bot)

                corners_top = corners[idx_top].reshape(4, 2)
                corners_bot = corners[idx_bot].reshape(4, 2)
                img_points = np.vstack((corners_top, corners_bot))

                # 计算 H
                H_new, _ = cv2.findHomography(img_points, self.obj_points, cv2.RANSAC, 5.0)

                if H_new is not None:
                    self.H = H_new
                    self.last_H = H_new
                    self.H_inv = np.linalg.inv(H_new)
                    self.missing_frames = 0
                    found_current_frame = True

        # === 改进 logic: 记忆功能 ===
        if found_current_frame:
            return True
        else:
            # 当前帧没找到，查看是否有历史缓存
            if self.last_H is not None and self.missing_frames < self.MAX_MISSING_FRAMES:
                self.missing_frames += 1
                self.H = self.last_H
                self.H_inv = np.linalg.inv(self.last_H)
                # 可以在画面上打印警告，表示正在使用缓存
                cv2.putText(image, f"Keep-Alive: {self.missing_frames}/{self.MAX_MISSING_FRAMES}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                return True
            else:
                # 丢失太久，重置
                self.H = None
                self.H_inv = None
                return False

    def pixel_to_world(self, px_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.H is None:
            return None

        pt = np.array([[px_point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self.H)

        x_cm = dst[0][0][0]
        y_cm = dst[0][0][1] - self.zero_offset_cm

        return (x_cm, y_cm)

    def world_to_pixel(self, world_point: Tuple[float, float]) -> Tuple[int, int]:
        """ 将世界坐标 (x_cm, y_cm) 投影回 屏幕像素坐标 (用于绘图) """
        if self.H_inv is None:
            return (0, 0)

        pt = np.array([[world_point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self.H_inv)
        return (int(dst[0][0][0]), int(dst[0][0][1]))

    def draw_augmented_reality(self, image, finger_world_y=None):
        """ 在图像上绘制透视网格、原点和成绩线 """
        if self.H_inv is None:
            return

        # 1. 绘制世界坐标系原点
        origin_px = self.world_to_pixel((0, 0))
        cv2.circle(image, origin_px, 8, (0, 0, 255), -1)
        cv2.putText(image, "Origin (0,0)", (origin_px[0] + 10, origin_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 2. 绘制距离刻度线 (Grid) - 模拟地板上的线
        # 假设测量范围从 0cm 到 60cm
        for score_cm in range(-20, 51, 10):
            draw_y = score_cm + self.zero_offset_cm

            p_start = self.world_to_pixel((-10, draw_y))
            p_end = self.world_to_pixel((self.marker_size + 10, draw_y))

            color = (0, 0, 255) if score_cm == 0 else (0, 255, 0)

            cv2.line(image, p_start, p_end, color, 1)
            cv2.putText(image, f"{score_cm}cm", (p_start[0] - 55, p_start[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 3. 如果检测到中指，绘制成绩投影线
        if finger_world_y is not None:
            # 在该 Y 处画一条横线（青色）
            draw_y = finger_world_y + self.zero_offset_cm

            fw_start = self.world_to_pixel((-15, draw_y))
            fw_end = self.world_to_pixel((self.marker_size + 15, draw_y))

            cv2.line(image, fw_start, fw_end, (255, 255, 0), 2)

            # 绘制原点到当前线的连线（表示测量距离）
            # 沿着 X = center 的轴线
            center_x = self.marker_size / 2
            axis_top = self.world_to_pixel((center_x, self.zero_offset_cm))
            axis_cur = self.world_to_pixel((center_x, draw_y))
            cv2.line(image, axis_top, axis_cur, (0, 165, 255), 2, cv2.LINE_AA)


# =============================================================================
# 3. 主程序
# =============================================================================
def main():
    # --- 参数设置 (请根据实际情况修改) ---
    ARUCO_SIZE_CM = 5.6  # 打印出来的 ArUco 黑色方块边长
    ARUCO_GAP_CM = 16.0  # 上方码(ID 0)底部 到 下方码(ID 1)顶部 的距离

    # 假设“脚跟线”就在 ID 0 的位置。如果是坐位体前屈，通常脚跟是 0 点。
    # 这里我们定义：ID 0 左上角为 Y=0。如果脚跟在 ID 0 下方 10cm，结果需减去 10。

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # 初始化模块
    calibrator = PerspectiveCalibrator(ARUCO_SIZE_CM, ARUCO_GAP_CM, marker_ids=(0, 1))
    detector = HandDetector(max_hands=1)

    print("[INFO] 系统启动...")
    print(f"[INFO] 请放置两个 ArUco 码 (ID 0 上, ID 1 下)，间距 {ARUCO_GAP_CM} cm")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 尝试标定 (每帧更新，适应相机移动；如果是固定相机，可只标定一次锁死)
        is_calibrated = calibrator.calibrate(frame)

        # 2. 手部检测 (检测中指)
        finger_pt_px = detector.find_middle_finger(frame)  # 返回 (x, y) 像素

        score_cm = None

        # 3. 计算与绘制
        if is_calibrated:
            # 画透视网格
            calibrator.draw_augmented_reality(frame, finger_world_y=score_cm)

            if finger_pt_px:
                # 坐标变换：像素 -> 世界
                world_pt = calibrator.pixel_to_world(finger_pt_px)

                if world_pt:
                    # 坐位体前屈关注的是 Y 轴方向的伸展距离
                    score_cm = world_pt[1]

                    # 重新带入 score_cm 绘制结果线
                    calibrator.draw_augmented_reality(frame, finger_world_y=score_cm)

                    # 屏幕显示具体数值
                    fx, fy = int(finger_pt_px[0]), int(finger_pt_px[1])
                    cv2.circle(frame, (fx, fy), 10, (0, 255, 255), -1)
                    cv2.putText(frame, f"Score: {score_cm:.1f} cm", (fx + 15, fy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # 状态提示
            cv2.putText(frame, "Mode: Perspective Corrected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No ArUco Found (Need ID 0 & 1)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Sit-and-Reach (Middle Finger + Perspective)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()