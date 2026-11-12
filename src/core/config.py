# core/config.py
"""统一应用配置：业务常量、绝对路径、端口与日志级别。

零副作用：不配置 logging（唯一日志配置点在 core/log.py）、不读 Django settings。
"""
import os
import sys
from pathlib import Path

# ---------- 路径（锚定仓库结构；第 2 步起消费方全部改走这些常量） ----------
SRC_DIR = Path(__file__).resolve().parents[1]        # sport-backend/src
BACKEND_ROOT = SRC_DIR.parent                        # sport-backend
VENDOR_DIR = BACKEND_ROOT / 'vendor'                 # 资产区：模型 / 字体 / 第三方
sys.path.insert(0, str(VENDOR_DIR))  # face_sdk 等第三方包（import core.config 即生效）
FONT_DIR = VENDOR_DIR / 'fonts'
REPORT_DIR = SRC_DIR / 'report_pdf'                  # 报告读写同处（与 views 读取端一致）
FACE_MODEL_DIR = VENDOR_DIR / 'face_sdk' / 'models'
FACE_DB_PATH = BACKEND_ROOT / 'data' / 'faces.db'

# ---------- 进程与会话 ----------
IDLE_TIMEOUT = 60                # 会话空闲销毁与 worker 驻留同口径（秒）
DAPHNE_PORT = int(os.environ.get('DAPHNE_PORT', '8090'))
PREWARM_ENABLED = False          # 开机预热子进程：默认关（模型打热文件缓存，有效性存疑）

# ---------- 人脸门禁（第 7 步，ADR-0004） ----------
FACE_GATE_ENABLED = True        # 总开关：True 时 spawn 人脸服务并在 situp 转 READY 前门禁（默认关）
FACE_GATE_TIMEOUT = 3           # 识别应答超时（秒）：超时按未通过处理并清在途，下一帧重试
FACE_MIN_SEND_INTERVAL = 0.5     # 客户端硬限流：同一 client 两次 send 的最小间隔（秒），应答再快也不放宽

# ---------- 日志 ----------
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# ---------- 摄像头 ----------
CAMERA_INDEX = 0                 # 后端直接开摄像头时的设备号

# ---------- 姿态推理规格（仰卧起坐 / 引体向上共用 YOLO Pose） ----------
POSE_YOLO_MODEL_PATH = str(VENDOR_DIR / 'yolo_pose' / 'yolov8n-pose.pt')
POSE_YOLO_IMG_SZ = 640
POSE_YOLO_BOX_CONF = 0.5
POSE_YOLO_KPT_CONF = 0.5

# ---------- 仰卧起坐标志物 ----------
SITUP_MARKER_ENABLED = False
SITUP_MARKER_LOWER = (35, 50, 100)
SITUP_MARKER_UPPER = (100, 255, 255)

# ---------- 仰卧起坐胯部漂移（换人误检忽略 / 大幅移动判定） ----------
SITUP_HIP_DRIFT_WINDOW = 1.0     # 连续无效墙钟时长（秒），超过即判"大幅移动"作弊
SITUP_HIP_DRIFT_THRESHOLD = 0.12  # 胯部中点与参照的分轴偏移阈值（1 = 画面宽）
SITUP_HIP_DRIFT_ALPHA = 0.05      # 胯部参照 EMA 权重（原膝盖检测同值）
SITUP_HIP_DRIFT_EMA_MAX = 0.05    # EMA 更新准入：分轴位移≤该值才慢跟随（1=画面宽）
SITUP_HIP_DIR_DEADBAND = 0.05     # 胯膝左右方向死区：水平距离＜该值不判方向（1=画面宽）
SITUP_HIP_DRIFT_DRAW = True      # 是否绘制胯部有效区域矩形（参照 ± 分轴阈值，纯可视化）

# ---------- 引体向上传感（MediaPipe 遗留常量；第 8 步已换 YOLO，按用户决定死代码先保留） ----------
PULLUP_STATIC_IMAGE_MODE = True
