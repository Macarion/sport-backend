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

# ---------- 引体向上传感（MediaPipe；第 8 步换 YOLO 后删除本节） ----------
PULLUP_STATIC_IMAGE_MODE = True
