# 摄像头ID，默认为0，仅在后端使用摄像头时有效
CAMERA_INDEX = 0

import os
import logging

# ---------- 日志配置 ----------
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)


def get_logger(name):
    return logging.getLogger(name)
