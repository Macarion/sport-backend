# config.py
import os
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------- 日志配置 ----------
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)


def get_logger(name):
    return logging.getLogger(name)

# ---------- 项目与模型配置 ----------
PROJECTS = {
    'situp': {
        'model_path': str(BASE_DIR / 'models' / 'situp.h5'),
        'input_size': (640, 480)
    },
    'pullup': {
        'model_path': str(BASE_DIR / 'models' / 'pullup.h5'),
        'input_size': (640, 480)
    },
    'sitreach': {
        'model_path': str(BASE_DIR / 'models' / 'sitreach.h5'),
        'input_size': (640, 480)
    },
    'jump': {
        'model_path': str(BASE_DIR / 'models' / 'jump.h5'),
        'input_size': (640, 480)
    }
}

# ---------- 动态伸缩参数 ----------
USERS_PER_PROCESS = 2
MAX_PROCESSES_PER_PROJECT = 4
IDLE_TIMEOUT = 60

# ---------- 共享内存池配置 ----------
SHM_POOL_SIZE = 30
SHM_RECLAIM_TIMEOUT = 10  # 共享内存块超时回收时间（秒）
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CHANNELS = 3
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS

# ---------- 进程间通信（IPC）配置 ----------
MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50000
AUTH_KEY = b'bodytest_secret_key'
