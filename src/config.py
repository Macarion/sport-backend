# config.py
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

# ---------- 项目配置 ----------
SPORTS = {
    'situp': {
        'frame_shape': (480, 640, 3)
    },
    'pullup': {
        'frame_shape': (480, 640, 3)
    },
    'sitreach': {
        'frame_shape': (480, 640, 3)
    },
    'jump': {
        'frame_shape': (720, 1280, 3)
    }
}

# ---------- 动态伸缩参数 ----------
USERS_PER_PROCESS = 0.5         # 每个进程的用户数，小于1表示多个进程处理一个用户
MAX_PROCESSES_PER_PROJECT = 4   # 每个项目最多启动的进程数
IDLE_TIMEOUT = 60               # 进程空闲超时时间（秒）

# ---------- 共享内存池配置 ----------
SHM_POOL_SIZE = 30
SHM_RECLAIM_TIMEOUT = 10  # 共享内存块超时回收时间（秒）

MAX_FRAME_HEIGHT = max(project['frame_shape'][0] for project in SPORTS.values())
MAX_FRAME_WIDTH = max(project['frame_shape'][1] for project in SPORTS.values())
MAX_FRAME_CHANNELS = max(project['frame_shape'][2] for project in SPORTS.values())
MAX_FRAME_BYTES = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * MAX_FRAME_CHANNELS

# ---------- 进程间通信（IPC）配置 ----------
MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50000
AUTH_KEY = b'bodytest_secret_key'

# ---------- 仰卧起坐配置 ----------
SITUP_STATIC_IMAGE_MODE = True

SITUP_MARKER_ENABLED = False
SITUP_MARKER_LOWER = (35, 50, 100)
SITUP_MARKER_UPPER = (100, 255, 255)