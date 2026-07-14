# config.py
import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent

# ---------- 项目与模型配置 ----------
# 定义所有体测项目及其对应的模型文件路径
PROJECTS = {
    'situp': {
        'model_path': str(BASE_DIR / 'models' / 'situp.h5'),
        'input_size': (640, 480)  # 宽, 高
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
USERS_PER_PROCESS = 2      # 每个推理进程承载的用户数（根据推理耗时调整）
MAX_PROCESSES_PER_PROJECT = 4  # 每个项目最大进程数（防止内存爆炸）
IDLE_TIMEOUT = 60          # 子进程空闲超时（秒），超时后自杀

# ---------- 共享内存池配置 ----------
SHM_POOL_SIZE = 30         # 预分配内存块数量（支持并发用户数 * 2）
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CHANNELS = 3
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS  # ~0.9MB

# ---------- Launcher 调度间隔 ----------
SCAN_INTERVAL = 5          # Launcher 扫描扩缩容的间隔（秒）

# ---------- 进程间通信（IPC）配置 ----------
MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50000
AUTH_KEY = b'bodytest_secret_key'  # 连接认证密钥