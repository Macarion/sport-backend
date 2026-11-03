# core/shm.py
"""per-user 共享内存槽：父进程创建并全程持有句柄，用户进程按帧 attach。"""
import numpy as np
from multiprocessing import shared_memory

from .log import get_logger

log = get_logger('SHM')

# 槽位按全局最大帧分配（沿用旧池 MAX_FRAME_BYTES 口径），保持对 ≤720p 输入的容忍
MAX_FRAME_BYTES = 720 * 1280 * 3
SLOTS_PER_USER = 4


class SlotSet:
    """单个体测会话的槽位集合；父进程持有句柄保证 Windows 下名字存活，destroy 时 unlink。"""

    def __init__(self, uid, count=SLOTS_PER_USER):
        self.names = []
        self._shms = []
        for i in range(count):
            name = f"sport_{uid}_{i}"
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=MAX_FRAME_BYTES)
            except FileExistsError:
                # 上次异常退出遗留（POSIX 语义下名字残留）：清掉重建
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(name=name, create=True, size=MAX_FRAME_BYTES)
            self._shms.append(shm)
            self.names.append(name)

    def destroy(self):
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                log.error("销毁共享内存槽失败: %s", e)
        self._shms.clear()
        self.names.clear()


def attach_slot(name, shape):
    """attach 一个槽，返回 (SharedMemory, ndarray)；调用方用完 close。"""
    shm = shared_memory.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    return shm, arr
