# shared_memory_pool.py
import os
import time
import numpy as np
from multiprocessing import shared_memory, Lock
from .config import SHM_POOL_SIZE, SHM_RECLAIM_TIMEOUT, MAX_FRAME_BYTES, MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT, MAX_FRAME_CHANNELS, get_logger

log = get_logger('SHM')

SHM_NAMESPACE = 'bodytest'


class SharedMemoryPool:
    def __init__(self, namespace=None):
        self.lock = Lock()
        self.blocks = []
        self.in_use = [False] * SHM_POOL_SIZE
        self.acquire_times = [0.0] * SHM_POOL_SIZE
        self.shape = (MAX_FRAME_HEIGHT, MAX_FRAME_WIDTH, MAX_FRAME_CHANNELS)

        pid = namespace or str(os.getpid())
        log.info("正在初始化 %d 个共享内存块 (namespace=%s)...", SHM_POOL_SIZE, pid)
        for i in range(SHM_POOL_SIZE):
            name = f"frame_{pid}_{i}"
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=MAX_FRAME_BYTES)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(name=name, create=True, size=MAX_FRAME_BYTES)

            shm.buf[:MAX_FRAME_BYTES] = b'\x00' * MAX_FRAME_BYTES
            self.blocks.append({
                'name': name,
                'shm': shm,
                'size': MAX_FRAME_BYTES
            })
        log.info("初始化完成")

    def acquire_block(self, timeout=1.0):
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                self._reclaim_expired()
                for i, used in enumerate(self.in_use):
                    if not used:
                        self.in_use[i] = True
                        self.acquire_times[i] = time.time()
                        return self.blocks[i]['name']
            time.sleep(0.001)
        return None

    def release_block(self, shm_name):
        with self.lock:
            for i, block in enumerate(self.blocks):
                if block['name'] == shm_name:
                    self.in_use[i] = False
                    self.acquire_times[i] = 0.0
                    return True
        return False

    def get_array(self, shm_name):
        for block in self.blocks:
            if block['name'] == shm_name:
                return np.ndarray(self.shape, dtype=np.uint8, buffer=block['shm'].buf)
        return None

    @staticmethod
    def attach(shm_name, shape):
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        return shm, arr

    def _reclaim_expired(self):
        now = time.time()
        for i in range(SHM_POOL_SIZE):
            if self.in_use[i] and self.acquire_times[i] > 0:
                if now - self.acquire_times[i] > SHM_RECLAIM_TIMEOUT:
                    log.warning("回收超时块: %s", self.blocks[i]['name'])
                    self.in_use[i] = False
                    self.acquire_times[i] = 0.0
                    self.blocks[i]['shm'].buf[:MAX_FRAME_BYTES] = b'\x00' * MAX_FRAME_BYTES

    def cleanup(self):
        log.info("正在清理共享内存...")
        for block in self.blocks:
            try:
                block['shm'].close()
                block['shm'].unlink()
            except Exception as e:
                log.error("清理 %s 时出错: %s", block['name'], e)
