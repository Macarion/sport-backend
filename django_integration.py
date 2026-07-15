# django_integration.py
import os
import numpy as np
from multiprocessing.managers import BaseManager
from config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, FRAME_BYTES, get_logger
from shared_memory_pool import SharedMemoryPool

log = get_logger('WebSocket')

MANAGER_HOST = os.environ.get('MANAGER_HOST', '127.0.0.1')
MANAGER_PORT = int(os.environ.get('MANAGER_PORT', 50000))
AUTH_KEY = bytes.fromhex(os.environ.get('AUTH_KEY', '626f6479746573745f7365637265745f6b6579'))

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue')
QueueManager.register('get_command_queue')
QueueManager.register('get_send_queue')
QueueManager.register('get_shm_pool')


def get_manager():
    manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
    manager.connect()
    return manager


class VideoConsumer:
    def __init__(self):
        self.manager = None
        self.cmd_queue = None
        self.task_queue = None
        self.send_queue = None
        self.shm_pool = None
        self.user_id = None
        self.project = None

    def connect(self, user_id, project):
        self.user_id = user_id
        self.project = project

        self.manager = get_manager()
        self.cmd_queue = self.manager.get_command_queue()
        self.cmd_queue.put(('register', self.project, 1))

        self.task_queue = self.manager.get_task_queue(self.project)
        self.send_queue = self.manager.get_send_queue()
        self.shm_pool = self.manager.get_shm_pool()

    def receive_frame(self, frame_bytes):
        if self.task_queue is None or self.shm_pool is None:
            return False

        shm_name = self.shm_pool.acquire_block(timeout=0.1)
        if shm_name is None:
            log.warning(f"共享内存池已满，丢弃帧 user={self.user_id}")
            return False

        shm, arr = SharedMemoryPool.attach(shm_name, SHAPE)
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(SHAPE)
        arr[:] = frame
        shm.close()

        self.task_queue.put((shm_name, self.user_id, 0))
        return True

    def get_result(self, timeout=0.01):
        if self.send_queue is None:
            return None
        try:
            return self.send_queue.get(timeout=timeout)
        except:
            return None

    def read_result_frame(self, shm_name):
        shm, arr = SharedMemoryPool.attach(shm_name, SHAPE)
        frame = arr.copy()
        shm.close()
        return frame

    def release_frame(self, shm_name):
        if self.shm_pool is not None:
            self.shm_pool.release_block(shm_name)

    def disconnect(self):
        if self.manager:
            self.cmd_queue.put(('unregister', self.project, -1))
            self.task_queue.put((None, self.user_id, -1))
