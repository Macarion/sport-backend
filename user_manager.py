import asyncio
import numpy as np
from multiprocessing.managers import BaseManager
from shared_memory_pool import SharedMemoryPool
from config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, MANAGER_HOST, MANAGER_PORT, AUTH_KEY, get_logger

log = get_logger('UserManager')

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue')
QueueManager.register('get_command_queue')
QueueManager.register('get_result_queue')
QueueManager.register('get_shm_pool')


def get_manager():
    manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
    manager.connect()
    return manager


class UserManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self._sessions = {}
        self._consumers = {}
        self._loop = None
        self._ipc = None
        self._result_queue = None
        self._shm_pool = None
        self._cmd_queue = None
        self._result_task = None

    def _ensure_ipc(self):
        if self._ipc is None:
            self._ipc = get_manager()
            self._result_queue = self._ipc.get_result_queue()
            self._shm_pool = self._ipc.get_shm_pool()
            self._cmd_queue = self._ipc.get_command_queue()

    def start_sport_test(self, uid, project="situp"):
        self._ensure_ipc()
        task_queue = self._ipc.get_task_queue(project)

        self._sessions[uid] = {
            "project": project,
            "task_queue": task_queue,
            "frame_id": 0,
            "config": {},
        }

        self._cmd_queue.put(('register', project, 1))
        log.info(f"用户开始测试: uid={uid}, project={project}")
        return True

    def stop_sport_test(self, uid):
        session = self._sessions.pop(uid, None)
        if not session:
            return

        session["task_queue"].put((None, uid, -1, session.get("config", {})))
        self._cmd_queue.put(('unregister', session["project"], -1))
        log.info(f"用户停止测试: uid={uid}")

    async def register_consumer(self, consumer):
        self._ensure_ipc()
        uid = str(consumer.uid)
        self._consumers[uid] = consumer

        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        if self._result_task is None or self._result_task.done():
            self._result_task = asyncio.ensure_future(self._poll_results())

    def unregister_consumer(self, uid):
        self._consumers.pop(str(uid), None)

    def push_frame(self, uid, frame_bytes, config=None):
        session = self._sessions.get(uid)
        if not session:
            return False

        shm_name = self._shm_pool.acquire_block(timeout=0.05)
        if shm_name is None:
            log.warning(f"共享内存池已满，丢弃帧 uid={uid}")
            return False

        shm, arr = SharedMemoryPool.attach(shm_name, SHAPE)
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(SHAPE)
        arr[:] = frame
        shm.close()

        frame_id = session["frame_id"]
        session["frame_id"] += 1
        session["task_queue"].put((shm_name, uid, frame_id, config or session.get("config", {})))
        return True

    def _read_and_release_frame(self, shm_name):
        shm, arr = SharedMemoryPool.attach(shm_name, SHAPE)
        frame = arr.copy() if arr is not None else None
        shm.close()
        self._shm_pool.release_block(shm_name)
        return frame

    async def _poll_results(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                result = await loop.run_in_executor(
                    None, lambda: self._result_queue.get(timeout=0.05)
                )
                if result is None:
                    continue

                shm_name, user_id, eval_data = result
                uid_str = str(user_id)

                consumer = self._consumers.get(uid_str)
                if consumer is None:
                    if shm_name:
                        self._shm_pool.release_block(shm_name)
                    continue

                if shm_name:
                    frame = await loop.run_in_executor(
                        None, lambda: self._read_and_release_frame(shm_name)
                    )
                    if frame is not None:
                        consumer.output_track.push_frame(frame)

                if eval_data is not None:
                    await consumer.channel_layer.send(consumer.channel_name, {
                        "type": "eval_result",
                        "data": eval_data,
                    })
                else:
                    await consumer.channel_layer.send(consumer.channel_name, {
                        "type": "test_end",
                    })
                    self._consumers.pop(uid_str, None)

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.01)


user_manager = UserManager()
