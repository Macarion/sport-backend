# evaluation_worker.py
import os
import sys
import time
import queue
from collections import defaultdict
from config import FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, get_logger
from shared_memory_pool import SharedMemoryPool

log = get_logger('Evaluation')

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def run_evaluation_worker(result_queue, send_to_ws_queue, shm_pool):
    log.info(f"进程启动，PID={os.getpid()}")

    user_sessions = defaultdict(lambda: {'count': 0, 'last_pose': None})

    while True:
        try:
            shm_name, user_id, keypoints_list = result_queue.get(timeout=1)

            shm, frame = SharedMemoryPool.attach(shm_name, SHAPE)
            if frame is None:
                shm_pool.release_block(shm_name)
                continue

            new_count = 10  # 模拟计数，后续替换为实际计数逻辑
            shm.close()

            send_to_ws_queue.put((shm_name, user_id, {'count': new_count}))

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"错误: {e}")
            time.sleep(1)

    log.info("进程退出")
